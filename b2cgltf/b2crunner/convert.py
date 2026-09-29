"""b2crunner's inputs -> the arrays SPEC 4.2/4.3 stores, as numpy only (no glTF here; subject.py writes them).

- the trainer PLY (3DGS layout) -> `KHR_gaussian_splatting` attributes + `_SEG_*` (SPEC 4.2), with the writer's
  refusals and drops;
- MHR's sparse skin weights -> `JOINTS_n`/`WEIGHTS_n` sets (SPEC 4.3);
- the fitted joints (world positions + rotations) -> joint-node TRS and inverse bind matrices (SPEC 4.3);
- the body record's version-1 rotations -> version 2 (SPEC 4.6).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814

#: per-vertex properties the writer drops (SPEC 4.2): debug evidence, and b2crig-owned open/cage state
DROPPED_PREFIXES = ("ev_", "open_", "cage_fill", "cage_gate_")

_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
              "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def read_trainer_ply(path: str | Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """The vertex element of a binary little-endian PLY as {property: array}, plus its header comments (without
    the `comment ` prefix). Refuses anything but one `vertex` element of scalar properties."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY")
        comments, fields, count, fmt = [], [], None, None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            words = line.decode("ascii").rstrip("\r\n").split(" ")
            if words[0] == "end_header":
                break
            if words[0] == "format":
                fmt = words[1]
            elif words[0] == "comment":
                comments.append(" ".join(words[1:]))
            elif words[0] == "element":
                if count is not None or words[1] != "vertex":
                    raise ValueError(f"{path}: expected one 'vertex' element, got {words[1]!r}")
                count = int(words[2])
            elif words[0] == "property":
                if words[1] == "list":
                    raise ValueError(f"{path}: list property {words[-1]!r} in a splat PLY")
                fields.append((words[2], "<" + _PLY_TYPES[words[1]]))
        if fmt != "binary_little_endian" or count is None:
            raise ValueError(f"{path}: need a binary_little_endian vertex element")
        data = np.fromfile(f, dtype=np.dtype(fields), count=count)
    if len(data) != count:
        raise ValueError(f"{path}: {len(data)} of {count} vertices")
    return {name: np.ascontiguousarray(data[name]) for name, _ in fields}, comments


def splat_attributes(props: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[str]]:
    """SPEC 4.2: {glTF attribute: array} and the dropped property names.

    Refuses: an unknown property, a float32 opacity of exactly 0 or 1, a `seg_label` that is not an integer in
    0-255, a zero quaternion, a partial SH degree."""
    names = set(props)
    dropped = sorted(n for n in names if n.startswith(DROPPED_PREFIXES))
    base = {"x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2", "opacity",
            "f_dc_0", "f_dc_1", "f_dc_2"}
    missing = base - names
    if missing:
        raise ValueError(f"splat PLY lacks {sorted(missing)}")
    n_rest = sum(1 for n in names if n.startswith("f_rest_"))
    degree = {0: 0, 9: 1, 24: 2, 45: 3}.get(n_rest)
    if degree is None or any(f"f_rest_{i}" not in names for i in range(n_rest)):
        raise ValueError(f"splat PLY has {n_rest} f_rest_* properties; not a whole SH degree")
    known = base | {f"f_rest_{i}" for i in range(n_rest)} | {"seg_label", "seg_conf"} | set(dropped)
    known |= {"nx", "ny", "nz"} & names   # exporters' zero normals: not data
    unknown = names - known
    if unknown:
        raise ValueError(f"splat PLY has properties SPEC 4.2 does not place: {sorted(unknown)}")
    dropped += sorted({"nx", "ny", "nz"} & names)

    f32 = lambda *keys: np.stack([props[k].astype(np.float32) for k in keys], axis=1)
    out: dict[str, np.ndarray] = {}
    out["POSITION"] = f32("x", "y", "z")

    wxyz = f32("rot_0", "rot_1", "rot_2", "rot_3").astype(np.float64)
    norm = np.linalg.norm(wxyz, axis=1, keepdims=True)
    if not np.all(norm > 0):
        raise ValueError(f"{int(np.sum(norm == 0))} splat(s) with a zero quaternion")
    out["KHR_gaussian_splatting:ROTATION"] = (wxyz / norm)[:, [1, 2, 3, 0]].astype(np.float32)

    out["KHR_gaussian_splatting:SCALE"] = np.exp(f32("scale_0", "scale_1", "scale_2").astype(np.float64)
                                                 ).astype(np.float32)

    opacity = (1.0 / (1.0 + np.exp(-props["opacity"].astype(np.float64)))).astype(np.float32)
    bad = (opacity == 0.0) | (opacity == 1.0)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} splat(s) with opacity exactly 0 or 1 in float32 (logit range "
                         f"{props['opacity'].min():.3g}..{props['opacity'].max():.3g}); refusing, not clamping")
    out["KHR_gaussian_splatting:OPACITY"] = opacity

    dc = f32("f_dc_0", "f_dc_1", "f_dc_2")
    out["KHR_gaussian_splatting:SH_DEGREE_0_COEF_0"] = dc
    if n_rest:
        per_channel = n_rest // 3   # channel-major: coefficients of R, then G, then B
        rest = np.stack([props[f"f_rest_{i}"].astype(np.float32) for i in range(n_rest)], axis=1)
        rest = rest.reshape(-1, 3, per_channel)   # [N, channel, coefficient]
        k = 0
        for l in range(1, degree + 1):
            for m in range(2 * l + 1):
                out[f"KHR_gaussian_splatting:SH_DEGREE_{l}_COEF_{m}"] = np.ascontiguousarray(rest[:, :, k])
                k += 1

    out["COLOR_0"] = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0).astype(np.float32)

    if "seg_label" in names:
        label = props["seg_label"]
        if not (np.all(label == np.round(label)) and label.min() >= 0 and label.max() <= 255):
            raise ValueError("seg_label is not an integer class id in 0-255")
        out["_SEG_LABEL"] = label.astype(np.uint8)
    if "seg_conf" in names:
        out["_SEG_CONF"] = props["seg_conf"].astype(np.float32)
    return out, dropped


def skin_sets(skin_vertex: np.ndarray, skin_joint: np.ndarray, skin_weight: np.ndarray, n_verts: int,
              n_joints: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """MHR's sparse (vertex, joint, weight) triplets -> [(JOINTS_n, WEIGHTS_n)], untruncated (SPEC 4.3).

    Within a vertex, influences are sorted by weight (largest first); an unused slot has weight 0 and joint 0.
    Refuses a vertex without influences or whose weights do not sum to 1 (within float32): renormalising would
    change the body b2crig's cage is made of."""
    v = np.asarray(skin_vertex, np.int64)
    j = np.asarray(skin_joint, np.int64)
    w = np.asarray(skin_weight, np.float64)
    if v.min() < 0 or v.max() >= n_verts or j.min() < 0 or j.max() >= n_joints:
        raise ValueError("skin indices out of range")
    keep = w != 0
    v, j, w = v[keep], j[keep], w[keep]
    counts = np.bincount(v, minlength=n_verts)
    if np.any(counts == 0):
        raise ValueError(f"{int(np.sum(counts == 0))} body vertices have no skin influence")
    sums = np.bincount(v, weights=w, minlength=n_verts)
    if np.max(np.abs(sums - 1.0)) > 1e-6:
        raise ValueError(f"skin weights do not sum to 1 (range {sums.min():.9g}..{sums.max():.9g}); "
                         "refusing to renormalise")
    order = np.lexsort((-w, v))   # by vertex, then weight descending
    v, j, w = v[order], j[order], w[order]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.arange(len(v)) - starts[v]
    n_sets = (int(counts.max()) + 3) // 4
    joints = np.zeros((n_verts, 4 * n_sets), np.uint16 if n_joints > 255 else np.uint8)
    weights = np.zeros((n_verts, 4 * n_sets), np.float32)
    joints[v, slot] = j
    weights[v, slot] = w
    return [(np.ascontiguousarray(joints[:, 4 * s:4 * s + 4]), np.ascontiguousarray(weights[:, 4 * s:4 * s + 4]))
            for s in range(n_sets)]


def global_rots_v1_to_v2(rotation: np.ndarray, global_rots_v1: np.ndarray) -> np.ndarray:
    """SPEC 4.6: a version-1 body record's rotations in the joints' frame (docs/ply-header-records.md)."""
    r = np.asarray(rotation, np.float64).reshape(3, 3)
    return r @ np.diag([1.0, -1.0, -1.0]) @ r.T @ np.asarray(global_rots_v1, np.float64).reshape(-1, 3, 3)


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Rotation matrices [..., 3, 3] -> unit quaternions [..., 4] (x y z w, w >= 0)."""
    m = np.asarray(m, np.float64)
    flat = m.reshape(-1, 3, 3)
    q = np.empty((len(flat), 4))
    for i, r in enumerate(flat):
        t = np.trace(r)
        if t > 0:
            s = 2.0 * np.sqrt(1.0 + t)
            q[i] = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
        else:
            a = int(np.argmax(np.diag(r)))
            b, c = (a + 1) % 3, (a + 2) % 3
            s = 2.0 * np.sqrt(1.0 + r[a, a] - r[b, b] - r[c, c])
            q[i, a] = 0.25 * s
            q[i, b] = (r[b, a] + r[a, b]) / s
            q[i, c] = (r[c, a] + r[a, c]) / s
            q[i, 3] = (r[c, b] - r[b, c]) / s
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 3] < 0] *= -1
    return q.reshape(m.shape[:-2] + (4,))


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(np.asarray(q, np.float64), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def joint_nodes(positions: np.ndarray, rotations: np.ndarray, parents: np.ndarray, tol: float = 1e-4) -> dict:
    """The fitted joints (b2crunner frame) -> glTF joint-node TRS relative to the parent (the root's parent is
    `b2c_skeleton`), and inverse bind matrices relative to `b2c_skeleton` (SPEC 4.3).

    The IBMs invert the joints as a reader composes them from the float32 TRS written here, so bind pose is the
    identity to float64 precision. Refuses a rotation that is not proper (det +1) or not orthonormal to `tol`."""
    p = np.asarray(positions, np.float64).reshape(-1, 3)
    r = np.asarray(rotations, np.float64).reshape(-1, 3, 3)
    parents = np.asarray(parents, np.int64).reshape(-1)
    n = len(parents)
    if len(p) != n or len(r) != n:
        raise ValueError(f"{len(p)} positions, {len(r)} rotations, {n} parents")
    det = np.linalg.det(r)
    ortho = np.abs(np.einsum("nij,nkj->nik", r, r) - np.eye(3)).max(axis=(1, 2))
    if np.any(np.abs(det - 1) > tol) or np.any(ortho > tol):
        worst = int(np.argmax(np.abs(det - 1) + ortho))
        raise ValueError(f"joint {worst}: rotation is not proper/orthonormal (det {det[worst]:.6g}, "
                         f"orthonormality error {ortho[worst]:.3g})")
    for i, par in enumerate(parents):
        if par >= i and par != -1:
            raise ValueError(f"joint {i}'s parent {par} does not precede it; joint order must be topological")

    glob = np.zeros((n, 4, 4))
    glob[:, :3, :3], glob[:, :3, 3], glob[:, 3, 3] = r, p, 1.0
    translation = np.zeros((n, 3), np.float32)
    rotation = np.zeros((n, 4), np.float32)
    composed = np.zeros((n, 4, 4))
    for i, par in enumerate(parents):
        local = glob[i] if par < 0 else np.linalg.inv(glob[par]) @ glob[i]
        translation[i] = local[:3, 3]
        rotation[i] = quat_from_matrix(local[:3, :3])
        m = np.eye(4)
        m[:3, :3] = matrix_from_quat(rotation[i].astype(np.float64))
        m[:3, 3] = translation[i]
        composed[i] = m if par < 0 else composed[par] @ m
    return {"translation": translation, "rotation": rotation,
            "inverse_bind": np.linalg.inv(composed).astype(np.float32),
            "composed_error": float(np.abs(composed - glob).max())}
