"""Reading subject and clip files: the current splat (SPEC 7.1), the skeleton, and posing in the b2crunner frame
(SPEC 7.2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .document import Document

KHR = "KHR_gaussian_splatting"
SH_PER_DEGREE = (1, 3, 5, 7)


# ---------------------------------------------------------------- transforms

def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """[..., 4] unit quaternions (x, y, z, w) -> [..., 3, 3]."""
    q = np.asarray(q, np.float64)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """[..., 3, 3] rotations -> [..., 4] (x, y, z, w), w >= 0."""
    R = np.asarray(R, np.float64)
    m = R.reshape(-1, 3, 3)
    out = np.empty((len(m), 4))
    for i, r in enumerate(m):
        t = np.trace(r)
        if t > 0:
            s = 2 * np.sqrt(1 + t); out[i] = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, s / 4]
        else:
            k = int(np.argmax(np.diag(r))); a, b = (k + 1) % 3, (k + 2) % 3
            s = 2 * np.sqrt(1 + r[k, k] - r[a, a] - r[b, b])
            v = np.empty(4); v[k] = s / 4; v[a] = (r[a, k] + r[k, a]) / s; v[b] = (r[b, k] + r[k, b]) / s; v[3] = (r[b, a] - r[a, b]) / s
            out[i] = v
    out[out[:, 3] < 0] *= -1
    return out.reshape(*R.shape[:-2], 4)


def node_matrix(node: dict) -> np.ndarray:
    """A node's local 4x4 (row-major here; glTF's `matrix` is column-major)."""
    if "matrix" in node:
        return np.asarray(node["matrix"], np.float64).reshape(4, 4).T
    M = np.eye(4)
    M[:3, :3] = quat_to_mat(node.get("rotation", [0, 0, 0, 1])) * np.asarray(node.get("scale", [1, 1, 1]))[None]
    M[:3, 3] = node.get("translation", [0, 0, 0])
    return M


# ---------------------------------------------------------------- skeleton

@dataclass
class Skeleton:
    root: int                 # node index of the skeleton root (b2c_skeleton, or the clip's copy)
    W: np.ndarray             # [4, 4] the root's matrix: b2crunner frame -> glTF
    joints: list[int]         # node index per joint (skin order = MHR joint order)
    parents: np.ndarray       # [J] joint index of the parent, -1 under the root
    rest_q: np.ndarray        # [J, 4] rest local rotations (x y z w)
    rest_t: np.ndarray        # [J, 3] rest local translations
    ibm: np.ndarray           # [J, 4, 4] inverse bind matrices (row-major)
    names: list[str]

    def rel_matrices(self, q: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Local TRS [..., J, 4], [..., J, 3] -> the skinning matrices in the b2crunner frame,
        W^-1 · joint_world · IBM = (joint transform relative to the root) · IBM  [..., J, 4, 4] (SPEC 7.2)."""
        L = np.zeros((*q.shape[:-1], 4, 4))
        L[..., :3, :3] = quat_to_mat(q); L[..., :3, 3] = t; L[..., 3, 3] = 1
        G = np.empty_like(L)
        for j in self.order:
            p = self.parents[j]
            G[..., j, :, :] = L[..., j, :, :] if p < 0 else G[..., p, :, :] @ L[..., j, :, :]
        return G @ self.ibm

    @property
    def order(self) -> list[int]:
        out, seen = [], set()

        def visit(j):
            if j in seen:
                return
            if self.parents[j] >= 0:
                visit(self.parents[j])
            seen.add(j); out.append(j)
        for j in range(len(self.parents)):
            visit(j)
        return out


def skeleton(doc: Document, skin: int = 0) -> Skeleton:
    js = doc.json
    sk = js["skins"][skin]
    joints = list(sk["joints"])
    jof = {n: j for j, n in enumerate(joints)}
    parent_node = {}
    for i, nd in enumerate(js["nodes"]):
        for c in nd.get("children", []):
            parent_node[c] = i
    root = sk.get("skeleton")
    parents = np.array([jof.get(parent_node.get(n), -1) for n in joints])
    if root is None:
        root = parent_node.get(joints[int(np.nonzero(parents < 0)[0][0])])
    for j, n in enumerate(joints):   # every top joint hangs directly under the root
        if parents[j] < 0 and parent_node.get(n) != root:
            raise ValueError(f"joint {n} is not under the skeleton root {root}")
    W = node_matrix(js["nodes"][root]) if root is not None else np.eye(4)
    rest_q = np.array([js["nodes"][n].get("rotation", [0, 0, 0, 1]) for n in joints], np.float64)
    rest_t = np.array([js["nodes"][n].get("translation", [0, 0, 0]) for n in joints], np.float64)
    ibm = doc.accessor(sk["inverseBindMatrices"]).astype(np.float64) if "inverseBindMatrices" in sk else np.tile(np.eye(4), (len(joints), 1, 1))
    return Skeleton(root, W, joints, parents, rest_q, rest_t, ibm, [js["nodes"][n].get("name", "") for n in joints])


def skin_points(V: np.ndarray, idx: np.ndarray, w: np.ndarray, mats: np.ndarray) -> np.ndarray:
    """Linear blend skinning: V [n, 3], idx/w [n, k], mats [..., J, 4, 4] -> [..., n, 3]."""
    V = np.asarray(V, np.float64); w = np.asarray(w, np.float64)
    out = np.zeros((*mats.shape[:-3], len(V), 3))
    for k in range(idx.shape[1]):
        m = mats[..., idx[:, k], :, :]
        out += w[:, k, None] * ((m[..., :3, :3] @ V[..., None])[..., 0] + m[..., :3, 3])
    return out


def joints_weights(doc: Document, prim: dict) -> tuple[np.ndarray, np.ndarray]:
    """Every JOINTS_n / WEIGHTS_n set of a primitive, concatenated: idx [n, 4s], w [n, 4s]."""
    at = prim["attributes"]
    sets = sorted(int(k.split("_")[1]) for k in at if k.startswith("JOINTS_"))
    idx = np.concatenate([doc.accessor(at[f"JOINTS_{s}"]).astype(np.int64) for s in sets], 1)
    w = np.concatenate([doc.accessor(at[f"WEIGHTS_{s}"], normalized=True).astype(np.float64) for s in sets], 1)
    return idx, w


# ---------------------------------------------------------------- the splat

def current_splat(doc: Document) -> int:
    """SPEC 7.1: the splat node the default scene lists (directly or below a listed root) that no other node
    supersedes."""
    js = doc.json
    scene = js["scenes"][js.get("scene", 0)]
    listed, stack = [], list(scene["nodes"])
    while stack:
        n = stack.pop(0); listed.append(n); stack += js["nodes"][n].get("children", [])
    superseded = {nd.get("extras", {}).get("supersedes") for nd in js["nodes"]} - {None}
    def is_splat(n):
        m = js["nodes"][n].get("mesh")
        return m is not None and any(KHR in (p.get("extensions") or {}) for p in js["meshes"][m]["primitives"])
    cand = [n for n in listed if is_splat(n) and n not in superseded]
    if len(cand) != 1:
        raise ValueError(f"the default scene has {len(cand)} current splats: {cand}")
    return cand[0]


def splat(doc: Document, node: int | None = None) -> dict:
    """The splat's arrays in display form: pos, rot (x y z w), scale (linear), opacity (0..1), sh [n, K, 3]
    (coefficient-major: DC first), seg_label / seg_conf when present."""
    js = doc.json
    node = current_splat(doc) if node is None else node
    prim = js["meshes"][js["nodes"][node]["mesh"]]["primitives"][0]
    at = prim["attributes"]
    g = lambda k: doc.accessor(at[k])
    coefs = [g(f"{KHR}:SH_DEGREE_0_COEF_0")]
    for d in (1, 2, 3):
        if f"{KHR}:SH_DEGREE_{d}_COEF_0" not in at:
            break
        coefs += [g(f"{KHR}:SH_DEGREE_{d}_COEF_{c}") for c in range(SH_PER_DEGREE[d])]
    out = {"node": node, "primitive": prim, "pos": g("POSITION"), "rot": g(f"{KHR}:ROTATION"), "scale": g(f"{KHR}:SCALE"),
           "opacity": g(f"{KHR}:OPACITY"), "sh": np.stack(coefs, 1)}
    if "_SEG_LABEL" in at:
        out["seg_label"] = g("_SEG_LABEL"); out["seg_conf"] = g("_SEG_CONF")
    return out


def to_trainer_ply_fields(s: dict) -> dict:
    """The splat as the trainer PLY's properties (SPEC 4.2 reversed): log scales, logit opacity, w x y z."""
    op = s["opacity"].astype(np.float64)
    K = s["sh"].shape[1]
    f = {"x": s["pos"][:, 0], "y": s["pos"][:, 1], "z": s["pos"][:, 2],
         "scale_0": np.log(s["scale"][:, 0]), "scale_1": np.log(s["scale"][:, 1]), "scale_2": np.log(s["scale"][:, 2]),
         "opacity": np.log(op / (1 - op)), "rot_0": s["rot"][:, 3], "rot_1": s["rot"][:, 0], "rot_2": s["rot"][:, 1],
         "rot_3": s["rot"][:, 2], "f_dc_0": s["sh"][:, 0, 0], "f_dc_1": s["sh"][:, 0, 1], "f_dc_2": s["sh"][:, 0, 2]}
    for c in range(3):
        for k in range(1, K):
            f[f"f_rest_{c * (K - 1) + k - 1}"] = s["sh"][:, k, c]
    if "seg_label" in s:
        f["seg_label"] = s["seg_label"].astype(np.float32); f["seg_conf"] = s["seg_conf"]
    return {k: np.asarray(v, np.float32) for k, v in f.items()}
