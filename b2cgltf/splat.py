"""The splat encoding of SPEC 4.2, shared by both writers: b2crunner's delivered splat and a splat b2crig retrains.

`attributes(w, fields)` takes the trainer PLY's per-vertex properties (a dict of numpy arrays, or a structured
array) and appends the `KHR_gaussian_splatting` accessors plus `COLOR_0` and the seg attributes; it returns the
primitive's `attributes` dict. `primitive(...)` wraps it with the extension object.
"""
from __future__ import annotations

import numpy as np

from .document import ARRAY_BUFFER, FLOAT, UNSIGNED_BYTE, RuleError, Writer

KHR = "KHR_gaussian_splatting"
KHR_OBJECT = {"kernel": "ellipse", "colorSpace": "srgb_rec709_display"}
SH_PER_DEGREE = (1, 3, 5, 7)
SH_C0 = 0.2820948
DROPPED = ("ev_", "open_", "cage_fill", "cage_gate_")      # b2crunner drops these (the open family is b2crig's, 5.3)
KNOWN = {"x", "y", "z", "nx", "ny", "nz", "opacity", "seg_label", "seg_conf"} | {f"scale_{k}" for k in range(3)} | \
    {f"rot_{k}" for k in range(4)} | {f"f_dc_{k}" for k in range(3)} | {f"f_rest_{k}" for k in range(45)}


def _fields(fields) -> dict:
    if isinstance(fields, np.ndarray) and fields.dtype.names:
        return {k: fields[k] for k in fields.dtype.names}
    return dict(fields)


def attributes(w: Writer, fields, *, drop_unknown: bool = False, log=print) -> dict:
    f = _fields(fields)
    dropped = sorted(k for k in f if k.startswith(DROPPED))
    unknown = sorted(k for k in f if k not in KNOWN and not k.startswith(DROPPED))
    if unknown and not drop_unknown:
        raise RuleError(f"unknown per-vertex properties {unknown}: refusing instead of dropping them silently (SPEC 4.2)")
    if dropped or unknown:
        log(f"b2cgltf.splat: dropped {dropped + unknown}")
    n = len(f["x"])
    pos = np.stack([f["x"], f["y"], f["z"]], 1).astype(np.float32)
    q = np.stack([f[f"rot_{k}"] for k in range(4)], 1).astype(np.float64)
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    if not np.all(np.isfinite(qn)) or np.any(qn == 0):
        raise RuleError(f"{int(np.sum(~np.isfinite(qn) | (qn == 0)))} splats have a zero or non-finite rotation (SPEC 4.2)")
    q /= qn
    rot = np.concatenate([q[:, 1:], q[:, :1]], 1).astype(np.float32)          # w x y z -> x y z w
    scale = np.exp(np.stack([f[f"scale_{k}"] for k in range(3)], 1).astype(np.float64)).astype(np.float32)
    op = (1 / (1 + np.exp(-np.asarray(f["opacity"], np.float64)))).astype(np.float32)
    if np.any(op == 0) or np.any(op == 1):
        raise RuleError(f"{int(np.sum((op == 0) | (op == 1)))} splats have a float32 opacity of exactly 0 or 1 (SPEC 4.2)")
    dc = np.stack([f[f"f_dc_{c}"] for c in range(3)], 1).astype(np.float32)
    n_rest = sum(1 for k in f if k.startswith("f_rest_"))
    if n_rest not in (0, 9, 24, 45) or any(f"f_rest_{i}" not in f for i in range(n_rest)):
        raise RuleError(f"f_rest_0..{n_rest - 1} is not a complete SH layout (0, 9, 24 or 45 properties) (SPEC 4.2)")
    K = n_rest // 3 + 1
    degree = int(round(np.sqrt(K))) - 1
    add = lambda a, t: w.add_accessor(a, t, target=ARRAY_BUFFER)
    at = {"POSITION": w.add_accessor(pos, "VEC3", target=ARRAY_BUFFER, minmax=True),
          f"{KHR}:ROTATION": add(rot, "VEC4"), f"{KHR}:SCALE": add(scale, "VEC3"), f"{KHR}:OPACITY": add(op, "SCALAR"),
          f"{KHR}:SH_DEGREE_0_COEF_0": add(dc, "VEC3")}
    k = 0
    for d in range(1, degree + 1):
        for c in range(SH_PER_DEGREE[d]):   # f_rest is channel-major: K-1 R, then G, then B
            at[f"{KHR}:SH_DEGREE_{d}_COEF_{c}"] = add(np.stack([f[f"f_rest_{ch * (K - 1) + k}"] for ch in range(3)], 1).astype(np.float32), "VEC3")
            k += 1
    at["COLOR_0"] = add(np.clip(0.5 + SH_C0 * dc, 0, 1).astype(np.float32), "VEC3")
    if "seg_label" in f:
        if "seg_conf" not in f:
            raise RuleError("seg_label without seg_conf (SPEC 4.2)")
        lab = np.asarray(f["seg_label"], np.float64)
        if np.any(lab != np.round(lab)) or np.any(lab < 0) or np.any(lab > 255):
            raise RuleError("seg_label must be integers in 0..255 (SPEC 4.2)")
        at["_SEG_LABEL"] = w.add_accessor(lab.astype(np.uint8), "SCALAR", component=UNSIGNED_BYTE, target=ARRAY_BUFFER)
        at["_SEG_CONF"] = add(np.asarray(f["seg_conf"], np.float32), "SCALAR")
    return at


def primitive(attrs: dict, extensions: dict | None = None) -> dict:
    return {"attributes": attrs, "mode": 0, "extensions": {KHR: dict(KHR_OBJECT), **(extensions or {})}}


def splat_attribute_keys(attrs: dict) -> dict:
    """The attributes that describe the Gaussians themselves (what a primitive reusing another's splat copies)."""
    return {k: v for k, v in attrs.items() if k == "POSITION" or k.startswith(f"{KHR}:") or k == "COLOR_0"}
