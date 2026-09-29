"""Clip files (SPEC 6): one animation of the subject's skeleton plus the cage data that makes it exact.

    write(path, subject_doc, name, fps, q, t, residual=posed - lbs, ...)   # b2crig
    posed = pose(subject_doc, clip_doc)                                     # [T, nv, 3], b2crunner frame (7.2 steps 1-2)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import read
from ..document import SHORT, UNSIGNED_SHORT, Document, RuleError
from . import rig as R

CLIP, RES, GATE, MOTION = "B2CRIG_clip", "B2CRIG_cage_residual", "B2CRIG_open_gate", "B2CRIG_motion"


def lbs(subject: Document, q: np.ndarray, t: np.ndarray, cage: R.Cage | None = None) -> np.ndarray:
    """Plain skinning of the subject's cage by local joint TRS [T, J, 4] / [T, J, 3], in the b2crunner frame."""
    sk = read.skeleton(subject)
    cg = cage or R.cage(subject)
    return read.skin_points(cg.verts, cg.joints, cg.weights, sk.rel_matrices(q, t))


def write(path: str | Path, subject: Document, name: str, fps: float, q: np.ndarray, t: np.ndarray, *,
          residual: np.ndarray | None = None, open_gate_deg: np.ndarray | None = None, motion: dict | None = None,
          subject_uri: str | None = None, tool: str = "b2crig", version: str = "", commit: str = "") -> dict:
    """q [T, J, 4] (x y z w), t [T, J, 3]: local joint TRS per frame, relative to the skeleton root (the b2crunner
    frame). residual [T, nv, 3] (b2crig's posed cage minus lbs(); b2crunner frame). open_gate_deg [T, nv].
    motion: {"body_params": [T,130], "expr": [T,72], optional "root_R" [T,3,3], "root_t" [T,3], "root_c" [3],
    "global_trans" [T,3]}. Returns a report."""
    if subject.id is None:
        raise RuleError("the subject file has no asset.extras.b2c_id (R5)")
    sk = read.skeleton(subject)
    T, J = q.shape[:2]
    if J != len(sk.joints):
        raise ValueError(f"{J} joints, the subject has {len(sk.joints)}")
    cg = R.cage(subject)
    doc, w = Document.new("b2crig", tool=tool, version=version, commit=commit, subject=False,
                          generator="b2cgltf (b2crig clip)")
    # the skeleton copy: root with W, joints with the subject's names, order, parents and rest TRS
    sjs = subject.json
    root = {"name": "b2crig_skeleton", "matrix": sk.W.T.reshape(-1).tolist(), "children": []}
    root_i = w.add_node(root)
    node_of = []
    for j in range(J):
        src = sjs["nodes"][sk.joints[j]]
        nd = {"name": src.get("name", f"joint_{j:03d}"), "rotation": [float(x) for x in sk.rest_q[j]],
              "translation": [float(x) for x in sk.rest_t[j]]}
        node_of.append(w.add_node(nd, joint=True))
    js = doc.json
    for j in range(J):
        p = sk.parents[j]
        (js["nodes"][root_i] if p < 0 else js["nodes"][node_of[p]]).setdefault("children", []).append(node_of[j])
    w.add_scene({"name": "b2crig_clip", "nodes": [root_i]})
    js["scene"] = 0
    # the animation
    times = w.add_accessor((np.arange(T) / fps).astype(np.float32), "SCALAR", minmax=True)
    samplers, channels = [], []
    for j in range(J):
        for path_, arr, typ in (("rotation", q[:, j], "VEC4"), ("translation", t[:, j], "VEC3")):
            samplers.append({"input": times, "output": w.add_accessor(np.asarray(arr, np.float32), typ), "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": node_of[j], "path": path_}})
    anim = {"name": f"b2crig_{name}", "samplers": samplers, "channels": channels, "extensions": {}}
    rep = {"frames": T, "vertices": len(cg.verts)}
    if residual is not None:
        res = np.asarray(residual, np.float64)
        if res.shape != (T, len(cg.verts), 3):
            raise ValueError(f"residual {res.shape}, expected {(T, len(cg.verts), 3)}")
        s = float(np.abs(res).max()) or 1e-9
        qr = np.round(res / s * 32767).clip(-32767, 32767).astype(np.int16)
        rep["residual_quant_max_m"] = float(np.abs(qr / 32767.0 * s - res).max())
        anim["extensions"][RES] = {"version": 1, "accessor": w.add_accessor(qr.reshape(-1, 3), "VEC3", component=SHORT, normalized=True),
                                   "scale": s, "frames": T, "vertices": len(cg.verts)}
    if open_gate_deg is not None:
        g = np.round(np.clip(np.asarray(open_gate_deg, np.float64), 0, 180) / 180 * 65535).astype(np.uint16)
        anim["extensions"][GATE] = {"version": 1, "accessor": w.add_accessor(g.reshape(-1), "SCALAR", component=UNSIGNED_SHORT, normalized=True),
                                    "frames": T, "vertices": len(cg.verts), "rangeDeg": 180}
    if motion is not None:   # each array flat (SCALAR float) with its shape
        m = {"version": 1}
        for k, v in motion.items():
            a = np.asarray(v, np.float32)
            m[k] = {"accessor": w.add_accessor(a.reshape(-1), "SCALAR"), "shape": list(a.shape)}
        anim["extensions"][MOTION] = m
    if not anim["extensions"]:
        del anim["extensions"]
    w.add_animation(anim)
    w.set_extension(CLIP, {"version": 1, "name": name, "fps": float(fps), "skeleton": {"root": root_i, "joints": node_of},
                           "subject": {"id": subject.id, **({"uri": subject_uri} if subject_uri else {})},
                           "rig": {"cageSha256": R.rig_hash(subject), "cageVertexCount": len(cg.verts), "jointCount": J}})
    w.save(path)
    rep["path"] = str(path)
    return rep


def check(subject: Document, clip: Document) -> None:
    """SPEC 6 checks: same subject, same rig, and the clip's skeleton copy equals the subject's."""
    c = clip.extension(CLIP)
    if c is None:
        raise ValueError("not a clip file (no B2CRIG_clip)")
    if c["subject"]["id"] != subject.id:
        raise RuleError(f"the clip belongs to subject {c['subject']['id']}, not {subject.id}")
    if c["rig"]["cageSha256"] != R.rig_hash(subject):
        raise RuleError("the clip was made for another rig (cageSha256 differs)")
    sk, cs = read.skeleton(subject), skeleton_copy(clip)
    if (len(cs["joints"]) != len(sk.joints) or not np.array_equal(cs["parents"], sk.parents)
            or not np.array_equal(cs["rest_q"].astype(np.float32), sk.rest_q.astype(np.float32))
            or not np.array_equal(cs["rest_t"].astype(np.float32), sk.rest_t.astype(np.float32))):
        raise RuleError("the clip's skeleton copy differs from the subject's")


def skeleton_copy(clip: Document) -> dict:
    """The clip's copy of the skeleton: joint node indices in MHR order, parents, rest TRS."""
    js = clip.json
    c = clip.extension(CLIP)["skeleton"]
    joints = c["joints"]
    jof = {n: j for j, n in enumerate(joints)}
    parents = np.full(len(joints), -1)
    for i, nd in enumerate(js["nodes"]):
        for ch in nd.get("children", []):
            if ch in jof and i in jof:
                parents[jof[ch]] = jof[i]
    return {"root": c["root"], "joints": joints, "parents": parents,
            "rest_q": np.array([js["nodes"][n].get("rotation", [0, 0, 0, 1]) for n in joints], np.float64),
            "rest_t": np.array([js["nodes"][n].get("translation", [0, 0, 0]) for n in joints], np.float64)}


def channels(clip: Document) -> tuple[np.ndarray, np.ndarray, float]:
    """The clip's local joint TRS per frame: q [T, J, 4], t [T, J, 3], fps (one key per frame, SPEC 6)."""
    sk = skeleton_copy(clip)
    js = clip.json
    anim = js["animations"][0]
    jof = {n: j for j, n in enumerate(sk["joints"])}
    T = None
    q = t = None
    for ch in anim["channels"]:
        j = jof[ch["target"]["node"]]
        out = clip.accessor(anim["samplers"][ch["sampler"]]["output"]).astype(np.float64)
        if T is None:
            T = len(out)
            q = np.tile(sk["rest_q"], (T, 1, 1)); t = np.tile(sk["rest_t"], (T, 1, 1))
        (q if ch["target"]["path"] == "rotation" else t)[:, j] = out
    return q, t, clip.extension(CLIP)["fps"]


def pose(subject: Document, clip: Document, *, residual: bool = True) -> np.ndarray:
    """SPEC 7.2 steps 1-2: the posed cage per frame [T, nv, 3] in the b2crunner frame."""
    check(subject, clip)
    q, t, _ = channels(clip)
    P = lbs(subject, q, t)
    anim = clip.json["animations"][0]
    e = (anim.get("extensions") or {}).get(RES)
    if residual and e:
        P = P + clip.accessor(e["accessor"], normalized=True).reshape(e["frames"], e["vertices"], 3) * e["scale"]
    return P
