"""Test fixtures: a minimal b2crunner-shaped subject file.

This is NOT b2crunner's writer (that is b2cgltf/b2crunner/). It writes just enough of SPEC 4 (scene 0, the three
roots, splat, skinned body, B2C_mhr and B2C_orbit stubs) for the core and b2crig tests, and for b2crig to try its
enhancement on real subjects before the real writer exists. `W` is a parameter so the frame rule is tested with a
non-identity root (SPEC 8).
"""
from __future__ import annotations

import numpy as np

from b2cgltf import splat as S
from b2cgltf.b2crunner.convert import joint_nodes
from b2cgltf.document import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, UNSIGNED_BYTE, UNSIGNED_INT, UNSIGNED_SHORT, Document

W_TEST = np.array([[0.0, 0.0, 1.0, 1.0],     # 90 degrees about +Y, then a translation
                   [0.0, 1.0, 0.0, 2.0],
                   [-1.0, 0.0, 0.0, 3.0],
                   [0.0, 0.0, 0.0, 1.0]])


def write_subject(path, *, splat_fields: dict, body_verts, body_faces, joint_idx, joint_w, joint_pos, joint_rot,
                  parents, W=np.eye(4), joint_names=None, final_cameras: dict | None = None, mhr: dict | None = None):
    """joint_idx/joint_w [nv, 4s]: body skin sets; joint_pos [J, 3], joint_rot [J, 3, 3]: bind pose (b2crunner frame)."""
    doc, w = Document.new("b2crunner", tool="b2cgltf tests/fixtures.py")
    J = len(parents)
    jn = joint_nodes(joint_pos, joint_rot, parents)
    Wl = np.asarray(W, np.float64).T.reshape(-1).tolist()
    world = w.add_node({"name": "b2c_world", "matrix": Wl, "children": []})
    at = S.attributes(w, splat_fields)
    sm = w.add_mesh({"name": "b2c_splat", "primitives": [S.primitive(at)]})
    sn = w.add_node({"name": "b2c_splat", "mesh": sm})
    doc.json["nodes"][world]["children"].append(sn)
    skel = w.add_node({"name": "b2c_skeleton", "matrix": Wl, "children": []})
    names = joint_names or [f"b2c_joint_{j:03d}" for j in range(J)]
    jnodes = [w.add_node({"name": names[j], "rotation": jn["rotation"][j].tolist(), "translation": jn["translation"][j].tolist()},
                         joint=True) for j in range(J)]
    for j, p in enumerate(parents):
        (doc.json["nodes"][skel] if p < 0 else doc.json["nodes"][jnodes[p]]).setdefault("children", []).append(jnodes[j])
    skin = w.add_skin({"name": "b2c_body", "joints": jnodes, "skeleton": skel,
                       "inverseBindMatrices": w.add_accessor(jn["inverse_bind"], "MAT4")})
    ct = UNSIGNED_BYTE if J < 256 else UNSIGNED_SHORT
    battr = {"POSITION": w.add_accessor(np.asarray(body_verts, np.float32), "VEC3", target=ARRAY_BUFFER, minmax=True)}
    ji, jw = np.asarray(joint_idx), np.asarray(joint_w, np.float32)
    for s in range(ji.shape[1] // 4):
        battr[f"JOINTS_{s}"] = w.add_accessor(ji[:, 4 * s:4 * s + 4].astype(np.uint8 if J < 256 else np.uint16), "VEC4", component=ct, target=ARRAY_BUFFER)
        battr[f"WEIGHTS_{s}"] = w.add_accessor(jw[:, 4 * s:4 * s + 4], "VEC4", target=ARRAY_BUFFER)
    bprim = {"attributes": battr, "mode": 4,
             "indices": w.add_accessor(np.asarray(body_faces, np.uint32).reshape(-1), "SCALAR", component=UNSIGNED_INT, target=ELEMENT_ARRAY_BUFFER)}
    bm = w.add_mesh({"name": "b2c_body", "primitives": [bprim]})
    mhr_ext = {"version": 1, "model": {"repo": "test", "file": "none", "sha256": "0" * 64}, "sourceRecordVersion": 2,
               **(mhr or {})}
    bn = w.add_node({"name": "b2c_body", "mesh": bm, "skin": skin, "extensions": {"B2C_mhr": mhr_ext}})
    w.add_scene({"name": "b2c", "nodes": [world, skel, bn]})
    doc.json["scene"] = 0
    fc = final_cameras or {"rotation": np.tile(np.eye(3), (1, 1, 1)), "position": np.array([[0, 1, 3.0]]),
                           "intrinsics": np.array([[500, 500, 256, 256.0]]), "image_size": [512, 512], "names": ["0000.png"]}
    orbit = {"version": 1, "migrated": True, "final_cameras": {
        "rotation": w.add_accessor(np.asarray(fc["rotation"], np.float32), "MAT3"),
        "position": w.add_accessor(np.asarray(fc["position"], np.float32), "VEC3"),
        "intrinsics": w.add_accessor(np.asarray(fc["intrinsics"], np.float32), "VEC4"),
        "image_size": list(fc["image_size"]), "names": list(fc["names"])}}
    w.set_extension("B2C_orbit", orbit)
    w.save(path)
    return doc


def synthetic(path, W=W_TEST, n_splats: int = 60, seed: int = 0):
    """A 3-joint chain along +Y with a skinned column as the body and a splat around it."""
    rng = np.random.default_rng(seed)
    J = 3
    parents = np.array([-1, 0, 1])
    joint_pos = np.array([[0, 0, 0], [0, 0.5, 0], [0, 1.0, 0]], np.float64)
    joint_rot = np.tile(np.eye(3), (J, 1, 1))
    # body: a square column of 5 rings x 4 vertices, y = 0..1
    ring = np.array([[0.1, 0, 0.1], [0.1, 0, -0.1], [-0.1, 0, -0.1], [-0.1, 0, 0.1]])
    V = np.concatenate([ring + [0, y, 0] for y in np.linspace(0, 1, 5)])
    F = []
    for r in range(4):
        for k in range(4):
            a, b, c, d = r * 4 + k, r * 4 + (k + 1) % 4, (r + 1) * 4 + k, (r + 1) * 4 + (k + 1) % 4
            F += [(a, b, c), (b, d, c)]
    F = np.array(F)
    y = V[:, 1]
    w1 = np.clip(1 - np.abs(y - 0.5) / 0.5, 0, 1); w0 = np.clip(0.5 - y, 0, 1) * 2; w2 = np.clip(y - 0.5, 0, 1) * 2
    Wt = np.stack([w0, w1, w2], 1); Wt /= Wt.sum(1, keepdims=True)
    order = np.argsort(-Wt, 1)
    ji = np.zeros((len(V), 4), np.uint8); jw = np.zeros((len(V), 4), np.float32)
    ji[:, :3] = order; jw[:, :3] = np.take_along_axis(Wt, order, 1)
    ji[jw == 0] = 0
    # splat: points near the column surface
    fi = rng.integers(0, len(F), n_splats)
    b = rng.dirichlet([1, 1, 1], n_splats)
    P = (V[F[fi]] * b[:, :, None]).sum(1) + rng.normal(0, 0.005, (n_splats, 3))
    q = rng.normal(size=(n_splats, 4))
    fields = {"x": P[:, 0], "y": P[:, 1], "z": P[:, 2], "opacity": rng.normal(0, 2, n_splats),
              **{f"scale_{k}": rng.normal(-4, 0.5, n_splats) for k in range(3)},
              **{f"rot_{k}": q[:, k] for k in range(4)},
              **{f"f_dc_{k}": rng.normal(0, 0.5, n_splats) for k in range(3)},
              **{f"f_rest_{k}": rng.normal(0, 0.1, n_splats) for k in range(45)},
              "seg_label": rng.integers(0, 29, n_splats).astype(np.float32), "seg_conf": rng.uniform(0.3, 1, n_splats)}
    fields = {k: np.asarray(v, np.float32) for k, v in fields.items()}
    doc = write_subject(path, splat_fields=fields, body_verts=V, body_faces=F, joint_idx=ji, joint_w=jw,
                        joint_pos=joint_pos, joint_rot=joint_rot, parents=parents, W=W)
    return doc, {"fields": fields, "V": V, "F": F, "Wt": Wt, "parents": parents, "joint_pos": joint_pos, "fi": fi, "b": b}
