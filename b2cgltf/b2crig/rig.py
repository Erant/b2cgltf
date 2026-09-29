"""b2crig's enhancement of a subject file (SPEC 5): the cage, the rigged splat with b2ctrain's binding, scene 1.

The caller (b2crig) computes the arrays: cage layers, weights, b2ctrain's binding (`b2ctrain render
--export-binding`). This module only lays them out; it knows nothing of MHR or of how b2crig poses.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .. import read, splat as S
from ..document import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, FLOAT, UNSIGNED_BYTE, UNSIGNED_INT, UNSIGNED_SHORT, Document, RuleError, Writer

RIG, BIND, APP = "B2CRIG_rig", "B2CRIG_splat_cage", "B2CRIG_cage_app"
POSING = "b2ctrain cage.cu pose_bound v1"
UNBOUND = 0xFFFFFFFF


@dataclass
class Layer:
    name: str
    classes: list[int]
    faces: np.ndarray                       # [f, 3] local vertex indices
    verts: np.ndarray | None = None         # [v, 3] b2crunner frame; None for layer 0 (the body's own)
    joints: np.ndarray | None = None        # [v, k] skin joint indices (k <= 8); None for layer 0
    weights: np.ndarray | None = None       # [v, k]
    body_index: np.ndarray | None = None    # [v] the body vertex an offset layer copies


@dataclass
class Binding:
    face: np.ndarray                        # [n] int, global cage face, -1 = unbound
    barycentric: np.ndarray                 # [n, 2]
    offset: np.ndarray                      # [n, 3]
    alt: dict | None = None                 # altSplat, altFace, altBarycentric, altOffset, altWeight arrays


@dataclass
class RigResult:
    scene: int
    cage_node: int
    splat_node: int
    layers: list[dict] = field(default_factory=list)


def body_primitive(doc: Document) -> dict:
    js = doc.json
    return js["meshes"][js["nodes"][doc.node_index("b2c_body")]["mesh"]]["primitives"][0]


def _joint_sets(w: Writer, idx: np.ndarray, wt: np.ndarray, njoints: int) -> dict:
    """JOINTS_n / WEIGHTS_n sets of 4, zero-weight slots at joint 0 (validator rule)."""
    k = idx.shape[1]
    pad = (-k) % 4
    idx = np.concatenate([idx, np.zeros((len(idx), pad), idx.dtype)], 1)
    wt = np.concatenate([wt, np.zeros((len(wt), pad), wt.dtype)], 1)
    idx = np.where(wt > 0, idx, 0)
    ct, dt = (UNSIGNED_BYTE, np.uint8) if njoints < 256 else (UNSIGNED_SHORT, np.uint16)
    out = {}
    for s in range(idx.shape[1] // 4):
        out[f"JOINTS_{s}"] = w.add_accessor(idx[:, 4 * s:4 * s + 4].astype(dt), "VEC4", component=ct, target=ARRAY_BUFFER)
        out[f"WEIGHTS_{s}"] = w.add_accessor(wt[:, 4 * s:4 * s + 4].astype(np.float32), "VEC4", target=ARRAY_BUFFER)
    return out


def enhance(doc: Document, w: Writer, layers: list[Layer], binding: Binding, *, render: dict,
            retrained_splat=None, splat_extra: dict | None = None, app: dict | None = None) -> RigResult:
    """Append b2crig's rig to a subject file (SPEC 5.1-5.3).

    layers[0] is the body layer (its faces index the body's vertices; verts/weights come from b2c_body).
    retrained_splat: trainer-PLY fields of a splat b2crig retrained (then it supersedes b2c_splat); None = rig the
    delivered splat, reusing its accessors.
    splat_extra: b2crig per-splat attributes, e.g. {"_B2CRIG_OPEN_R": array, ...}.
    app: the appearance MLP as {"params": array, "latents": array [nv, n_lat], **scalars}.
    """
    if w.owner != "b2crig":
        raise RuleError("the rig is b2crig's")
    js = doc.json
    sk = read.skeleton(doc)
    J = len(sk.joints)
    body = body_primitive(doc)
    bidx, bw = read.joints_weights(doc, body)
    skel_node = sk.root

    # ---- cage
    prims, meta, v_off, f_off = [], [], 0, 0
    all_idx, all_w = [], []
    for li, L in enumerate(layers):
        F = np.asarray(L.faces, np.uint32)
        if li == 0:
            if L.verts is not None:
                raise ValueError("layer 0 is the body: it reuses b2c_body's vertices")
            nv = doc.json["accessors"][body["attributes"]["POSITION"]]["count"]
            at = {k: v for k, v in body["attributes"].items() if k == "POSITION" or k.startswith(("JOINTS_", "WEIGHTS_"))}
            all_idx.append(bidx); all_w.append(bw)
        else:
            nv = len(L.verts)
            at = {"POSITION": w.add_accessor(np.asarray(L.verts, np.float32), "VEC3", target=ARRAY_BUFFER, minmax=True)}
            at.update(_joint_sets(w, np.asarray(L.joints), np.asarray(L.weights, np.float32), J))
            if L.body_index is not None:
                at["_B2CRIG_BODY_INDEX"] = w.add_accessor(np.asarray(L.body_index, np.float32), "SCALAR", target=ARRAY_BUFFER)
            all_idx.append(np.asarray(L.joints)); all_w.append(np.asarray(L.weights, np.float64))
        if F.size and F.max() >= nv:
            raise ValueError(f"layer {L.name}: a face indexes vertex {int(F.max())} of {nv}")
        at_idx = w.add_accessor(F.reshape(-1), "SCALAR", component=UNSIGNED_INT, target=ELEMENT_ARRAY_BUFFER)
        prims.append({"attributes": at, "indices": at_idx, "mode": 4})
        meta.append({"name": L.name, "classes": [int(c) for c in L.classes], "primitive": li, "vertexOffset": v_off,
                     "vertexCount": nv, "faceOffset": f_off, "faceCount": len(F)})
        v_off += nv; f_off += len(F)
    cage_mesh = w.add_mesh({"name": "b2crig_cage", "primitives": prims})
    cage_node = w.add_node({"name": "b2crig_cage", "mesh": cage_mesh, "skin": 0})

    # ---- rigged splat
    n = len(binding.face)
    if retrained_splat is not None:
        sat = S.attributes(w, retrained_splat)
        sat = {k: v for k, v in sat.items() if not k.startswith("_SEG_")}   # b2crunner's names; b2crig keeps them below
        f = S._fields(retrained_splat)
        if "seg_label" in f:
            sat["_B2CRIG_SEG_LABEL"] = w.add_accessor(np.asarray(f["seg_label"]).astype(np.uint8), "SCALAR", component=UNSIGNED_BYTE, target=ARRAY_BUFFER)
            sat["_B2CRIG_SEG_CONF"] = w.add_accessor(np.asarray(f["seg_conf"], np.float32), "SCALAR", target=ARRAY_BUFFER)
        supersedes = doc.node_index("b2c_splat")
    else:
        d = read.splat(doc, doc.node_index("b2c_splat"))
        sat = S.splat_attribute_keys(d["primitive"]["attributes"])   # R1a: b2crunner's accessors, referenced
        supersedes = None
    cnt = doc.json["accessors"][sat["POSITION"]]["count"]
    if cnt != n:
        raise ValueError(f"the binding has {n} splats, the splat {cnt}")
    # preview skin: the bound triangle's vertex weights blended by barycentrics, top 4
    Fall = np.concatenate([np.asarray(L.faces, np.int64) + m["vertexOffset"] for L, m in zip(layers, meta)])
    kmax = max(a.shape[1] for a in all_idx)
    CI = np.concatenate([np.pad(a, ((0, 0), (0, kmax - a.shape[1]))) for a in all_idx])
    CW = np.concatenate([np.pad(a, ((0, 0), (0, kmax - a.shape[1]))) for a in all_w])
    face = np.asarray(binding.face, np.int64)
    ok = face >= 0
    tri = Fall[np.where(ok, face, 0)]
    b = np.asarray(binding.barycentric, np.float64)
    bw3 = np.stack([1 - b[:, 0] - b[:, 1], b[:, 0], b[:, 1]], 1)
    dense = np.zeros((n, J))
    rows = np.repeat(np.arange(n), kmax)
    for c in range(3):
        np.add.at(dense, (rows, CI[tri[:, c]].reshape(-1)), (bw3[:, c:c + 1] * CW[tri[:, c]]).reshape(-1))
    top = np.argsort(-dense, 1)[:, :4]
    tw = np.take_along_axis(dense, top, 1)
    if top.shape[1] < 4:   # fewer than 4 joints in the skeleton
        top = np.pad(top, ((0, 0), (0, 4 - top.shape[1]))); tw = np.pad(tw, ((0, 0), (0, 4 - tw.shape[1])))
    tw[~ok] = [1, 0, 0, 0]; top[~ok] = 0
    tw /= np.maximum(tw.sum(1, keepdims=True), 1e-12)
    sat.update(_joint_sets(w, top, tw, J))
    for k, v in (splat_extra or {}).items():
        if not k.startswith("_B2CRIG_"):
            raise RuleError(f"b2crig splat attribute {k} needs the _B2CRIG_ prefix")
        sat[k] = w.add_accessor(np.asarray(v, np.float32), "SCALAR", target=ARRAY_BUFFER)
    ext = {"version": 1, "posing": POSING,
           "face": w.add_accessor(np.where(ok, face, UNBOUND).astype(np.uint32), "SCALAR", component=UNSIGNED_INT),
           "barycentric": w.add_accessor(np.asarray(binding.barycentric, np.float32), "VEC2"),
           "offset": w.add_accessor(np.asarray(binding.offset, np.float32), "VEC3")}
    if binding.alt:
        a = binding.alt
        ext.update({"altSplat": w.add_accessor(np.asarray(a["altSplat"], np.uint32), "SCALAR", component=UNSIGNED_INT),
                    "altFace": w.add_accessor(np.asarray(a["altFace"], np.uint32), "SCALAR", component=UNSIGNED_INT),
                    "altBarycentric": w.add_accessor(np.asarray(a["altBarycentric"], np.float32), "VEC2"),
                    "altOffset": w.add_accessor(np.asarray(a["altOffset"], np.float32), "VEC3"),
                    "altWeight": w.add_accessor(np.asarray(a["altWeight"], np.float32), "SCALAR")})
    splat_mesh = w.add_mesh({"name": "b2crig_splat", "primitives": [S.primitive(sat, {BIND: ext})]})
    node = {"name": "b2crig_splat", "mesh": splat_mesh, "skin": 0}
    if supersedes is not None:
        node["extras"] = {"supersedes": supersedes}
    splat_node = w.add_node(node)

    # ---- scene 1, document extension
    scene = w.add_scene({"name": "b2crig", "nodes": [skel_node, cage_node, splat_node]})
    w.set_default_scene(scene)
    rig = {"version": 1, "layers": meta, "render": dict(render), "bodyLayer": 0, "cageNode": cage_node,
           "splatNode": splat_node}
    w.set_extension(RIG, rig)
    if app is not None:   # SPEC 5.3: the appearance MLP (b2ctrain's .app), a document extension of its own
        lat = np.asarray(app["latents"], np.float32)
        w.set_extension(APP, {"version": 1, "params": w.add_accessor(np.asarray(app["params"], np.float32).reshape(-1), "SCALAR"),
                              "latents": w.add_accessor(lat.reshape(-1), "SCALAR"), "latentSize": int(lat.shape[-1]),
                              **{k: v for k, v in app.items() if k not in ("params", "latents")}})
    return RigResult(scene, cage_node, splat_node, meta)


# ---------------------------------------------------------------- reading the rig

@dataclass
class Cage:
    verts: np.ndarray        # [nv, 3] b2crunner frame
    faces: np.ndarray        # [nf, 3] global
    joints: np.ndarray       # [nv, k]
    weights: np.ndarray      # [nv, k]
    layers: list[dict]
    render: dict


def cage(doc: Document) -> Cage:
    js = doc.json
    rig = doc.extension(RIG)
    if rig is None:
        raise ValueError("no B2CRIG_rig: the subject has not been rigged")
    mesh = js["meshes"][js["nodes"][rig["cageNode"]]["mesh"]]
    V, F, I, Wt = [], [], [], []
    for m, p in zip(rig["layers"], mesh["primitives"]):
        V.append(doc.accessor(p["attributes"]["POSITION"]).astype(np.float64))
        F.append(doc.accessor(p["indices"]).astype(np.int64).reshape(-1, 3) + m["vertexOffset"])
        i, w = read.joints_weights(doc, p); I.append(i); Wt.append(w)
    k = max(a.shape[1] for a in I)
    I = np.concatenate([np.pad(a, ((0, 0), (0, k - a.shape[1]))) for a in I])
    Wt = np.concatenate([np.pad(a, ((0, 0), (0, k - a.shape[1]))) for a in Wt])
    return Cage(np.concatenate(V), np.concatenate(F), I, Wt, rig["layers"], rig["render"])


def binding(doc: Document) -> Binding:
    js = doc.json
    rig = doc.extension(RIG)
    prim = js["meshes"][js["nodes"][rig["splatNode"]]["mesh"]]["primitives"][0]
    e = prim["extensions"][BIND]
    f = doc.accessor(e["face"]).astype(np.int64)
    f[f == UNBOUND] = -1
    return Binding(f, doc.accessor(e["barycentric"]), doc.accessor(e["offset"]))


def rig_hash(doc: Document) -> str:
    """B2CRIG_clip.rig.cageSha256 (SPEC 6): sha256 over, in primitive order of b2crig_cage, each primitive's POSITION,
    index and JOINTS_n / WEIGHTS_n accessor bytes (n ascending), then the skin's inverse bind matrices and every
    joint's rest rotation and translation (float32, little-endian). A clip's residual is only valid against
    exactly this cage, weighting and skeleton."""
    js = doc.json
    rig = doc.extension(RIG)
    h = hashlib.sha256()
    mesh = js["meshes"][js["nodes"][rig["cageNode"]]["mesh"]]
    for p in mesh["primitives"]:
        at = p["attributes"]
        keys = ["POSITION"] + sorted((k for k in at if k.startswith(("JOINTS_", "WEIGHTS_"))), key=lambda k: (int(k.split("_")[1]), k))
        for k in keys[:1]:
            h.update(np.ascontiguousarray(doc.accessor(at[k])).tobytes())
        h.update(np.ascontiguousarray(doc.accessor(p["indices"])).tobytes())
        for k in keys[1:]:
            h.update(np.ascontiguousarray(doc.accessor(at[k])).tobytes())
    h.update(_skeleton_bytes(doc))
    return h.hexdigest()


def _skeleton_bytes(doc: Document, skin: int = 0) -> bytes:
    sk = read.skeleton(doc, skin)
    return (np.asarray(sk.ibm, np.float32).tobytes() + np.asarray(sk.rest_q, np.float32).tobytes()
            + np.asarray(sk.rest_t, np.float32).tobytes())
