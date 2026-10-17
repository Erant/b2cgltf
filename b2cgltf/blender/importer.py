"""Import b2c subject and clip files into Blender (bpy).

A rigged subject becomes, in a collection named after the file:
  armature   the skeleton, and the cage skinned to it by an Armature modifier (Blender's own glTF importer, fed the
             subject stripped to skeleton + skin + cage: web.rig_glb). The importer keeps only the vertices a cage
             primitive's faces use; b2c_src (a cage point attribute) is each vertex's index in the file.
  b2c_splat  the current splat (SPEC 7.1) as a GAUSSIAN_SPLAT point cloud with the binding as point attributes and the
             b2c_pose_splat node modifier, which poses it from the evaluated cage.
  offsets    a hidden helper mesh; the cage's b2c_offsets node modifier adds its positions after skinning, and the
             runtime (runtime.py) writes the pose correctives and the active clip's residual into it.
An unrigged subject (b2crunner's file before rigging) becomes the static splat only.

A clip becomes an action on the armature (one key per frame), with the clip file's path on the action
(b2c_clip): the runtime reads its residual from there, so the .blend holds no copy of it.
"""
from __future__ import annotations

import os
import tempfile

import bpy
import numpy as np

from .. import read, web
from ..b2crig import clip as bclip, rig as brig
from ..document import RuleError, load
from . import nodes, runtime
from .posing import Frame, mat_to_quat, splat_to_blender, tri_frames


def _collection(name: str, parent) -> bpy.types.Collection:
    c = bpy.data.collections.new(name); parent.children.link(c)
    return c


def _move(obj, coll) -> None:
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


def _set(attrs, name: str, dtype: str, domain: str, values) -> None:
    a = attrs.get(name) or attrs.new(name, dtype, domain)
    key = {"FLOAT": "value", "INT": "value", "BOOLEAN": "value", "QUATERNION": "value"}.get(dtype, "vector")
    kind = np.int32 if dtype == "INT" else (bool if dtype == "BOOLEAN" else np.float32)
    a.data.foreach_set(key, np.ascontiguousarray(values, kind).ravel())


def _splat_object(name: str, sb: dict, coll) -> bpy.types.Object:
    pc = bpy.data.pointclouds.new(name); pc.resize(len(sb["position"])); pc.type = "GAUSSIAN_SPLAT"
    at = pc.attributes
    _set(at, "position", "FLOAT_VECTOR", "POINT", sb["position"])
    _set(at, "rotation", "QUATERNION", "POINT", sb["rotation"])
    _set(at, "scale", "FLOAT_VECTOR", "POINT", sb["scale"])
    _set(at, "radiance:base", "FLOAT4", "POINT", sb["base"])
    for i in range(sb["sh"].shape[1]):
        _set(at, f"radiance:sh_{i}", "FLOAT_VECTOR", "POINT", sb["sh"][:, i])
    o = bpy.data.objects.new(name, pc); coll.objects.link(o)
    return o


def _node_group(name: str, make):
    return bpy.data.node_groups.get(name) or make()


def import_subject(context, path: str, *, correctives: str | None = None) -> dict:
    """Returns {"cage", "armature", "splat", "report"}; cage and armature are None for an unrigged subject."""
    doc = load(path)
    stem = os.path.basename(path).removesuffix(".glb")
    coll = _collection(stem, context.scene.collection)
    s = read.splat(doc)
    rigx = doc.extension(brig.RIG)
    W = read.skeleton(doc).W if doc.json.get("skins") else np.eye(4)
    frame = Frame(W)
    sb = splat_to_blender(s, frame)
    report = {"subject": stem, "splats": len(sb["position"]), "rigged": rigx is not None}
    if rigx is None:
        splat = _splat_object(f"{stem}_splat", sb, coll)
        return {"cage": None, "armature": None, "splat": splat, "report": report}

    # skeleton + cage through Blender's glTF importer
    before = set(bpy.data.objects)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "rig.glb")
        with open(p, "wb") as f:
            f.write(web.rig_glb(doc))
        bpy.ops.import_scene.gltf(filepath=p)
    new = [o for o in bpy.data.objects if o not in before]
    for o in new:
        _move(o, coll)
    arm = next(o for o in new if o.type == "ARMATURE")
    cage = next(o for o in new if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers))
    for o in new:
        if o not in (arm, cage) and o.parent is None and o.type == "MESH":   # the importer's bone-shape helper
            o.hide_set(True); o.hide_render = True
    cg = brig.cage(doc)
    keep = np.concatenate([np.unique(cg.faces[m["faceOffset"]:m["faceOffset"] + m["faceCount"]]) for m in cg.layers])
    remap = np.full(len(cg.verts), -1, np.int64); remap[keep] = np.arange(len(keep))
    me = cage.data
    if len(me.vertices) != len(keep):
        raise RuleError(f"the importer kept {len(me.vertices)} cage vertices, expected {len(keep)}")
    Vb = np.zeros(len(keep) * 3, np.float32); me.vertices.foreach_get("co", Vb)
    Mc = np.array(cage.matrix_world)
    V0 = frame.points(cg.verts)                                          # Blender world, rest
    err = np.abs(Vb.reshape(-1, 3) @ Mc[:3, :3].T + Mc[:3, 3] - V0[keep]).max()
    report["cage_rest_err_mm"] = float(err * 1000)
    if err > 1e-4 + 1e-6 * np.abs(V0).max():
        raise RuleError(f"the imported cage is {err * 1000:.3f} mm off the file's: frame conversion mismatch")
    cage.name = f"{stem}_cage"; arm.name = f"{stem}_armature"
    cage.hide_render = True; cage.display_type = "WIRE"

    # the rig data the runtime needs, on the cage
    sk = read.skeleton(doc)
    nb = rigx["layers"][rigx["bodyLayer"]]["vertexCount"]
    body = np.full(len(cg.verts), -1, np.int64); body[:nb] = np.arange(nb)
    js = doc.json
    prims = js["meshes"][js["nodes"][rigx["cageNode"]]["mesh"]]["primitives"]
    for m, p in zip(rigx["layers"], prims):   # offset layers: vertices that copy their body vertex (and its weights)
        if "_B2CRIG_BODY_INDEX" in p["attributes"]:
            bi = np.rint(doc.accessor(p["attributes"]["_B2CRIG_BODY_INDEX"])).astype(np.int64)
            v = np.arange(m["vertexOffset"], m["vertexOffset"] + m["vertexCount"])
            wa, wb = np.sort(cg.weights[v], 1), np.sort(cg.weights[bi], 1)
            same = np.abs(wa - wb).max(1) < 1e-6
            body[v[same]] = bi[same]
    _set(me.attributes, "b2c_src", "INT", "POINT", keep)
    _set(me.attributes, "b2c_body", "INT", "POINT", body[keep])
    for k in range(cg.joints.shape[1]):
        _set(me.attributes, f"b2c_j{k}", "INT", "POINT", cg.joints[keep, k])
        _set(me.attributes, f"b2c_w{k}", "FLOAT", "POINT", cg.weights[keep, k])
    Ma = np.array(arm.matrix_world)
    Afull = np.linalg.inv(Ma) @ frame.matrix                              # b2crunner frame -> armature space
    G_rest = np.linalg.inv(sk.ibm)                                        # joint transforms at bind, rel. the root
    missing = [n for n in sk.names if n not in arm.data.bones]
    if missing:
        raise RuleError(f"joints without a bone after import: {missing[:5]}")
    Mrest = np.array([np.array(arm.data.bones[n].matrix_local) for n in sk.names])
    Xfull = np.linalg.inv(Afull @ G_rest) @ Mrest
    mhr = next((n["extensions"]["B2C_mhr"] for n in js["nodes"] if "B2C_mhr" in n.get("extensions", {})), None)
    cage["b2c"] = {
        "subject_id": doc.id or "", "rig_hash": brig.rig_hash(doc), "subject_path": os.path.abspath(path),
        "bones": list(sk.names), "parents": [int(x) for x in sk.parents],
        "Afull": Afull.ravel().tolist(), "Xfull": Xfull.ravel().tolist(), "rest_G": G_rest[:, :3, :3].ravel().tolist(),
        "to_cage": (np.linalg.inv(Mc[:3, :3]) @ frame.M).ravel().tolist(),   # b2crunner vectors -> cage local
        "corr_scale": 0.01 * float(mhr["world_from_raw"]["scale"]) if mhr and "world_from_raw" in mhr else 0.0,
        "mhr_sha256": (mhr or {}).get("model", {}).get("sha256", ""), "max_growth": float(cg.render.get("maxGrowth") or 0),
    }
    cage["b2c_corrective_weight"] = 1.0
    cage["b2c_residual_weight"] = 1.0
    if correctives:
        cage["b2c_correctives"] = correctives

    # the offsets helper and the cage's modifier that adds it
    hm = bpy.data.meshes.new(f"{stem}_offsets"); hm.vertices.add(len(keep))
    helper = bpy.data.objects.new(f"{stem}_offsets", hm); coll.objects.link(helper)
    helper.hide_render = True; helper.hide_select = True; helper.display_type = "BOUNDS"
    helper.hide_set(True)
    mod = cage.modifiers.new("b2c offsets", "NODES"); mod.node_group = _node_group(nodes.OFFSET_TREE, nodes.offset_graph)
    nodes.set_input(mod, "Offsets", helper)

    # the rigged splat
    bd = brig.binding(doc)
    F = cg.faces
    R0, k0 = tri_frames(V0, F)
    bound = bd.face >= 0; fc = np.where(bound, bd.face, 0)
    splat = _splat_object(f"{stem}_splat", sb, coll)
    at = splat.data.attributes
    for i in range(sb["sh"].shape[1]):
        _set(at, f"b2c_sh{i}", "FLOAT_VECTOR", "POINT", sb["sh"][:, i])
    _set(at, "b2c_q0", "QUATERNION", "POINT", sb["rotation"]); _set(at, "b2c_s0", "FLOAT_VECTOR", "POINT", sb["scale"])
    for j, nm in enumerate(("b2c_ia", "b2c_ib", "b2c_ic")):
        _set(at, nm, "INT", "POINT", remap[F[fc, j]])
    _set(at, "b2c_bary", "FLOAT_VECTOR", "POINT", np.c_[bd.barycentric, np.zeros(len(fc))])
    _set(at, "b2c_off", "FLOAT_VECTOR", "POINT", bd.offset)
    _set(at, "b2c_k0", "FLOAT", "POINT", np.where(bound, k0[fc], 1.0))
    _set(at, "b2c_r0inv", "QUATERNION", "POINT", mat_to_quat(np.transpose(R0[fc], (0, 2, 1))))
    _set(at, "b2c_bound", "BOOLEAN", "POINT", bound)
    mod = splat.modifiers.new("b2c pose", "NODES")
    nsh = sb["sh"].shape[1]
    mod.node_group = _node_group(f"{nodes.POSE_TREE}_sh{nsh}", lambda: _named(nodes.pose_graph(nsh), f"{nodes.POSE_TREE}_sh{nsh}"))
    nodes.set_input(mod, "Cage", cage); nodes.set_input(mod, "Max Growth", cage["b2c"]["max_growth"])

    cage["b2c_armature"] = arm; cage["b2c_splat"] = splat; cage["b2c_offsets"] = helper
    arm["b2c_cage"] = cage; splat["b2c_cage"] = cage
    report.update(bound=int(bound.sum()), cage_vertices=len(keep), joints=len(sk.names),
                  correctives=runtime.corrective_status(cage))
    runtime.update(cage, force=True)
    return {"cage": cage, "armature": arm, "splat": splat, "report": report}


def _named(nt, name):
    nt.name = name
    return nt


# ---------------------------------------------------------------- clips

def find_rig(obj) -> bpy.types.Object | None:
    """The cage of the b2c rig obj belongs to (the cage, its armature or its splat)."""
    if obj is None:
        return None
    if "b2c" in obj.keys():
        return obj
    c = obj.get("b2c_cage")
    return c if isinstance(c, bpy.types.Object) else None


def import_clip(context, cage, path: str) -> dict:
    """The clip as an action on the rig's armature, keyed at every frame from frame 0; the scene's fps and frame range
    follow the clip. Refuses a clip of another subject or rig (SPEC 6 checks)."""
    clip = load(path)
    e = clip.extension(bclip.CLIP)
    if e is None:
        raise RuleError(f"{path}: not a b2c clip file")
    info = cage["b2c"]
    if e["subject"]["id"] != info["subject_id"]:
        raise RuleError(f"the clip is of subject {e['subject']['id']}, the rig is {info['subject_id']}")
    if e["rig"]["cageSha256"] != info["rig_hash"]:
        raise RuleError("the clip was made for another rig of this subject (cage hash differs)")
    q, t, fps = bclip.channels(clip)
    G = runtime.clip_joint_transforms(q, t, np.array(info["parents"]))           # [T, J, 4, 4]
    arm = cage["b2c_armature"]
    J = len(info["bones"])
    Afull = np.array(info["Afull"]).reshape(4, 4); Xfull = np.array(info["Xfull"]).reshape(J, 4, 4)
    M = Afull @ G @ Xfull                                                     # armature-space bone matrices [T, J, 4, 4]
    T = len(M)
    idx = {n: j for j, n in enumerate(info["bones"])}
    name = e.get("name") or os.path.basename(path).removesuffix(".clip.glb")
    act = bpy.data.actions.new(name)
    if arm.animation_data is None:
        arm.animation_data_create()
    slot = act.slots.new(id_type="OBJECT", name=arm.name)
    from bpy_extras import anim_utils
    bag = anim_utils.action_ensure_channelbag_for_slot(act, slot)
    frames = np.arange(T, dtype=np.float32)
    for n, j in idx.items():
        bone = arm.data.bones[n]
        pb = arm.pose.bones[n]
        pb.rotation_mode = "QUATERNION"
        if bone.parent is None:
            basis = np.linalg.inv(np.array(bone.matrix_local)) @ M[:, j]
        else:
            pj = idx[bone.parent.name]
            rest_rel = np.linalg.inv(np.array(bone.parent.matrix_local)) @ np.array(bone.matrix_local)
            basis = np.linalg.inv(rest_rel) @ np.linalg.inv(M[:, pj]) @ M[:, j]
        loc = basis[:, :3, 3]
        quat = mat_to_quat(basis[:, :3, :3])
        for i in range(1, T):                                                 # keep consecutive keys on one hemisphere
            if np.dot(quat[i], quat[i - 1]) < 0:
                quat[i] = -quat[i]
        for path_, vals in (("location", loc), ("rotation_quaternion", quat)):
            for c in range(vals.shape[1]):
                fc = bag.fcurves.new(f'pose.bones["{n}"].{path_}', index=c, group_name=n)
                fc.keyframe_points.add(T)
                fc.keyframe_points.foreach_set("co", np.c_[frames, vals[:, c]].astype(np.float32).ravel())
                fc.keyframe_points.foreach_set("interpolation", np.full(T, 1, np.int32))   # LINEAR (BEZT_IPO_LIN)
                fc.update()
    act["b2c_clip"] = os.path.abspath(path)
    act["b2c_fps"] = float(fps)
    act.use_fake_user = True
    arm.animation_data.action = act
    arm.animation_data.action_slot = slot
    scn = context.scene
    scn.render.fps = int(round(fps)); scn.render.fps_base = scn.render.fps / fps
    scn.frame_start, scn.frame_end = 0, T - 1
    runtime.update(cage, force=True)
    has_res = bool((clip.json["animations"][0].get("extensions") or {}).get(bclip.RES))
    return {"clip": name, "frames": T, "fps": fps, "residual": has_res}
