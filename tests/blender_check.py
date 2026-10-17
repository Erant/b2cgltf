"""Run inside Blender by tests/test_blender.py:
    blender -b --factory-startup --python blender_check.py -- REPO SUBJECT CLIP UNRIGGED OUT.json
Registers the add-on from the repo, imports through its operators, and writes the errors against the numpy references."""
import json
import sys

import bpy
import numpy as np

repo, subject, clip_path, unrigged, out = sys.argv[sys.argv.index("--") + 1:]
sys.path.insert(0, repo)
import b2cgltf.blender as addon                                   # noqa: E402
from b2cgltf import load, read                                     # noqa: E402
from b2cgltf.b2crig import clip as bclip, rig as brig              # noqa: E402
from b2cgltf.blender import runtime                                # noqa: E402
from b2cgltf.blender.posing import Frame, pose_reference, splat_to_blender   # noqa: E402

addon.register()
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
res = {"frames": []}


def attrs(obj, names):
    dg = bpy.context.evaluated_depsgraph_get(); d = obj.evaluated_get(dg).data
    out_ = {}
    for n, w, key in names:
        a = d.attributes[n]; v = np.zeros(len(a.data) * w, np.float32); a.data.foreach_get(key, v); out_[n] = v.reshape(-1, w)
    return out_


assert bpy.ops.import_scene.b2c_subject(filepath=subject) == {"FINISHED"}
assert bpy.ops.import_scene.b2c_clip(filepath=clip_path) == {"FINISHED"}
cage = next(o for o in bpy.data.objects if "b2c" in o.keys())
res["correctives"] = runtime.corrective_status(cage)
doc, cdoc = load(subject), load(clip_path)
frame = Frame(read.skeleton(doc).W)
cg, bd = brig.cage(doc), brig.binding(doc)
sb = splat_to_blender(read.splat(doc), frame)
V0 = frame.points(cg.verts)
src = np.zeros(len(cage.data.vertices), np.int32); cage.data.attributes["b2c_src"].data.foreach_get("value", src)
exact_all = bclip.pose(doc, cdoc)
res["clip_frames"] = len(exact_all)
splat = cage["b2c_splat"]
K = sb["sh"].shape[1]
for f in range(len(exact_all)):
    bpy.context.scene.frame_set(f)
    exact = frame.points(exact_all[f])
    c = attrs(cage, [("position", 3, "vector")])["position"]
    M = np.array(cage.matrix_world); c = c @ M[:3, :3].T + M[:3, 3]
    g = attrs(splat, [("position", 3, "vector"), ("rotation", 4, "value"), ("scale", 3, "vector")]
              + [(f"radiance:sh_{i}", 3, "vector") for i in range(K)])
    Ms = np.array(splat.matrix_world)
    ref = pose_reference(V0, exact, cg.faces, bd.face, bd.barycentric, bd.offset, sb, cage["b2c"]["max_growth"])
    q = g["rotation"] / np.linalg.norm(g["rotation"], axis=1, keepdims=True)
    res["frames"].append({
        "frame": f, "cage_err": float(np.abs(c - exact[src]).max()),
        "pos_err": float(np.abs(g["position"] @ Ms[:3, :3].T + Ms[:3, 3] - ref["position"]).max()),
        "rot_err_deg": float(np.degrees(2 * np.arccos(np.clip(np.abs((q * ref["rotation"]).sum(1)), 0, 1))).max()),
        "scale_rel_err": float(np.abs(g["scale"] / ref["scale"] - 1).max()),
        "sh_err": float(max(np.abs(g[f"radiance:sh_{i}"] - ref["sh"][:, i]).max() for i in range(K))),
        "moved": float(np.abs(ref["position"] - sb["position"]).max())})

assert bpy.ops.import_scene.b2c_subject(filepath=unrigged) == {"FINISHED"}
static = [o for o in bpy.data.objects if o.type == "POINTCLOUD" and not o.modifiers][-1]
p = attrs(static, [("position", 3, "vector")])["position"]
res["unrigged_err"] = float(np.abs(p - sb["position"]).max())
json.dump(res, open(out, "w"), indent=1)
