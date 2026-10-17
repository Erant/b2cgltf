"""The add-on's per-frame work (bpy): writes the pose correctives and the active clip's residual onto each b2c rig's
offsets helper, which the cage's b2c_offsets node modifier adds after the Armature modifier.

Runs on frame change and on depsgraph updates (posing by hand), and skips a rig whose pose, frame, weights and action
are unchanged. Per rig (custom properties on the cage):
  b2c_correctives           path of the corrective weights (correctives.py export); else the add-on preference
  b2c_corrective_weight     wc: the live MHR pose corrective of the armature's current pose
  b2c_residual_weight       wr: the active clip's residual (posed cage minus plain skinning, SPEC 6), read from the
                            clip file named on the action (b2c_clip)
offsets = wc * L(pose) + wr * (residual(frame) - wc * L(clip pose at frame)): the residual already holds the clip
pose's corrective, so with both at 1 a clip plays exactly, and an edited clip keeps a live corrective.
"""
from __future__ import annotations

import os
import time
import traceback

import bpy
import numpy as np

from ..b2crig import clip as bclip
from ..document import load
from .correctives import Net, local_rotations, skin
from .posing import quat_to_mat, xyzw_to_wxyz

PREFS = {"package": None}           # the add-on's package name (its preferences hold the default weights path)
_NETS: dict[str, Net | str] = {}
_RIGS: dict[int, "Rig"] = {}
_CLIPS: dict[str, "Clip"] = {}
_LAST: dict[int, tuple] = {}
_ERRORS: set[str] = set()
STATS = {"updates": 0, "seconds": 0.0}


def clip_joint_transforms(q: np.ndarray, t: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Local joint TRS [T, J, 4] (x y z w), [T, J, 3] -> joint transforms relative to the skeleton root [T, J, 4, 4]."""
    L = np.zeros(q.shape[:-1] + (4, 4))
    L[..., :3, :3] = quat_to_mat(xyzw_to_wxyz(np.asarray(q, np.float64))); L[..., :3, 3] = t; L[..., 3, 3] = 1
    G = np.empty_like(L)
    done = np.zeros(len(parents), bool)

    def visit(j):
        if done[j]:
            return
        p = parents[j]
        if p >= 0:
            visit(p)
        G[:, j] = L[:, j] if p < 0 else G[:, p] @ L[:, j]
        done[j] = True
    for j in range(len(parents)):
        visit(j)
    return G


def _net(path: str) -> Net | str:
    p = bpy.path.abspath(path)
    if p not in _NETS:
        try:
            _NETS[p] = Net(np.load(p))
        except Exception as e:   # noqa: BLE001 - reported in the panel
            _NETS[p] = f"cannot read {p}: {e}"
    return _NETS[p]


def weights_path(cage) -> str:
    p = cage.get("b2c_correctives") or ""
    if not p and PREFS["package"]:
        prefs = bpy.context.preferences.addons.get(PREFS["package"])
        p = prefs.preferences.correctives_path if prefs else ""
    return p or os.environ.get("B2C_MHR_CORRECTIVES", "")


class Rig:
    """A rig's arrays, read once from the cage (custom properties + point attributes)."""

    def __init__(self, cage):
        info = cage["b2c"].to_dict()
        self.bones = list(info["bones"]); J = len(self.bones)
        self.parents = np.array(info["parents"])
        Afull = np.array(info["Afull"]).reshape(4, 4)
        self.Ainv = np.linalg.inv(Afull[:3, :3])
        self.Xinv = np.linalg.inv(np.array(info["Xfull"]).reshape(J, 4, 4)[:, :3, :3])
        self.rest_G = np.array(info["rest_G"]).reshape(J, 3, 3)
        self.to_cage = np.array(info["to_cage"]).reshape(3, 3)
        self.scale = float(info["corr_scale"])
        at = cage.data.attributes
        get = lambda n: _read(at[n], np.int64 if at[n].data_type == "INT" else np.float64)
        self.src, self.body = get("b2c_src"), get("b2c_body")
        k = sum(1 for a in at if a.name.startswith("b2c_j"))
        self.idx = np.stack([get(f"b2c_j{i}") for i in range(k)], 1)
        self.w = np.stack([get(f"b2c_w{i}") for i in range(k)], 1)
        self.net, self.status = None, "off"
        path = weights_path(cage)
        if not self.scale:
            self.status = "off: the subject has no MHR body record"
        elif not path:
            self.status = "off: no corrective weights (add-on preferences)"
        else:
            net = _net(path)
            if isinstance(net, str):
                self.status = f"off: {net}"
            elif net.joints != J:
                self.status = f"off: the weights are for {net.joints} joints, the rig has {J}"
            elif info.get("mhr_sha256") and net.model_sha256 and info["mhr_sha256"] != net.model_sha256:
                self.status = "off: the weights come from another MHR model than the subject's"
            else:
                self.net, self.status = net, "on"
                self.has = self.body >= 0
                self.c0 = self._unposed(self.rest_G)

    def _unposed(self, G: np.ndarray) -> np.ndarray:
        return self.net(self.net.features(local_rotations(G, self.parents)))

    def corrective(self, G: np.ndarray) -> np.ndarray:
        """Joint rotations relative to the root, b2crunner frame [J, 3, 3] -> per cage vertex, cage local."""
        d = (self._unposed(G) - self.c0) * self.scale
        out = np.zeros((len(self.body), 3))
        h = self.has
        out[h] = skin(G, self.net.Libm, self.idx[h], self.w[h], d[self.body[h]])
        return out @ self.to_cage.T

    def joint_rotations(self, M: np.ndarray) -> np.ndarray:
        """Armature-space bone rotations [J, 3, 3] -> joint rotations relative to the root, b2crunner frame."""
        return self.Ainv @ M @ self.Xinv


class Clip:
    """A clip's residual (cage local, per Blender cage vertex) and joint rotations, from its file."""

    def __init__(self, path: str, rig: Rig):
        doc = load(path)
        e = (doc.json["animations"][0].get("extensions") or {}).get(bclip.RES)
        if e:
            r = doc.accessor(e["accessor"], normalized=True).reshape(e["frames"], e["vertices"], 3) * e["scale"]
            self.res = (r[:, rig.src] @ rig.to_cage.T).astype(np.float32)
        else:
            self.res = None
        q, t, self.fps = bclip.channels(doc)
        self.G = clip_joint_transforms(q, t, rig.parents)[..., :3, :3]
        self.frames = len(self.G)
        self._corr: dict[int, np.ndarray] = {}

    def corrective(self, rig: Rig, i: int) -> np.ndarray:
        if i not in self._corr:
            self._corr[i] = rig.corrective(self.G[i])
        return self._corr[i]


def _read(attr, dtype) -> np.ndarray:
    a = np.zeros(len(attr.data), dtype); attr.data.foreach_get("value", a)
    return a


def rig_state(cage) -> Rig:
    k = cage.session_uid
    if k not in _RIGS:
        _RIGS[k] = Rig(cage)
    return _RIGS[k]


def corrective_status(cage) -> str:
    try:
        return rig_state(cage).status
    except Exception as e:   # noqa: BLE001
        return f"error: {e}"


def clip_path(action) -> str | None:
    p = action.get("b2c_clip") if action else None
    if not p:
        return None
    if os.path.isfile(p):
        return p
    alt = bpy.path.abspath("//" + os.path.basename(p))          # moved with the .blend
    return alt if os.path.isfile(alt) else None


def _clip(path: str, rig: Rig) -> Clip:
    if path not in _CLIPS:
        _CLIPS[path] = Clip(path, rig)
    return _CLIPS[path]


def offsets(cage, depsgraph=None) -> tuple[np.ndarray, tuple]:
    rig = rig_state(cage)
    arm = cage["b2c_armature"]
    src = arm.evaluated_get(depsgraph) if depsgraph is not None else arm
    pb = src.pose.bones
    M = np.array([np.array(pb[n].matrix, np.float64)[:3, :3] for n in rig.bones])
    wc = float(cage.get("b2c_corrective_weight", 1.0)) if rig.net else 0.0
    wr = float(cage.get("b2c_residual_weight", 1.0))
    act = arm.animation_data.action if arm.animation_data else None
    scn = bpy.context.scene
    f = scn.frame_current + scn.frame_subframe
    key = (np.round(M, 7).tobytes(), f, wc, wr, act.name if act else None)
    out = np.zeros((len(rig.body), 3))
    if wc:
        out += wc * rig.corrective(rig.joint_rotations(M))
    path = clip_path(act)
    if path and wr:
        c = _clip(path, rig)
        i0 = int(np.clip(np.floor(f), 0, c.frames - 1)); i1 = min(i0 + 1, c.frames - 1); a = float(np.clip(f - i0, 0, 1))
        lerp = lambda x0, x1: x0 * (1 - a) + x1 * a if a else x0
        r = lerp(c.res[i0], c.res[i1]) if c.res is not None else 0.0
        if wc:
            r = r - wc * lerp(c.corrective(rig, i0), c.corrective(rig, i1))
        out += wr * r
    return out, key


def update(cage, depsgraph=None, force: bool = False) -> bool:
    out, key = offsets(cage, depsgraph)
    k = cage.session_uid
    if not force and _LAST.get(k) == key:
        return False
    _LAST[k] = key
    me = cage["b2c_offsets"].data
    me.vertices.foreach_set("co", out.astype(np.float32).ravel())
    me.update()
    return True


def rigs(scene):
    return [o for o in scene.objects if o.type == "MESH" and "b2c" in o.keys() and o.get("b2c_offsets")]


@bpy.app.handlers.persistent
def _on_update(scene, depsgraph=None):
    t = time.perf_counter()
    for cage in rigs(scene):
        try:
            STATS["updates"] += update(cage, depsgraph)
        except Exception:   # noqa: BLE001 - a handler must not raise; report once per message
            msg = traceback.format_exc()
            if msg not in _ERRORS:
                _ERRORS.add(msg); print(f"b2c: {cage.name}: {msg}")
    STATS["seconds"] += time.perf_counter() - t


@bpy.app.handlers.persistent
def _on_load(*_):
    reset()


def reset():
    """Forget cached rigs, clips and weights (after loading a file or changing the weights path)."""
    _RIGS.clear(); _CLIPS.clear(); _LAST.clear(); _NETS.clear(); _ERRORS.clear()


HANDLERS = ((bpy.app.handlers.frame_change_post, _on_update), (bpy.app.handlers.depsgraph_update_post, _on_update),
            (bpy.app.handlers.load_post, _on_load))


def register():
    unregister()
    for hs, h in HANDLERS:
        hs.append(h)


def unregister():
    for hs, h in HANDLERS:
        for x in [x for x in hs if getattr(x, "__name__", "") == h.__name__ and getattr(x, "__module__", "") == h.__module__]:
            hs.remove(x)
