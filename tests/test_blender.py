"""The Blender add-on (b2cgltf/blender): its numpy side here, and the add-on itself inside Blender when B2C_BLENDER
names a Blender 5.3+ executable (tests/blender_check.py runs there)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from b2cgltf import read
from b2cgltf.b2crig import clip as C
from b2cgltf.b2crig import rig as R
from b2cgltf.blender import correctives as K
from b2cgltf.blender import posing as P

from fixtures import W_TEST, synthetic
from test_core import rig_subject


def _rot(rng, n=None):
    q = rng.normal(size=(4,) if n is None else (n, 4))
    return P.quat_to_mat(q)


def _sh_eval(sh, d):
    """Bands 1-3 of an SH function [K, 3] at directions d [m, 3] -> [m, 3]."""
    out = 0
    for l, (a, b) in P.SH_BANDS.items():
        if b <= len(sh):
            out = out + P.sh_basis(l, d) @ sh[a:b]
    return out


def test_quaternion_round_trip():
    R = _rot(np.random.default_rng(0), 100)
    assert np.abs(P.quat_to_mat(P.mat_to_quat(R)) - R).max() < 1e-12


@pytest.mark.parametrize("K_", [3, 8, 15])
def test_rotate_sh_turns_the_radiance(K_):
    rng = np.random.default_rng(K_)
    sh = rng.normal(size=(4, K_, 3)); R = _rot(rng, 4)
    out = P.rotate_sh(sh, R)
    d = rng.normal(size=(50, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    for i in range(4):                                     # g(d) = f(R^T d)
        assert np.abs(_sh_eval(out[i], d) - _sh_eval(sh[i], d @ R[i])).max() < 1e-10


def test_frame():
    f = P.Frame(W_TEST)
    p = np.random.default_rng(0).normal(size=(5, 3))
    assert np.allclose(f.points(p), (p @ W_TEST[:3, :3].T + W_TEST[:3, 3]) @ P.C.T)
    assert np.allclose(P.Frame(np.diag([2.0, 2, 2, 1])).s, 2.0)
    with pytest.raises(ValueError):
        P.Frame(np.diag([1.0, 2, 1, 1]))


def test_pose_reference_follows_a_rigid_motion():
    """Moving the whole cage rigidly moves every bound splat rigidly: positions, rotations, scales, SH."""
    rng = np.random.default_rng(3)
    V0 = rng.normal(size=(30, 3)); F = np.array([rng.choice(30, 3, replace=False) for _ in range(40)])
    n = 25
    face = rng.integers(0, len(F), n); face[0] = -1
    bary = rng.dirichlet([1, 1, 1], n)[:, 1:]; off = rng.normal(0, 0.1, (n, 3))
    splat = {"position": rng.normal(size=(n, 3)), "rotation": P.mat_to_quat(_rot(rng, n)),
             "scale": rng.uniform(0.01, 0.1, (n, 3)), "sh": rng.normal(size=(n, 15, 3))}
    rest = P.pose_reference(V0, V0, F, face, bary, off, splat, 1.15)
    R, t = _rot(rng), rng.normal(size=3)
    moved = P.pose_reference(V0, V0 @ R.T + t, F, face, bary, off, splat, 1.15)
    b = face >= 0
    assert np.abs(moved["position"][b] - (rest["position"][b] @ R.T + t)).max() < 1e-10
    assert np.abs(P.quat_to_mat(moved["rotation"][b]) - R @ P.quat_to_mat(rest["rotation"][b])).max() < 1e-9
    assert np.abs(moved["scale"] - rest["scale"]).max() < 1e-10
    assert np.abs(moved["sh"][b] - P.rotate_sh(rest["sh"][b], R)).max() < 1e-9
    assert np.array_equal(moved["position"][0], splat["position"][0])          # unbound: untouched


def test_corrective_net_matches_dense():
    rng = np.random.default_rng(5)
    J, H, N = 6, 10, 7 * 3
    W1 = rng.normal(size=(H, 6 * (J - 2))) * (rng.random((H, 6 * (J - 2))) < 0.3)
    W2 = rng.normal(size=(N, H)) * (rng.random((N, H)) < 0.5)
    r1, c1 = np.nonzero(W1)
    nz = [np.nonzero(c)[0] for c in W2.T]
    arrays = dict(l1_rows=r1.astype(np.int32), l1_cols=c1.astype(np.int32), l1_vals=W1[r1, c1].astype(np.float32),
                  n_hidden=H, n_out=N, col_ptr=np.concatenate([[0], np.cumsum([len(x) for x in nz])]),
                  rows=np.concatenate(nz).astype(np.int32), vals=np.concatenate([W2[x, k] for k, x in enumerate(nz)]).astype(np.float32),
                  prerot=P.mat_to_quat(_rot(rng, J))[:, [1, 2, 3, 0]], first_joint=2,
                  ibm_q=np.tile([0, 0, 0, 1.0], (J, 1)), ibm_s=np.ones(J))
    net = K.Net(arrays)
    L = _rot(rng, J)
    Rr = np.swapaxes(P.quat_to_mat(P.xyzw_to_wxyz(arrays["prerot"])), -1, -2) @ L
    f = np.concatenate([Rr[:, :, 0] - [1, 0, 0], Rr[:, :, 1] - [0, 1, 0]], 1)[2:].ravel()
    assert np.allclose(net.features(L), f)
    assert np.abs(net(f) - (W2 @ np.maximum(W1 @ f, 0)).reshape(-1, 3)).max() < 1e-4
    parents = np.array([-1, 0, 1, 1, 3, 4]); G = _rot(rng, J)
    Ll = K.local_rotations(G, parents)
    assert np.allclose(G[parents[4]] @ Ll[4], G[4]) and np.allclose(Ll[0], G[0])


# ------------------------------------------------------------------ inside Blender

BLENDER = os.environ.get("B2C_BLENDER")


@pytest.mark.skipif(not BLENDER, reason="no Blender (set B2C_BLENDER to a Blender 5.3+ executable)")
def test_addon_in_blender(tmp_path):
    """Import the synthetic subject (non-identity W) and a bending clip with a residual through the add-on's operators;
    the cage must match SPEC 7.2's posed cage, and the splats b2ctrain's posing rule, in Blender's frame."""
    path = tmp_path / "scene.glb"
    _, info = synthetic(path)
    unrigged = tmp_path / "unrigged.glb"
    unrigged.write_bytes(path.read_bytes())
    doc = rig_subject(path, info)
    sk = read.skeleton(doc)
    T = 4
    q = np.tile(sk.rest_q, (T, 1, 1)); t = np.tile(sk.rest_t, (T, 1, 1))
    for f, a in enumerate((0.0, 25.0, 50.0, 70.0)):
        q[f, 1] = [0, 0, np.sin(np.radians(a) / 2), np.cos(np.radians(a) / 2)]
        q[f, 2] = [np.sin(np.radians(a) / 3), 0, 0, np.cos(np.radians(a) / 3)]
    residual = np.random.default_rng(2).normal(0, 0.004, (T, len(R.cage(doc).verts), 3))
    C.write(tmp_path / "bend.clip.glb", doc, "bend", 24.0, q, t, residual=residual)
    out = tmp_path / "result.json"
    r = subprocess.run([BLENDER, "-b", "--factory-startup", "--python", str(Path(__file__).with_name("blender_check.py")), "--",
                        str(Path(__file__).resolve().parents[1]), str(path), str(tmp_path / "bend.clip.glb"), str(unrigged), str(out)],
                       capture_output=True, text=True, timeout=600)
    assert out.exists(), r.stdout[-3000:] + r.stderr[-3000:]
    res = json.loads(out.read_text())
    assert res["clip_frames"] == T and res["correctives"].startswith("off")
    for f in res["frames"]:
        assert f["cage_err"] < 2e-5, f
        assert f["pos_err"] < 2e-5 and f["rot_err_deg"] < 0.1 and f["scale_rel_err"] < 1e-4 and f["sh_err"] < 1e-4, f
    assert max(f["moved"] for f in res["frames"]) > 0.05          # the clip really bends the column
    assert res["unrigged_err"] < 1e-5
