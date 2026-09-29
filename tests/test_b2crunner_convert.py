"""b2cgltf.b2crunner.convert on synthetic inputs (no recorded dataset)."""
import numpy as np
import pytest

from b2cgltf.b2crunner import convert as C


def random_rotations(rng, n):
    q = rng.normal(size=(n, 4))
    return C.matrix_from_quat(q / np.linalg.norm(q, axis=1, keepdims=True))


def splat_props(n=5, degree=3, rng=None):
    rng = rng or np.random.default_rng(0)
    p = {k: rng.normal(size=n).astype(np.float32) for k in
         ("x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2", "opacity",
          "f_dc_0", "f_dc_1", "f_dc_2")}
    for i in range(3 * ((degree + 1) ** 2 - 1)):
        p[f"f_rest_{i}"] = rng.normal(size=n).astype(np.float32)
    p["seg_label"] = rng.integers(0, 28, n).astype(np.float32)
    p["seg_conf"] = rng.random(n).astype(np.float32)
    return p


def test_ply_round_trip(tmp_path):
    props = splat_props()
    names = list(props)
    rec = np.zeros(5, dtype=[(k, "<f4") for k in names])
    for k in names:
        rec[k] = props[k]
    header = "ply\nformat binary_little_endian 1.0\ncomment b2c.test 1 2\nelement vertex 5\n"
    header += "".join(f"property float {k}\n" for k in names) + "end_header\n"
    path = tmp_path / "s.ply"
    path.write_bytes(header.encode() + rec.tobytes())
    back, comments = C.read_trainer_ply(path)
    assert comments == ["b2c.test 1 2"]
    assert all(np.array_equal(back[k], props[k]) for k in names)


def test_skin_sets_sorted_padded_untruncated():
    # vertex 0: 2 influences, vertex 1: 5 influences (-> two sets), vertex 2: 1
    v = np.array([0, 0, 1, 1, 1, 1, 1, 2])
    j = np.array([3, 7, 1, 2, 4, 5, 6, 9])
    w = np.array([0.25, 0.75, 0.1, 0.3, 0.2, 0.25, 0.15, 1.0])
    sets = C.skin_sets(v, j, w, n_verts=3, n_joints=10)
    assert len(sets) == 2
    joints = np.concatenate([s[0] for s in sets], 1)
    weights = np.concatenate([s[1] for s in sets], 1)
    assert joints[0].tolist() == [7, 3, 0, 0, 0, 0, 0, 0]
    assert weights[0].tolist() == [0.75, 0.25, 0, 0, 0, 0, 0, 0]
    assert joints[1, :5].tolist() == [2, 5, 4, 6, 1]
    assert np.all(joints[weights == 0] == 0)
    assert np.allclose(weights.sum(1), 1)

    with pytest.raises(ValueError, match="sum to 1"):
        C.skin_sets(v, j, w * 0.9, n_verts=3, n_joints=10)
    with pytest.raises(ValueError, match="no skin influence"):
        C.skin_sets(v, j, w, n_verts=4, n_joints=10)


def test_v1_to_v2_matches_documented_frame():
    rng = np.random.default_rng(1)
    rotation = random_rotations(rng, 1)[0]
    rots_mhr = random_rotations(rng, 4)
    flip = np.diag([1.0, -1.0, -1.0])
    v1 = rotation @ rots_mhr
    v2 = rotation @ flip @ rots_mhr   # what the version-2 writer stores
    assert np.allclose(C.global_rots_v1_to_v2(rotation, v1), v2)


def test_quaternion_round_trip():
    rng = np.random.default_rng(2)
    r = random_rotations(rng, 50)
    r[0] = np.diag([1.0, -1.0, -1.0])   # trace -1: the non-trace branch
    assert np.allclose(C.matrix_from_quat(C.quat_from_matrix(r)), r, atol=1e-12)


def test_joint_nodes_bind_pose_is_identity():
    rng = np.random.default_rng(3)
    parents = np.array([-1, 0, 1, 1, 0])
    rots = random_rotations(rng, 5)
    pos = rng.normal(size=(5, 3))
    jn = C.joint_nodes(pos, rots, parents)
    assert jn["composed_error"] < 1e-6
    # a reader composes the float32 TRS; with the float32 IBMs, bind pose is the identity
    composed = []
    for i, par in enumerate(parents):
        m = np.eye(4)
        m[:3, :3] = C.matrix_from_quat(jn["rotation"][i].astype(np.float64))
        m[:3, 3] = jn["translation"][i]
        composed.append(m if par < 0 else composed[par] @ m)
    for i in range(5):
        assert np.allclose(composed[i] @ jn["inverse_bind"][i].astype(np.float64), np.eye(4), atol=1e-5)

    reflected = rots.copy()
    reflected[2] = reflected[2] @ np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="joint 2"):
        C.joint_nodes(pos, reflected, parents)
    with pytest.raises(ValueError, match="topological"):
        C.joint_nodes(pos, rots, np.array([-1, 2, 0, 1, 0]))
