"""b2cgltf.b2crunner.subject.write_subject on synthetic inputs (no recorded dataset)."""
import numpy as np
import pytest

from b2cgltf import document, read
from b2cgltf.b2crunner import convert, subject
from b2cgltf.validate import validate, validator
from fixtures import W_TEST


def splat_fields(n=20, rng=None):
    rng = rng or np.random.default_rng(0)
    f = {k: rng.normal(size=n).astype(np.float32) for k in
         ("x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2", "opacity",
          "f_dc_0", "f_dc_1", "f_dc_2")}
    f.update({f"f_rest_{i}": rng.normal(size=n).astype(np.float32) for i in range(45)})
    f["seg_label"] = rng.integers(0, 28, n).astype(np.float32)
    f["seg_conf"] = rng.random(n).astype(np.float32)
    f["ev_err"] = np.zeros(n, np.float32)
    return f


def inputs(rng=None):
    rng = rng or np.random.default_rng(1)
    parents = np.array([-1, 0, 1, 1])
    q = rng.normal(size=(4, 4))
    rots = convert.matrix_from_quat(q / np.linalg.norm(q, axis=1, keepdims=True))
    joints = rng.normal(size=(4, 3))
    verts = rng.normal(size=(6, 3)).astype(np.float32)
    faces = np.array([[0, 1, 2], [2, 3, 4], [3, 4, 5]])
    sv = np.array([0, 0, 1, 2, 2, 2, 3, 4, 5, 5])
    sj = np.array([0, 1, 1, 1, 2, 3, 2, 3, 0, 3])
    sw = np.array([0.4, 0.6, 1.0, 0.2, 0.3, 0.5, 1.0, 1.0, 0.9, 0.1])
    body = subject.Body(
        verts=verts, faces=faces, skin_vertex=sv, skin_joint=sj, skin_weight=sw, joint_parents=parents,
        joints=joints, global_rots=rots, model_sha256="0" * 64,
        world_from_raw={"scale": 1.0, "rotation": np.eye(3), "translation": np.zeros(3)},
        pose_params={"global_rot": np.zeros(3), "shape_params": np.ones(45)}, model_params=np.arange(204.0),
        hand_idx=np.array([2, 3]), joint_names=["body_world", "root", "l_upleg", "r_upleg"])
    cams = subject.Cameras(rotation=np.stack([np.eye(3)] * 3), position=rng.normal(size=(3, 3)),
                           intrinsics=np.tile([1000.0, 1000.0, 540.0, 960.0], (3, 1)), image_size=(1080, 1920),
                           names=["a.png", "b.png", "c.png"], dataset="../colmap/")
    capture = subject.Capture(final=cams, orbit=cams, helix={"n_frames": 3}, extras={"orbit_target": [0, 1, 0]},
                              images={"front": b"\x89PNG\r\n\x1a\nnot really"})
    return body, capture


@pytest.mark.parametrize("W", [np.eye(4), W_TEST])
def test_write_subject_reads_back(tmp_path, W):
    body, capture = inputs()
    fields = splat_fields()
    path = subject.write_subject(tmp_path / "s.glb", fields, body, capture, W=W, tool="test", log=lambda *_: None)
    doc = document.load(path)
    js = doc.json
    assert doc.id and [h["tool"] for h in js["asset"]["extras"]["b2c_history"]] == ["test"]
    assert js["scenes"][js["scene"]]["name"] == "b2c"

    s = read.splat(doc, read.current_splat(doc))
    back = read.to_trainer_ply_fields(s)
    for k in ("x", "f_dc_1", "f_rest_17", "seg_label"):
        assert np.array_equal(np.asarray(back[k], np.float32), fields[k])
    assert "ev_err" not in back

    sk = read.skeleton(doc)
    assert np.allclose(sk.W, W) and sk.names == body.joint_names
    node = next(n for n in js["nodes"] if n.get("name") == "b2c_body")
    prim = js["meshes"][node["mesh"]]["primitives"][0]
    V = doc.accessor(prim["attributes"]["POSITION"])
    idx, w = read.joints_weights(doc, prim)
    rest = read.skin_points(V, idx, w, sk.rel_matrices(sk.rest_q, sk.rest_t))
    assert np.abs(rest - V).max() < 1e-5   # bind pose is the identity in the b2crunner frame

    mhr = node["extensions"]["B2C_mhr"]
    assert mhr["model"]["file"] == subject.MHR_FILE and mhr["jointNames"] == body.joint_names
    assert np.array_equal(doc.accessor(mhr["model_params"]).reshape(-1), np.arange(204.0, dtype=np.float32))
    orbit = js["extensions"]["B2C_orbit"]
    assert orbit["final_cameras"]["names"] == ["a.png", "b.png", "c.png"] and orbit["migrated"] is False
    assert np.allclose(doc.accessor(orbit["final_cameras"]["position"]), capture.final.position, atol=1e-6)
    assert js["images"][orbit["images"]["front"]]["mimeType"] == "image/png"


def test_write_subject_refusals(tmp_path):
    body, capture = inputs()
    capture.final.names = ["a.png"]
    with pytest.raises(ValueError, match="one name per camera"):
        subject.write_subject(tmp_path / "s.glb", splat_fields(), body, capture, log=lambda *_: None)
    body, capture = inputs()
    shear = np.eye(4)
    shear[0, 1] = 0.5
    with pytest.raises(ValueError, match="rigid"):
        subject.write_subject(tmp_path / "s.glb", splat_fields(), body, capture, W=shear, log=lambda *_: None)
    body, capture = inputs()
    capture.images = {"thumbnail": b"x"}
    with pytest.raises(ValueError, match="reference/anchor/front"):
        subject.write_subject(tmp_path / "s.glb", splat_fields(), body, capture, log=lambda *_: None)


@pytest.mark.skipif(validator() is None, reason="no glTF-Validator (set B2C_GLTF_VALIDATOR)")
def test_write_subject_validates(tmp_path):
    body, capture = inputs()
    capture.images = {}   # the stand-in PNG bytes are not an image
    path = subject.write_subject(tmp_path / "s.glb", splat_fields(), body, capture, W=W_TEST, log=lambda *_: None)
    validate(path)
