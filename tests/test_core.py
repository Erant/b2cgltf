"""Core package tests (SPEC 8): preservation, the write rules, the splat encoding, the frame rule, the validator."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from b2cgltf import glb, read
from b2cgltf import splat as S
from b2cgltf.b2crig import clip as C
from b2cgltf.b2crig import rig as R
from b2cgltf.document import Document, RuleError
from b2cgltf.validate import ValidationError, validate, validator

from fixtures import W_TEST, synthetic


@pytest.fixture
def subject(tmp_path):
    doc, info = synthetic(tmp_path / "scene.glb")
    return tmp_path / "scene.glb", info


def rig_subject(path, info, W=W_TEST):
    """Rig the synthetic subject: layer 0 = the body; binding by nearest face; plus an offset layer."""
    doc = Document.load(path)
    w = doc.writer("b2crig", tool="test")
    V, F = info["V"], info["F"]
    offset = V + np.c_[np.sign(V[:, 0]), np.zeros(len(V)), np.sign(V[:, 2])] * 0.02
    ji = np.argsort(-info["Wt"], 1)[:, :3]
    jw = np.take_along_axis(info["Wt"], ji, 1)
    layers = [R.Layer("body", [], F), R.Layer("upper", [23], F, offset, ji, jw, np.arange(len(V)))]
    binding = R.Binding(info["fi"], info["b"][:, 1:], np.zeros((len(info["fi"]), 3)))
    R.enhance(doc, w, layers, binding, render={"maxGrowth": 1.15, "minConf": 0.5})
    w.save(path)
    return Document.load(path)


# ------------------------------------------------------------------ preservation (SPEC 8)

def test_preservation_plain_load_save(subject, tmp_path):
    path, _ = subject
    js, b = glb.read(path)
    js["nodes"][0]["extras"] = {"note": [1, {"x": None}]}
    js.setdefault("extensions", {})["VENDOR_unknown"] = {"deep": {"v": 1.5}}
    js["extensionsUsed"].append("VENDOR_unknown")
    glb.write(tmp_path / "a.glb", js, b)
    doc = Document.load(tmp_path / "a.glb")
    glb.write(tmp_path / "b.glb", doc.json, doc.bin)
    js2, b2 = glb.read(tmp_path / "b.glb")
    assert js2 == js and bytes(b2) == bytes(b)
    assert (tmp_path / "a.glb").read_bytes() == (tmp_path / "b.glb").read_bytes()


def test_writer_save_only_appends_history(subject, tmp_path):
    path, _ = subject
    before, bb = glb.read(path)
    doc = Document.load(path)
    doc.writer("b2crig", tool="noop").save(tmp_path / "c.glb")
    after, ba = glb.read(tmp_path / "c.glb")
    hist = after["asset"]["extras"].pop("b2c_history")
    assert hist[:-1] == before["asset"]["extras"].pop("b2c_history") and hist[-1]["tool"] == "noop"
    assert after == before and bytes(ba) == bytes(bb)


# ------------------------------------------------------------------ rules

def test_r1_r2_refuse_editing_the_other_owners_objects(subject, tmp_path):
    path, _ = subject
    doc = Document.load(path)
    w = doc.writer("b2crig", tool="t")
    doc.json["nodes"][0]["name"] = "b2c_world2"
    with pytest.raises(RuleError, match="changed"):
        w.save(tmp_path / "x.glb")


def test_r2_refuses_editing_even_own_saved_objects(subject, tmp_path):
    path, info = subject
    doc = rig_subject(path, info)
    w = doc.writer("b2crig", tool="t")
    doc.json["nodes"][doc.node_index("b2crig_cage")]["extras"] = {"x": 1}
    with pytest.raises(RuleError, match="changed"):
        w.save(tmp_path / "x.glb")


def test_r3_refuses_changed_bytes(subject, tmp_path):
    path, _ = subject
    doc = Document.load(path)
    w = doc.writer("b2crig", tool="t")
    doc.bin[0] ^= 0xFF
    with pytest.raises(RuleError, match="R3"):
        w.save(tmp_path / "x.glb")


def test_r1_prefixes(subject):
    path, _ = subject
    doc = Document.load(path)
    w = doc.writer("b2crig", tool="t")
    with pytest.raises(RuleError):
        w.add_node({"name": "b2c_mine"})
    with pytest.raises(RuleError):
        w.set_extension("B2C_orbit", {"version": 1})
    with pytest.raises(RuleError):
        w.add_mesh({"name": "b2crig_m", "primitives": [{"attributes": {"_SEG_LABEL": 0}}]})
    with pytest.raises(RuleError):
        w.add_node({"name": "joint_without_prefix"})
    w.add_node({"name": "b2crig_ok"})


def test_r1_other_owners_document_extension_untouched(subject, tmp_path):
    path, _ = subject
    doc = Document.load(path)
    w = doc.writer("b2crig", tool="t")
    doc.json["extensions"]["B2C_orbit"]["migrated"] = False
    with pytest.raises(RuleError, match="B2C_orbit"):
        w.save(tmp_path / "x.glb")


def test_r4_nothing_required(subject, tmp_path):
    path, _ = subject
    doc = Document.load(path)
    w = doc.writer("b2crig", tool="t")
    doc.json["extensionsRequired"] = ["B2CRIG_rig"]
    with pytest.raises(RuleError, match="R4"):
        w.save(tmp_path / "x.glb")


def test_r5_id_and_history(subject, tmp_path):
    path, _ = subject
    doc = Document.load(path)
    assert doc.id and len(doc.json["asset"]["extras"]["b2c_history"]) == 1
    w = doc.writer("b2crig", tool="t")
    doc.json["asset"]["extras"]["b2c_id"] = "other"
    with pytest.raises(RuleError, match="R5"):
        w.save(tmp_path / "x.glb")
    with pytest.raises(RuleError):
        Document.new("b2crig", tool="t")   # only b2crunner creates subject files


def test_r6_unknown_version_refused(subject, tmp_path):
    path, _ = subject
    js, b = glb.read(path)
    js["extensions"]["B2C_orbit"]["version"] = 2
    glb.write(tmp_path / "v.glb", js, b)
    with pytest.raises(RuleError, match="version"):
        Document.load(tmp_path / "v.glb")


def test_r8_stripped_is_never_enhanced(subject, tmp_path):
    path, _ = subject
    js, b = glb.read(path)
    js["asset"]["extras"]["b2c_stripped"] = ["animations/0"]
    glb.write(tmp_path / "s.glb", js, b)
    with pytest.raises(RuleError, match="R8"):
        Document.load(tmp_path / "s.glb").writer("b2crig", tool="t")


# ------------------------------------------------------------------ the splat (SPEC 4.2)

def test_splat_round_trip(subject):
    path, info = subject
    s = read.splat(Document.load(path))
    f = read.to_trainer_ply_fields(s)
    src = info["fields"]
    for k in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "seg_label"):
        assert np.array_equal(f[k], src[k]), k
    for k in range(45):
        assert np.array_equal(f[f"f_rest_{k}"], src[f"f_rest_{k}"])
    q = np.stack([src[f"rot_{k}"] for k in range(4)], 1).astype(np.float64); q /= np.linalg.norm(q, axis=1, keepdims=True)
    assert np.allclose(np.stack([f[f"rot_{k}"] for k in range(4)], 1), q, atol=1e-7)
    for k in range(3):
        assert np.allclose(f[f"scale_{k}"], src[f"scale_{k}"], atol=1e-5)
    assert np.allclose(f["opacity"], src["opacity"], atol=1e-4)


@pytest.mark.parametrize("bad", ["zero_quat", "missing_rest", "seg_no_conf", "saturated", "unknown", "fractional_label"])
def test_splat_refusals(subject, bad):
    _, info = subject
    f = copy.deepcopy(info["fields"])
    if bad == "zero_quat":
        for k in range(4):
            f[f"rot_{k}"][3] = 0
    elif bad == "missing_rest":
        del f["f_rest_44"]
    elif bad == "seg_no_conf":
        del f["seg_conf"]
    elif bad == "saturated":
        f["opacity"][0] = 40.0
    elif bad == "unknown":
        f["mystery"] = f["x"]
    elif bad == "fractional_label":
        f["seg_label"][0] = 1.5
    doc, w = Document.new("b2crunner", tool="t")
    with pytest.raises(RuleError):
        S.attributes(w, f)


def test_splat_drops_b2crig_and_evidence_properties(subject):
    _, info = subject
    f = dict(info["fields"], ev_views=info["fields"]["x"], open_r=info["fields"]["x"], cage_fill=info["fields"]["x"])
    doc, w = Document.new("b2crunner", tool="t")
    msgs = []
    at = S.attributes(w, f, log=msgs.append)
    assert "ev_views" in msgs[0] and not any(k.startswith("_OPEN") for k in at)


# ------------------------------------------------------------------ rig, clip and the frame rule (SPEC 5, 6, 7.2)

def test_current_splat_and_scenes(subject):
    path, info = subject
    doc0 = Document.load(path)
    assert read.current_splat(doc0) == doc0.node_index("b2c_splat")
    doc = rig_subject(path, info)
    assert doc.json["scene"] == 1
    assert read.current_splat(doc) == doc.node_index("b2crig_splat")
    assert doc.node_index("b2c_world") not in doc.json["scenes"][1]["nodes"]


def _world_skin(parents, joint_pos, q_local, t_local, W, ibm):
    """Independent reference: joint world matrices INCLUDING W, as a plain glTF player composes them."""
    Jn = len(parents)
    out = np.zeros((len(q_local), Jn, 4, 4))
    for f in range(len(q_local)):
        G = [None] * Jn
        for j in range(Jn):
            L = np.eye(4); L[:3, :3] = read.quat_to_mat(q_local[f, j]); L[:3, 3] = t_local[f, j]
            G[j] = (W if parents[j] < 0 else G[parents[j]]) @ L
            out[f, j] = G[j] @ ibm[j]
    return out


def test_frame_rule_with_non_identity_W(subject, tmp_path):
    path, info = subject
    doc = rig_subject(path, info)
    sk = read.skeleton(doc)
    assert np.allclose(sk.W, W_TEST)
    # bind pose: the relative skinning matrices are the identity (SPEC 4.3: joint_world · IBM = W)
    assert np.allclose(sk.rel_matrices(sk.rest_q[None], sk.rest_t[None])[0], np.eye(4), atol=1e-6)
    # a pose: bend the middle joint 40 degrees about Z
    T = 3
    q = np.tile(sk.rest_q, (T, 1, 1)); t = np.tile(sk.rest_t, (T, 1, 1))
    for f, a in enumerate((0.0, 20.0, 40.0)):
        h = np.radians(a) / 2
        q[f, 1] = [0, 0, np.sin(h), np.cos(h)]
    cg = R.cage(doc)
    residual = np.random.default_rng(1).normal(0, 0.003, (T, len(cg.verts), 3))
    rep = C.write(tmp_path / "bend.clip.glb", doc, "bend", 30.0, q, t, residual=residual)
    clipdoc = Document.load(tmp_path / "bend.clip.glb")
    posed_plain = C.pose(doc, clipdoc, residual=False)                 # b2crunner frame
    posed = C.pose(doc, clipdoc)
    # step 2: the residual adds in the b2crunner frame
    assert np.abs(posed - posed_plain - residual).max() <= rep["residual_quant_max_m"] + 1e-9
    # a plain glTF player (joint_world · IBM, W included) sees W applied once to the b2crunner-frame result
    player = read.skin_points(cg.verts, cg.joints, cg.weights, _world_skin(sk.parents, None, q, t, sk.W, sk.ibm))
    expect = posed_plain @ sk.W[:3, :3].T + sk.W[:3, 3]
    assert np.abs(player - expect).max() < 1e-6
    # frame 0 is the rest pose: the cage is where it was bound
    assert np.abs(posed_plain[0] - cg.verts).max() < 1e-6
    # the bent frames moved the top of the column, not the bottom
    top = cg.verts[:, 1] > 0.9
    assert np.abs(posed_plain[2, top] - cg.verts[top]).max() > 0.05
    assert np.abs(posed_plain[2, cg.verts[:, 1] < 0.01] - cg.verts[cg.verts[:, 1] < 0.01]).max() < 1e-6


def test_clip_checks(subject, tmp_path):
    path, info = subject
    doc = rig_subject(path, info)
    sk = read.skeleton(doc)
    C.write(tmp_path / "c.clip.glb", doc, "rest", 30.0, sk.rest_q[None], sk.rest_t[None])
    clipdoc = Document.load(tmp_path / "c.clip.glb")
    C.check(doc, clipdoc)
    # another subject: refused
    other, oinfo = synthetic(tmp_path / "other.glb")
    other = rig_subject(tmp_path / "other.glb", oinfo)
    with pytest.raises(RuleError, match="subject"):
        C.check(other, clipdoc)
    # same subject, another rig: refused
    js, b = glb.read(tmp_path / "c.clip.glb")
    js["extensions"]["B2CRIG_clip"]["rig"]["cageSha256"] = "0" * 64
    glb.write(tmp_path / "d.clip.glb", js, b)
    with pytest.raises(RuleError, match="rig"):
        C.check(doc, Document.load(tmp_path / "d.clip.glb"))


def test_rig_hash_covers_weights(subject):
    path, info = subject
    doc = rig_subject(path, info)
    h = R.rig_hash(doc)
    cage_mesh = doc.json["meshes"][doc.json["nodes"][doc.node_index("b2crig_cage")]["mesh"]]
    acc = doc.json["accessors"][cage_mesh["primitives"][1]["attributes"]["WEIGHTS_0"]]
    bv = doc.json["bufferViews"][acc["bufferView"]]
    doc.bin[bv["byteOffset"]] ^= 1
    assert R.rig_hash(doc) != h


# ------------------------------------------------------------------ validator (SPEC 8)

@pytest.mark.skipif(validator() is None, reason="no glTF-Validator (set B2C_GLTF_VALIDATOR)")
def test_validator(subject, tmp_path):
    path, info = subject
    validate(path)
    doc = rig_subject(path, info)
    validate(path)
    sk = read.skeleton(doc)
    C.write(tmp_path / "c.clip.glb", doc, "rest", 30.0, np.tile(sk.rest_q, (2, 1, 1)), np.tile(sk.rest_t, (2, 1, 1)),
            residual=np.zeros((2, len(R.cage(doc).verts), 3)))
    validate(tmp_path / "c.clip.glb")


def test_rigged_splat_keeps_the_binding_labels(subject):
    """b2ctrain binds by seg_label: the rigged splat must read back with the labels its binding was made with."""
    path, info = subject
    doc = rig_subject(path, info)
    rigged, plain = read.splat(doc), read.splat(doc, doc.node_index("b2c_splat"))
    assert rigged["node"] == doc.extension(R.RIG)["splatNode"]
    np.testing.assert_array_equal(rigged["seg_label"], plain["seg_label"])
    np.testing.assert_array_equal(rigged["seg_conf"], plain["seg_conf"])
