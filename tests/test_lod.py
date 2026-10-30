"""SPEC 4.7 / 5.3 / 7.1 / 8: splat LOD chains."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from b2cgltf import lod, read
from b2cgltf.b2crig import rig as R
from b2cgltf.document import Document, RuleError
from b2cgltf.validate import validate, validator

from fixtures import synthetic
from test_core import rig_subject


@pytest.fixture
def subject(tmp_path):
    doc, info = synthetic(tmp_path / "scene.glb")
    return tmp_path / "scene.glb", info


def level_fields(f: dict, n: int) -> dict:
    return {k: v[:n] for k, v in f.items()}


def chain_b2crunner(path, info, counts=(40, 20), **kw):
    doc = Document.load(path)
    w = doc.writer("b2crunner", tool="test")
    nodes = [lod.add_level_splat(w, level_fields(info["fields"], c), f"b2c_splat_lod{i + 1}") for i, c in enumerate(counts)]
    rec = lod.add_lods(w, doc.node_index("b2c_splat"), [{"node": n, "method": "test"} for n in nodes], **kw)
    w.save(path)
    return Document.load(path), nodes, rec


def test_source_hash_is_the_spec_definition(subject):
    path, info = subject
    doc = Document.load(path)
    s = read.splat(doc, doc.node_index("b2c_splat"))
    h = hashlib.sha256()
    for a in (s["pos"], s["rot"], s["scale"], s["opacity"]):
        h.update(np.asarray(a, "<f4").tobytes())
    for k in range(s["sh"].shape[1]):          # d = 0..3, n = 0..2d is coefficient order
        h.update(np.asarray(s["sh"][:, k], "<f4").tobytes())
    assert lod.source_hash(doc, doc.node_index("b2c_splat")) == "sha256:" + h.hexdigest()


def test_b2crunner_chain(subject):
    path, info = subject
    doc, nodes, rec = chain_b2crunner(path, info, coverage=True)
    base = doc.node_index("b2c_splat")
    assert read.lods(doc) == [(base, 60, None), (nodes[0], 40, 960.0), (nodes[1], 20, 480.0)]
    nd = doc.json["nodes"][base]
    assert nd["extensions"]["MSFT_lod"]["ids"] == nodes
    assert "MSFT_lod" in doc.json["extensionsUsed"] and "MSFT_lod" not in doc.json.get("extensionsRequired", [])
    cov = nd["extras"]["MSFT_screencoverage"]
    assert len(cov) == 3 and cov[0] > cov[1] > cov[2] == 0
    assert read.current_splat(doc) == base     # level nodes are in no scene
    assert lod.check(doc) == []


def test_levels_must_shrink_and_match(subject):
    path, info = subject
    doc = Document.load(path)
    w = doc.writer("b2crunner", tool="test")
    big = lod.add_level_splat(w, level_fields(info["fields"], 60), "b2c_splat_lod1")
    with pytest.raises(RuleError, match="not fewer"):
        lod.add_lods(w, doc.node_index("b2c_splat"), [{"node": big}])
    f = level_fields(info["fields"], 30)
    f = {k: v for k, v in f.items() if not k.startswith("seg_")}
    nolab = lod.add_level_splat(w, f, "b2c_splat_lod2")
    with pytest.raises(RuleError, match="_SEG_LABEL"):
        lod.add_lods(w, doc.node_index("b2c_splat"), [{"node": nolab}])
    ok = lod.add_level_splat(w, level_fields(info["fields"], 30), "b2c_splat_lod3")
    with pytest.raises(RuleError, match="strictly decrease"):
        lod.add_lods(w, doc.node_index("b2c_splat"), [{"node": ok, "maxScreenHeightPx": 100},
                                                      {"node": lod.add_level_splat(w, level_fields(info["fields"], 10), "b2c_splat_lod4"),
                                                       "maxScreenHeightPx": 200}])


def test_only_own_base_and_only_new_keys(subject, tmp_path):
    path, info = subject
    doc, nodes, _ = chain_b2crunner(path, info)
    w = doc.writer("b2crig", tool="test")
    with pytest.raises(RuleError, match="own splat"):
        lod.add_lods(w, doc.node_index("b2c_splat"), [{"node": nodes[0]}])
    w = doc.writer("b2crunner", tool="test")
    with pytest.raises(RuleError, match="only added"):
        w.extend_node(doc.node_index("b2c_splat"), extensions={"MSFT_lod": {"ids": []}})
    # a direct edit of an extended node is still refused
    w.extend_node(doc.node_index("b2c_splat"), extras={"note": 1})
    doc.json["nodes"][doc.node_index("b2c_splat")]["extensions"]["B2C_lod"]["levels"][0]["count"] = 1
    with pytest.raises(RuleError, match="changed"):
        w.save(tmp_path / "x.glb")


def test_stale_chain_is_dropped(subject):
    path, info = subject
    doc, nodes, _ = chain_b2crunner(path, info)
    base = doc.node_index("b2c_splat")
    acc = doc.json["accessors"][doc.json["meshes"][doc.json["nodes"][base]["mesh"]]["primitives"][0]["attributes"]["POSITION"]]
    doc.bin[doc.json["bufferViews"][acc["bufferView"]]["byteOffset"]] ^= 1
    msgs = []
    assert lod.chain(doc, log=msgs.append) == [(base, 60, None)]
    assert "stale" in msgs[0]


def test_b2crig_levels_after_rigging(subject, tmp_path):
    """b2crig appends levels to an already-rigged file: skinned level nodes, B2CRIG_lod on b2crig_splat."""
    path, info = subject
    doc = rig_subject(path, info)
    w = doc.writer("b2crig", tool="test")
    nodes = []
    for i, c in enumerate((40, 20)):
        bind = R.Binding(info["fi"][:c], info["b"][:c, 1:], np.zeros((c, 3)))
        nodes.append(R.add_level(doc, w, bind, name=f"b2crig_splat_lod{i + 1}", splat_fields=level_fields(info["fields"], c)))
    base = doc.extension(R.RIG)["splatNode"]
    lod.add_lods(w, base, [{"node": n, "method": "test"} for n in nodes])
    w.save(path)
    doc = Document.load(path)
    assert [c for _, c, _ in read.lods(doc)] == [60, 40, 20]
    assert "B2CRIG_lod" in doc.json["nodes"][base]["extensions"]
    lv = doc.json["nodes"][nodes[0]]
    assert lv["skin"] == 0
    prim = doc.json["meshes"][lv["mesh"]]["primitives"][0]
    assert R.BIND in prim["extensions"] and "_B2CRIG_SEG_LABEL" in prim["attributes"]


def test_b2crig_reuses_b2crunner_levels(subject):
    """The no-retrain case: b2crig_splat references b2c_splat, its levels reference b2crunner's levels (R1a); the
    sources match because the accessors are the same."""
    path, info = subject
    doc, b_nodes, b_rec = chain_b2crunner(path, info)
    doc = rig_subject(path, info)
    w = doc.writer("b2crig", tool="test")
    nodes = [R.add_level(doc, w, R.Binding(info["fi"][:c], info["b"][:c, 1:], np.zeros((c, 3))),
                         name=f"b2crig_splat_lod{i + 1}", reuse_node=bn) for i, (bn, c) in enumerate(zip(b_nodes, (40, 20)))]
    base = doc.extension(R.RIG)["splatNode"]
    rec = lod.add_lods(w, base, [{"node": n} for n in nodes])
    assert rec["levels"][0]["source"] == b_rec["levels"][0]["source"]
    for n, bn in zip(nodes, b_nodes):
        at = doc.json["meshes"][doc.json["nodes"][n]["mesh"]]["primitives"][0]["attributes"]
        bat = doc.json["meshes"][doc.json["nodes"][bn]["mesh"]]["primitives"][0]["attributes"]
        assert at["POSITION"] == bat["POSITION"] and at["_B2CRIG_SEG_LABEL"] == bat["_SEG_LABEL"]


def test_superseded_chain_is_dead(subject):
    """R7: b2crig's retrained splat supersedes b2c_splat and with it b2crunner's chain."""
    path, info = subject
    chain_b2crunner(path, info)
    doc = Document.load(path)
    w = doc.writer("b2crig", tool="test")
    V, F = info["V"], info["F"]
    R.enhance(doc, w, [R.Layer("body", [], F)], R.Binding(info["fi"], info["b"][:, 1:], np.zeros((60, 3))),
              render={"maxGrowth": 1.15, "minConf": 0.5}, retrained_splat=info["fields"])
    w.save(path)
    doc = Document.load(path)
    assert read.lods(doc) == [(doc.extension(R.RIG)["splatNode"], 60, None)]


@pytest.mark.skipif(validator() is None, reason="no glTF-Validator (set B2C_GLTF_VALIDATOR)")
def test_validator_with_levels(subject):
    path, info = subject
    chain_b2crunner(path, info, coverage=True)
    validate(path)
