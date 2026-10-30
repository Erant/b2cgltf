"""The web forms (SPEC 10): the rig keeps its hash, the clip residual round-trips within its step, and the web
subject's binding follows the SOG's splat order."""
from __future__ import annotations

import zipfile

import numpy as np
import pytest

from b2cgltf import glb, read, web
from b2cgltf.b2crig import clip as C
from b2cgltf.b2crig import rig as R
from b2cgltf.document import Document

from fixtures import W_TEST, synthetic


@pytest.fixture
def rigged(tmp_path):
    """The synthetic subject rigged with a non-zero, per-splat binding offset."""
    path = tmp_path / "scene.glb"
    _, info = synthetic(path, W=W_TEST)
    doc = Document.load(path)
    w = doc.writer("b2crig", tool="test")
    n = len(info["fi"])
    offset = np.random.default_rng(1).normal(0, 0.3, (n, 3))
    R.enhance(doc, w, [R.Layer("body", [], info["F"])], R.Binding(info["fi"], info["b"][:, 1:], offset),
              render={"maxGrowth": 1.15, "minConf": 0.5})
    w.save(path)
    return path, Document.load(path)


def test_rig_glb_keeps_the_rig_and_drops_the_rest(rigged):
    _, doc = rigged
    js, bin_ = web._glb_loads(web.rig_glb(doc))
    rd = Document(js, bin_)
    assert R.rig_hash(rd) == R.rig_hash(doc)
    assert np.allclose(read.skeleton(rd).W, read.skeleton(doc).W)
    assert [m["name"] for m in js["meshes"]] == ["b2crig_cage"]
    assert "images" not in js and "cameras" not in js
    assert js["asset"]["extras"]["b2c_id"] == doc.id
    assert rd.extension(R.RIG)["splatNode"] is None


def test_clip_round_trip(rigged, tmp_path):
    _, doc = rigged
    sk, nv = read.skeleton(doc), len(R.cage(doc).verts)
    T = 6
    q = np.repeat(sk.rest_q[None], T, 0); t = np.repeat(sk.rest_t[None], T, 0)
    res = np.cumsum(np.random.default_rng(2).normal(0, 0.002, (T, nv, 3)), 0)
    C.write(tmp_path / "a.clip.glb", doc, "a", 30.0, q, t, residual=res)
    info = web.clip(tmp_path / "a.clip.glb", tmp_path / "a.clip.zip", step_mm=0.05)
    assert info["max_residual_error_m"] <= 0.025e-3 + 1e-12
    js, b = glb.read(tmp_path / "a.clip.glb")
    rx = js["animations"][0]["extensions"][C.RES]
    a = js["accessors"][rx["accessor"]]; v = js["bufferViews"][a["bufferView"]]
    stored = np.frombuffer(bytes(b), "<i2", count=T * nv * 3, offset=v.get("byteOffset", 0)).reshape(T, nv, 3) * rx["scale"] / 32767
    assert np.abs(web.read_clip_residual(tmp_path / "a.clip.zip") - stored).max() <= 0.025e-3 + 1e-9
    with zipfile.ZipFile(tmp_path / "a.clip.zip") as z:
        cj, cb = web._glb_loads(z.read("clip.glb"))
    C.check(doc, Document(cj, cb))   # the subject checks (b2c_id, rig hash, skeleton) still pass
    assert len(cj["animations"][0]["channels"]) == len(js["animations"][0]["channels"])


def test_clip_step_overflow_refused(rigged, tmp_path):
    _, doc = rigged
    sk, nv = read.skeleton(doc), len(R.cage(doc).verts)
    res = np.zeros((2, nv, 3)); res[1, 0, 0] = 0.05
    C.write(tmp_path / "b.clip.glb", doc, "b", 30.0, sk.rest_q[None].repeat(2, 0), sk.rest_t[None].repeat(2, 0), residual=res)
    with pytest.raises(ValueError, match="int16"):
        web.clip(tmp_path / "b.clip.glb", tmp_path / "b.clip.zip", step_mm=0.001)


def _have_subject_tools():
    try:
        import PIL, scipy  # noqa: F401
        web.splat_transform()
        return True
    except (ImportError, FileNotFoundError):
        return False


@pytest.mark.skipif(not _have_subject_tools(), reason="needs Pillow, SciPy and splat-transform (SPLAT_TRANSFORM)")
def test_subject_binding_follows_the_sog_order(rigged, tmp_path):
    path, doc = rigged
    web.subject(path, tmp_path / "scene.sog")
    got = web.read_subject(tmp_path / "scene.sog")
    s, b = read.splat(doc), R.binding(doc)
    # match by centre, independently of the converter (brute force)
    d = np.linalg.norm(got["pos"][:, None] - s["pos"][None].astype(np.float64), axis=2)
    j = d.argmin(1)
    assert d.min(1).max() < 1e-3
    assert np.array_equal(got["face"], b.face[j])
    assert np.abs(got["barycentric"] - b.barycentric[j]).max() <= 0.5 / 255 + 1e-6
    span = b.offset.max(0) - b.offset.min(0)
    assert (np.abs(got["offset"] - b.offset[j]) <= span / 65535 + 1e-6).all()
    assert R.rig_hash(got["rig"]) == R.rig_hash(doc)
    meta = got["meta"]
    assert meta["version"] == 2 and meta["count"] == len(s["pos"]) and meta["b2c"]["version"] == web.WEB_VERSION


# ------------------------------------------------------------------ LOD levels (SPEC 10.3)

def _rigged_with_levels(rigged, counts=(40, 20)):
    """The rigged fixture plus b2crig levels: the first `counts` splats, each with its own non-zero offsets."""
    from b2cgltf import lod
    path, doc = rigged
    s, b = read.splat(doc), R.binding(doc)
    f = read.to_trainer_ply_fields(s)
    w = doc.writer("b2crig", tool="test")
    nodes = []
    for i, c in enumerate(counts):
        bind = R.Binding(b.face[:c], b.barycentric[:c], b.offset[:c] * (i + 2))
        nodes.append(R.add_level(doc, w, bind, name=f"b2crig_splat_lod{i + 1}", splat_fields={k: v[:c] for k, v in f.items()}))
    lod.add_lods(w, doc.extension(R.RIG)["splatNode"], [{"node": n, "method": "test"} for n in nodes])
    w.save(path)
    return path, Document.load(path)


def _matches(got, doc, node):
    s, b = read.splat(doc, node), R.binding(doc, node)
    d = np.linalg.norm(got["pos"][:, None] - s["pos"][None].astype(np.float64), axis=2)
    j = d.argmin(1)
    assert d.min(1).max() < 1e-3 and len(np.unique(j)) == len(s["pos"])
    assert np.array_equal(got["face"], b.face[j])
    span = b.offset.max(0) - b.offset.min(0)
    assert (np.abs(got["offset"] - b.offset[j]) <= span / 65535 + 1e-6).all()
    assert R.rig_hash(got["rig"]) == R.rig_hash(doc)


@pytest.mark.skipif(not _have_subject_tools(), reason="needs Pillow, SciPy and splat-transform (SPLAT_TRANSFORM)")
def test_subject_lods_binds_every_level_in_its_order(rigged, tmp_path):
    path, doc = _rigged_with_levels(rigged)
    out = tmp_path / "lods"
    info = web.subject_lods(path, out)
    chain = read.lods(doc)
    assert [lv["count"] for lv in info["levels"]] == [c for _, c, _ in chain]
    import json
    lm = json.loads((out / "lod-meta.json").read_text())
    assert lm["filenames"] == ["0_0/meta.json", "1_0/meta.json", "2_0/meta.json"]
    assert lm["b2c"]["rig"] == "rig.glb" and [lv["maxScreenHeightPx"] for lv in lm["b2c"]["levels"]] == [None, 960, 480]
    for i, (node, _, h) in enumerate(chain):
        got = web.read_subject(out / f"{i}_0")
        _matches(got, doc, node)
        assert got["meta"]["b2c"]["level"] == {"index": i, "maxScreenHeightPx": h}
    web.subject_lods(path, out)   # a second run replaces the first
    with pytest.raises(ValueError, match="not empty"):
        (tmp_path / "other").mkdir(); (tmp_path / "other" / "x").write_text("")
        web.subject_lods(path, tmp_path / "other")


@pytest.mark.skipif(not _have_subject_tools(), reason="needs Pillow, SciPy and splat-transform (SPLAT_TRANSFORM)")
def test_subject_single_level(rigged, tmp_path):
    path, doc = _rigged_with_levels(rigged)
    node, _, h = read.lods(doc)[2]
    web.subject(path, tmp_path / "l2.sog", level=2)
    got = web.read_subject(tmp_path / "l2.sog")
    _matches(got, doc, node)
    assert got["meta"]["b2c"]["level"] == {"index": 2, "maxScreenHeightPx": h}
    with pytest.raises(ValueError, match="no level 3"):
        web.subject(path, tmp_path / "l3.sog", level=3)


@pytest.mark.skipif(not _have_subject_tools(), reason="needs Pillow, SciPy and splat-transform (SPLAT_TRANSFORM)")
def test_subject_lods_unrigged(tmp_path):
    from test_lod import chain_b2crunner
    path = tmp_path / "scene.glb"
    _, info = synthetic(path)
    chain_b2crunner(path, info)
    out = tmp_path / "lods"
    res = web.subject_lods(path, out)
    import json
    lm = json.loads((out / "lod-meta.json").read_text())
    assert not res["rigged"] and "rig" not in lm["b2c"] and not (out / "rig.glb").exists()
    assert lm["counts"] == [60, 40, 20]
    assert "b2c" not in json.loads((out / "1_0" / "meta.json").read_text())
