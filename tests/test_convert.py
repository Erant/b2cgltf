"""The splat converter: the PLY is the trainer's bare vertex properties and reads back to the splat."""
from __future__ import annotations

import numpy as np
import pytest

from b2cgltf import convert, read
from b2cgltf.b2crunner import convert as C
from b2cgltf.document import Document

from fixtures import W_TEST, synthetic


def test_ply_is_the_bare_trainer_properties(tmp_path):
    glb_path = tmp_path / "scene.glb"
    synthetic(glb_path, W=W_TEST)
    info = convert.convert(glb_path, tmp_path / "out.ply")
    assert info["format"] == "ply"
    fields, comments = C.read_trainer_ply(tmp_path / "out.ply")
    assert comments == []
    want = read.to_trainer_ply_fields(read.splat(Document.load(glb_path)))
    assert list(fields) == list(want) and info["splats"] == len(want["x"])
    for k in want:
        assert np.array_equal(fields[k], want[k])


def test_format_follows_extension_or_flag(tmp_path):
    assert convert.format_of("a.PLY") == "ply"
    assert convert.format_of("a.sog") == "sog"
    assert convert.format_of("a.bin", "ply") == "ply"
    with pytest.raises(ValueError):
        convert.format_of("a.bin")
