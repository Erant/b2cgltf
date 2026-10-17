"""Convert a subject .glb's current splat to a plain splat file.

    python -m b2cgltf.convert SCENE.glb OUT.ply            # the format follows the extension ...
    python -m b2cgltf.convert SCENE.glb OUT --format sog   # ... or --format says

    ply   the trainer PLY's bare vertex properties (binary little-endian float32, no header comments); keeps seg_label / seg_conf
    sog   SOG v2 through splat-transform (SPLAT_TRANSFORM or PATH); without the binding textures and rig of `web.subject`

The glTF stays the source: these are derived, and a posed or rigged subject is exported as stored (the rest pose).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import glb, read
from .document import Document

FORMATS = ("ply", "sog")


def splat_transform() -> str:
    exe = os.environ.get("SPLAT_TRANSFORM") or shutil.which("splat-transform")
    if not exe:
        raise FileNotFoundError("no splat-transform: put it on PATH or set SPLAT_TRANSFORM (npm i @playcanvas/splat-transform)")
    return exe


def ply_bytes(fields: dict) -> bytes:
    """A binary little-endian PLY of float32 vertex properties, in the dict's order."""
    names, n = list(fields), len(next(iter(fields.values())))
    hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n%send_header\n" % (n, "".join(f"property float {k}\n" for k in names))
    return hdr.encode() + np.stack([fields[k] for k in names], 1).astype("<f4").tobytes()


def write_ply(path: str | Path, fields: dict) -> None:
    glb.write_bytes(path, ply_bytes(fields))


def write_sog(path: str | Path, fields: dict, *, exe: str | None = None) -> None:
    """`fields` through splat-transform. It writes in place, so the output is staged and renamed (atomic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        ply, sog = Path(tmp) / "splat.ply", Path(tmp) / "splat.sog"
        write_ply(ply, fields)
        run = subprocess.run([exe or splat_transform(), "-w", str(ply), str(sog)], capture_output=True, text=True)
        if run.returncode != 0 or not sog.exists():
            raise RuntimeError(f"splat-transform failed:\n{run.stdout[-2000:]}\n{run.stderr[-2000:]}")
        glb.write_bytes(path, sog.read_bytes())


def format_of(dst: str | Path, fmt: str | None = None) -> str:
    fmt = (fmt or Path(dst).suffix.lstrip(".")).lower()
    if fmt not in FORMATS:
        raise ValueError(f"{dst}: unknown format {fmt!r} (one of {', '.join(FORMATS)}); give --format")
    return fmt


def convert(src: str | Path, dst: str | Path, fmt: str | None = None, *, exe: str | None = None) -> dict:
    """The subject's current splat as `fmt` (default: from dst's extension)."""
    fmt = format_of(dst, fmt)
    f = read.to_trainer_ply_fields(read.splat(Document.load(src)))
    if fmt == "ply":
        write_ply(dst, f)
    else:
        f.pop("seg_label", None); f.pop("seg_conf", None)
        write_sog(dst, f, exe=exe)
    return {"format": fmt, "splats": len(f["x"]), "bytes": Path(dst).stat().st_size}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="a subject .glb")
    ap.add_argument("dst", type=Path)
    ap.add_argument("--format", choices=FORMATS, help="default: dst's extension")
    args = ap.parse_args(argv)
    info = convert(args.src, args.dst, args.format)
    print(f"{args.dst}: {info['format']}, {info['splats']} splats, {info['bytes'] / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
