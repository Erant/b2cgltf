#!/usr/bin/env python3
"""Package the Blender add-on (b2cgltf/blender) as a Blender 5.3+ extension zip, b2cgltf included.

    tools/build_blender_addon.py [--out dist/] [--blender BLENDER]

The zip's root is the extension: blender_manifest.toml, an __init__.py that registers b2cgltf.blender, and the
b2cgltf package (pure Python + numpy, which Blender ships). With --blender (or $BLENDER) the zip is checked with
`blender --command extension validate`. Install it with Edit > Preferences > Get Extensions > Install from Disk, or
`blender --command extension install-file -r user_default ZIP`.
"""
import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import b2cgltf  # noqa: E402

ID = "b2c_subjects"
MANIFEST = f"""schema_version = "1.0.0"
id = "{ID}"
version = "{b2cgltf.__version__}"
name = "b2c Subjects"
tagline = "Import b2c glTF subjects and clips as posable Gaussian splats"
maintainer = "Erant"
type = "add-on"
blender_version_min = "5.3.0"
license = ["SPDX:MIT"]
tags = ["Import-Export", "Animation"]

[permissions]
files = "Read b2c subject, clip and corrective-weight files"
"""
INIT = '''"""b2c Subjects: b2c glTF subjects and clips as posable Gaussian splats (b2cgltf.blender)."""
from .b2cgltf import blender as _addon


def register():
    _addon.register(__package__)


def unregister():
    _addon.unregister()
'''


def build(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{ID}-{b2cgltf.__version__}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("blender_manifest.toml", MANIFEST)
        z.writestr("__init__.py", INIT)
        z.write(ROOT / "LICENSE", "LICENSE")
        for p in sorted((ROOT / "b2cgltf").rglob("*.py")):
            if "__pycache__" not in p.parts:
                z.write(p, str(p.relative_to(ROOT)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    ap.add_argument("--blender", default=os.environ.get("BLENDER"))
    a = ap.parse_args()
    out = build(a.out)
    print(out)
    if a.blender:
        r = subprocess.run([a.blender, "--factory-startup", "--command", "extension", "validate", str(out)],
                           capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
        return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
