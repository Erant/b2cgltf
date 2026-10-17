#!/usr/bin/env python3
"""Convert a subject .glb's current splat to .ply or .sog: `tools/convert.py scene.glb out.ply [--format sog]`.

A runnable front for `python -m b2cgltf.convert` (see its docstring); works from a checkout without installing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from b2cgltf.convert import main  # noqa: E402

if __name__ == "__main__":
    main()
