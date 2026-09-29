"""The Khronos glTF-Validator check of SPEC 8: zero warnings; no errors except MESH_PRIMITIVE_INVALID_ATTRIBUTE on
`KHR_gaussian_splatting:*` attributes, allowed only for the validator versions that predate the extension."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

# validator version -> the errors it reports although the file is right
ALLOW = {"2.0.0-dev.3.10": {("MESH_PRIMITIVE_INVALID_ATTRIBUTE", "/attributes/KHR_gaussian_splatting:")}}


class ValidationError(Exception):
    pass


def validator() -> str | None:
    return os.environ.get("B2C_GLTF_VALIDATOR") or shutil.which("gltf_validator")


def validate(path: str | Path, exe: str | None = None) -> dict:
    """Run the validator; raise ValidationError unless the file passes SPEC 8. Returns the report."""
    exe = exe or validator()
    if not exe:
        raise FileNotFoundError("no glTF-Validator: set B2C_GLTF_VALIDATOR or put gltf_validator on PATH")
    out = subprocess.run([exe, "-o", "-a", str(path)], capture_output=True, text=True)
    rep = json.loads(out.stdout)
    ver = rep.get("validatorVersion", "")
    allow = ALLOW.get(ver, set())
    bad = []
    for m in rep["issues"]["messages"]:
        if m["severity"] == 0 and not any(m["code"] == c and p in m.get("pointer", "") for c, p in allow):
            bad.append(m)
        elif m["severity"] == 1:
            bad.append(m)
    if bad:
        raise ValidationError(f"{path}: {len(bad)} errors/warnings (validator {ver}), first: " +
                              "; ".join(f"{m['code']} {m.get('pointer', '')}: {m['message']}" for m in bad[:5]))
    return rep
