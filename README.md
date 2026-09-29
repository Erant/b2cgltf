# b2cgltf

The only reader and writer of the b2c glTF subject and clip files. `SPEC.md` is the format and the contract
between b2crunner and b2crig.

```
pip install -e .            # numpy only
pytest tests                # set B2C_GLTF_VALIDATOR=/path/to/gltf_validator to include the Khronos validator
```

| module | owner | content |
|---|---|---|
| `b2cgltf/glb.py`, `document.py` | shared | GLB I/O. `Document` does typed accessor reads; `Writer` is the only write path, appends only, and checks R1–R8 on save |
| `b2cgltf/read.py` | shared | the current splat (7.1), the skeleton, posing in the b2crunner frame (7.2), splat → trainer PLY fields |
| `b2cgltf/splat.py` | shared | the SPEC 4.2 splat encoding, used by both writers |
| `b2cgltf/validate.py` | shared | the Khronos glTF-Validator check of SPEC 8, with the pinned allow-list |
| `b2cgltf/b2crunner/` | b2crunner | the subject writer (section 4) and its input conversions |
| `b2cgltf/b2crig/` | b2crig | the rig enhancement (`rig.py`, section 5) and clip files (`clip.py`, section 6) |

The JSON stays a plain dict, so load → save reproduces a file exactly (SPEC 8, preservation). pygltflib was tried
and rejected because it writes default fields (`"normalized": false`, sampler wraps) into every object.

`tests/fixtures.py` writes a minimal b2crunner-shaped subject for the tests (with a non-identity `W`); it is not
b2crunner's writer.
