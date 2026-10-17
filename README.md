# b2cgltf

The only reader and writer of the b2c glTF subject and clip files. `SPEC.md` is the format and the contract
between b2crunner and b2crig.

```
pip install -e .            # numpy only
pip install -e '.[web]'     # web.py's subject form: Pillow, SciPy, and splat-transform (SPLAT_TRANSFORM or PATH)
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
| `b2cgltf/web.py` | shared | the web forms (section 10): `python -m b2cgltf.web subject scene.glb scene.sog`, `… clip x.clip.glb x.clip.zip` |
| `b2cgltf/convert.py` | shared | the current splat as a plain file: `python -m b2cgltf.convert scene.glb out.ply` (bare trainer vertex properties) or `--format sog` (via splat-transform) |
| `tools/convert.py` | shared | a runnable front for `b2cgltf.convert`: `tools/convert.py scene.glb out.ply` |
| `b2cgltf/blender/` | shared | the Blender 5.3+ add-on "b2c Subjects" (below); `posing.py` and `correctives.py` are numpy only |
| `tools/build_blender_addon.py` | shared | packages the add-on, b2cgltf included, as a Blender extension zip in `dist/` |

The JSON stays a plain dict, so load → save reproduces a file exactly (SPEC 8, preservation). pygltflib was tried
and rejected because it writes default fields (`"normalized": false`, sampler wraps) into every object.

`tests/fixtures.py` writes a minimal b2crunner-shaped subject for the tests (with a non-identity `W`); it is not
b2crunner's writer.

## Blender add-on

Blender 5.3 renders Gaussian splats natively (a `GAUSSIAN_SPLAT` point cloud). The add-on imports b2c files into it:

- **File > Import > b2c Subject**: a rigged subject becomes an armature, the cage (skinned by an Armature modifier;
  hidden in renders) and the splat, which a Geometry Nodes modifier poses from the evaluated cage exactly as b2ctrain
  does (cage.cu `pose_bound`: position, rotation, growth within `maxGrowth`, SH turned with the triangle). An unrigged
  subject becomes the static splat. Blender's own glTF importer cannot take a subject file: it gives the skinned
  splat an Armature modifier, which point clouds do not have.
- **File > Import > b2c Clip** (with the rig selected): each clip becomes an action. The clip's residual (SPEC 6) is
  read from the clip file at play time, so keep the clip files next to the .blend (or where they were imported from).
- **Pose correctives**: MHR's pose correctives are evaluated live from the armature (posing by hand, or an edited
  clip), when the add-on preferences name the corrective weights. Export them once where torch and the MHR model are:
  `python -m b2cgltf.blender.correctives mhr_correctives.npz`. Without them a clip still plays exactly (its residual
  holds its correctives); hand poses get plain skinning.
- The sidebar's **b2c** tab (3D view, N) shows the rig's clip and blends the clip residual and the correctives.

The posing runs in Geometry Nodes, so a .blend renders the splat without the add-on; the correctives and the clip
residual need it (they run in a handler). Do not Apply Transform on the splat object: Blender 5.3 does not transform
splat rotations, scales and SH that way.

```
tools/build_blender_addon.py [--blender BLENDER]         # dist/b2c_subjects-<version>.zip (validated with --blender)
blender --command extension install-file -r user_default -e dist/b2c_subjects-<version>.zip
B2C_BLENDER=/path/to/blender pytest tests/test_blender.py # the add-on inside Blender, on the synthetic subject
```
