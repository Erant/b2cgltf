# b2c glTF: the subject file format shared by b2crunner and b2crig

Version 1.0 (2026-09-29). Agreed between b2crunner and b2crig. Supersedes b2crunner's
`docs/ply-header-records.md` (PLY header records) and `docs/tools.md` (`mhr.npz`) as the hand-off contract.

A subject is **one glTF 2.0 binary file** (`.glb`). b2crunner creates it with the splat, the body and the capture
data. b2crig **enhances it in place** with its rig, adding objects next to b2crunner's and never translating the
file into a private layout. Animation clips live in **separate clip files** that refer to the subject file.
At every stage the subject file still opens in an ordinary glTF viewer that supports `KHR_gaussian_splatting`.

This repository's package (`b2cgltf`) is the only reader and writer for these files. b2crunner and b2crig
depend on it; neither edits the glTF JSON by hand.

---

## 1. Files

| file | written by | content |
|---|---|---|
| **subject file** (`scene.glb`, the master) | created by b2crunner, enhanced by b2crig | splat, body, skeleton, capture cameras and images, rig |
| **clip file** (`<clip>.clip.glb`) | b2crig | one animation of the subject's skeleton plus its cage data (section 6) |
| **distribution file** | derived by the package from a subject file | a stripped copy (R8); never enhanced |

Per-clip training data (WAN frames, segmentations, fit datasets), learned correctives (`model.pt`), renders, the
training dataset (`colmap/`) and regenerable caches (`seen.npy`) are not in any of these files.

## 2. Conventions

- **Owners and prefixes.** b2crunner owns objects named `b2c_*`, extensions `B2C_*` and attributes `_B2C_*` (plus
  the plain `_SEG_*` attributes, section 4.2). b2crig owns `b2crig_*`, `B2CRIG_*` and `_B2CRIG_*`.
- **Frame rule.** Every array defined here is in the **b2crunner frame**: the local frame of the root nodes
  `b2c_world` / `b2c_skeleton`, never glTF world. That covers the splat, body, cameras, cage, binding offsets,
  residuals, and the posed splat b2ctrain produces. Both roots carry the same matrix `W` (b2crunner frame → glTF:
  +Y up, subject facing +Z, metres, right-handed). `W` is written explicitly even when it is the identity, which it
  is for b2crunner's output: b24be4's subject file renders upright and facing the camera in a third-party viewer with
  `W` = identity (checked 2026-09-29).
- **Units:** metres. **Precision:** float32 accessors throughout, with no float64 data. Rotation matrices are
  MAT3 accessors (column-major, as glTF defines them).
- **Cameras:** camera-to-world, OpenGL axes (the rotation's columns are the camera's right, up and back; it looks
  down −Z), intrinsics `fx fy cx cy` in pixels at `image_size` `[w, h]`.
- **Versions.** Every `B2C_*` / `B2CRIG_*` extension object has an integer `version`. A reader refuses a version
  it does not know.

## 3. Rules for a file with two writers

- **R1 Ownership.** A tool never edits or deletes objects owned by the other. It adds its own objects beside them.
  - **R1a Referencing is not editing.** A tool may refer to the other's objects by index. For example, it may
    reuse its accessors in its own primitives, skin its own meshes with the other's skin, or target the other's
    joint nodes.
  - **R1b Scenes.** Scene 0 is b2crunner's. b2crig adds its own scene and may set the document's default `scene`
    to it. The top-level `scene` index is the only shared field both tools may set.
- **R2 Append-only indices.** A writer only appends to `nodes`, `meshes`, `accessors`, `bufferViews`, `images`,
  `skins`, `scenes` and the other arrays. Every existing index stays valid.
  - **R2a** Accessors referenced only from an extension are legal.
- **R3 One buffer.** The binary data is buffer 0, the GLB's BIN chunk. There are no external `.bin` files. An
  enhancing writer appends its bytes to buffer 0 and rewrites the file. The existing bytes stay identical.
- **R4 Nothing required.** No `B2C_*` / `B2CRIG_*` extension goes in `extensionsRequired`, only in
  `extensionsUsed`.
- **R5 Identity and provenance.**
  - `asset.extras.b2c_id` is a UUID that b2crunner assigns when it creates the file. It never changes afterwards,
    and clip files refer to it.
  - `asset.extras.b2c_history` is an append-only list of `{tool, version, commit, time, added: [...]}`, one entry
    per writer pass.
- **R6 Versioned extensions.** See section 2.
- **R7 Replace means supersede.** To replace something the other tool owns, add a new object with
  `"extras": {"supersedes": <node index>}`. The old object stays.
- **R8 Stripping makes a derived file.** Removing anything re-indexes the file. The result is a distribution file,
  with `asset.extras.b2c_stripped` listing what was removed, and the package refuses to enhance it. Only the
  unstripped subject file is enhanced.

## 4. What b2crunner writes

### 4.1 Scene graph

```
scene 0 "b2c"
├─ node "b2c_world"             matrix W
│  ├─ node "b2c_splat"          mesh: the splat (4.2)
│  ├─ node "b2c_cam_orbit_NNN"  glTF cameras, for viewing only (4.4)
│  └─ node "b2c_cam_final_NNN"
├─ node "b2c_skeleton"          matrix W (equal to b2c_world's; the package checks this)
│  └─ MHR joint nodes           in MHR joint order, parented by joint_parents
└─ node "b2c_body"              mesh: the MHR body, skinned (4.3)
```

- A skinned mesh node must be a scene root (glTF ignores its transform), so `b2c_body` is one.
- The same root may appear in several scenes; b2crig's scene lists `b2c_skeleton` but not `b2c_world` (5.1).
- Joint nodes are named with MHR's joint names if the model provides them, otherwise `b2c_joint_NNN`. Their
  **order** is the contract: node order = MHR joint index.

### 4.2 The splat: `KHR_gaussian_splatting`

One `POINTS` primitive carries
`"extensions": {"KHR_gaussian_splatting": {"kernel": "ellipse", "colorSpace": "srgb_rec709_display"}}`.

| trainer PLY property | glTF attribute | conversion |
|---|---|---|
| `x y z` | `POSITION` (float VEC3, with min/max) | none |
| `rot_0..3` (w x y z) | `KHR_gaussian_splatting:ROTATION` (float VEC4) | reorder to x y z w, normalise |
| `scale_0..2` (log) | `KHR_gaussian_splatting:SCALE` (float VEC3) | `exp` |
| `opacity` (logit) | `KHR_gaussian_splatting:OPACITY` (float) | `sigmoid` |
| `f_dc_0..2` | `KHR_gaussian_splatting:SH_DEGREE_0_COEF_0` | none (the same 0.5 bias as 3DGS) |
| `f_rest_0..44` (channel-major: 15 R, 15 G, 15 B) | `KHR_gaussian_splatting:SH_DEGREE_{1,2,3}_COEF_n` (VEC3) | regroup coefficient-major |
| `seg_label` | `_SEG_LABEL` (UNSIGNED_BYTE SCALAR, `byteStride` 4: glTF aligns every vertex-attribute element to 4 bytes): Sapiens2 Goliath class ids | cast |
| `seg_conf` | `_SEG_CONF` (float) | none |
| - | `COLOR_0` | diffuse colour from SH0 (`0.5 + 0.2820948 · f_dc`, clamped), for viewers without splat support |

The splat is stored in display form only, with no raw logit or log-scale copies. The writer:

- **refuses** a splat whose float32 opacity is exactly 0 or 1, and a `seg_label` that is not an integer in 0–255;
- **drops** `ev_*`, `open_*`, `cage_fill` and `cage_gate_*`, and logs what it dropped (the `open_*` family is
  b2crig's, 5.3);
- **refuses** any other per-vertex property it does not know.

The splat is not skinned. The deformation model is b2crig's cage binding (5.3).

### 4.3 The body: skinned mesh + `B2C_mhr`

- **Mesh `b2c_body`.** The MHR body replayed at the refitted pose (the pose the splat stands in), with its faces.
  The **bind pose** is this pose, not MHR's zero pose.
- **Skin weights.** They are MHR's sparse skin weights, untruncated, stored as `JOINTS_n/WEIGHTS_n` in sets of
  four. Within a vertex they are sorted by weight. A slot with weight 0 has joint index 0. The weights sum to 1
  within float32, and the writer refuses if MHR's do not.
- **Skin.** `joints` = the MHR joint nodes in order, `skeleton` = `b2c_skeleton`. The joint nodes' transforms
  relative to `b2c_skeleton` are the refitted joints: position = the fitted joint, rotation = the fitted global
  rotation in the joints' frame, which is record version 2 of the old PLY header. Every joint rotation has det +1,
  and the writer asserts it. `inverseBindMatrices` are the inverses of those transforms, relative to
  `b2c_skeleton`, so `joint_world · IBM = W` at bind pose.
- **Node extension `B2C_mhr`** on `b2c_body` holds what an exact MHR replay or re-pose needs. Plain skinning cannot
  express MHR's pose correctives. The joint nodes are the only source of joint positions and rotations.

```json
"B2C_mhr": {
  "version": 1,
  "model": {"repo": "facebook/sam-3d-body-dinov3", "file": "assets/mhr_model.pt", "sha256": "<hex>"},
  "sourceRecordVersion": 2,
  "world_from_raw": {"scale": <float>, "rotation": <accessor MAT3>, "translation": [x, y, z]},
  "flip": [1, -1, -1],
  "pose_params": {"global_rot": <accessor>, "body_pose_params": <accessor>, "hand_pose_params": <accessor>,
                  "scale_params": <accessor>, "shape_params": <accessor>, "expr_params": <accessor>,
                  "global_trans": <accessor>, "scale_offsets": <accessor>},
  "model_params": <accessor, float[204]>,
  "hand_idx": <accessor, int>,
  "jointNames": ["..."]
}
```

- `world_from_raw` maps SAM-3D-Body's raw frame to the b2crunner frame:
  `p = scale · raw @ rotation.T + translation`, with the MHR output first flipped into the raw frame by
  `diag(flip)`.
- `pose_params` are in the raw frame (`global_trans` in metres). `model_params` is the TorchScript model's full
  input row: global translation ×10, global rotation, 130 body params with the hand PCA expanded, then 68 scales. A
  consumer edits its body slots and leaves `hand_idx` alone.
- `sourceRecordVersion` is 1 when a migration converted an old v1 PLY header (section 4.6), otherwise 2.
- `jointNames` is present only if MHR provides names.
- The model is referenced by name and hash, never by path. Lookup rules are in 7.3.

### 4.4 Capture: document extension `B2C_orbit`

`B2C_orbit` sits in the root `extensions` object, so it does not depend on which scene is the default.

```json
"B2C_orbit": {
  "version": 1,
  "migrated": false,
  "helix": {"n_frames": 81, "n_loops": 1, "amplitude_deg": 0.0, "lead_in_deg": 0.0, "lead_out_deg": 0.0},
  "extension": {"before": 0, "after": 0, "overlap_before": 0, "overlap_after": 0, "tilt_deg": 0.0},
  "pass_frames": 81,
  "anchor_frame_index": 40,
  "orbit_cameras": {"rotation": <accessor MAT3>, "position": <accessor VEC3>,
                    "intrinsics": <accessor VEC4>, "image_size": [w, h]},
  "final_cameras": {"rotation": <accessor MAT3>, "position": <accessor VEC3>,
                    "intrinsics": <accessor VEC4>, "image_size": [w, h],
                    "names": ["frame_00001_.png", "..."], "dataset": "../colmap/"},
  "extras": {"orbit_target": [x, y, z], "original_focal_length": <float>, "...": "..."},
  "prompt": {},
  "settings": {},
  "images": {"reference": <image index>, "anchor": <image index>, "front": <image index>}
}
```

- **Required fields:** `version` and `final_cameras`. All the others may be missing, for instance in migrated
  files (4.6).
- **`orbit_cameras`:** the path the frames were denoised on, before camera refinement, at render resolution.
  They are unnamed; the index is the frame index.
- **`final_cameras`:** the cameras the splat was trained on (refined, deliverable resolution).
  - `names` are the training images' names, in the dataset's `images.txt` order. The writer checks the count, the
    order and that the poses agree to float32.
  - `dataset` is an optional relative URI hint for the training dataset. The dataset may be absent.
- **`extras`, `prompt`, `settings`:** the run's dataset extras (for example `orbit_target`), the subject description
  the denoise prompts used, and the run's seed, resolution and framing.
- **`anchor_frame_index`:** the photograph's frame on the orbit path.
- The `b2c_cam_*` glTF camera nodes are for viewing only. glTF cameras have no principal point and no fx ≠ fy, so
  the accessors above are authoritative.

### 4.5 Images

`reference`, `anchor` and `front` are PNGs embedded as glTF `images` (`bufferView` + `mimeType: image/png`) and
referenced from `B2C_orbit.images`. `front` is the identity reference b2crig uses for clips.

### 4.6 Writing

- **The pipeline's final step is strict.** It writes the subject file atomically (to a temporary file, then
  renamed), and only when the body fit and the orbit record both exist. Otherwise the run fails.
- **The migration tool is lenient, and records what is missing.** b2crunner's `export_glb` converts an existing
  delivery (the trainer PLY with its header records, plus the PNGs beside it).
  - A v1 body record gets its rotations converted with `rotation @ diag(1, -1, -1) @ rotation.T @ global_rots` and
    `sourceRecordVersion: 1`.
  - Without an orbit record, it writes `B2C_orbit` with `"migrated": true`, `final_cameras` taken from the run's
    COLMAP dataset (with names), and `extras.orbit_target` when the run's extras have it.
- The trainer PLY stays b2crunner's and b2ctrain's internal format. The subject file is the delivery.

## 5. What b2crig writes into the subject file

### 5.1 Scene

```
scene 1 "b2crig"                (b2crig sets the document's default scene to it)
├─ node "b2c_skeleton"          b2crunner's root, listed again
├─ node "b2crig_cage"           mesh b2crig_cage, skin = b2crunner's skin     (skinned: a scene root)
└─ node "b2crig_splat"          mesh b2crig_splat, skin = b2crunner's skin    (skinned: a scene root)
```

`b2c_world` is not listed in scene 1, so the rest-pose splat is not shown twice.

### 5.2 The cage: mesh `b2crig_cage`

- **Primitives.** One TRIANGLES primitive per cage layer, in b2ctrain's cage order, which is also its vertex and
  face order. Layer 0 is the MHR body; then come garment, hair and face layers.
- **Layer 0** reuses `b2c_body`'s `POSITION` and `JOINTS_n/WEIGHTS_n` (R1a). Its own index accessor holds the
  expression-stable body faces.
- **Other layers** have their own `POSITION`, `JOINTS_0/1`, `WEIGHTS_0/1` (up to 8 joints) and indices. An offset
  layer takes its body vertex's weights exactly. A rigid hair shell has one joint at weight 1. A skirt shell has its
  own smoothed weights.
- **`_B2CRIG_BODY_INDEX`** (on offset layers) is the body vertex each layer vertex copies.
- **Document extension `B2CRIG_rig`:**
  - `version`;
  - `layers[]`: `{name, classes: [Sapiens2 ids], primitive, vertexOffset, vertexCount, faceOffset, faceCount}`;
  - `render`: `{maxGrowth, fadeStart, fadeEnd, fillStart, fillEnd, minConf}`, b2ctrain's render settings;
  - `bodyLayer`, `cageNode`, `splatNode`.

### 5.3 The rigged splat: mesh `b2crig_splat`

- **Primitive.** One `POINTS` primitive with `KHR_gaussian_splatting`. It reuses `b2c_splat`'s accessors (R1a), or,
  after a b2crig retrain, has its own and supersedes `b2c_splat` (R7). Scene 0 keeps showing exactly what b2crunner
  delivered.
- **Preview skin.** `JOINTS_0/WEIGHTS_0` are the bound triangle's vertex weights blended by barycentrics, top 4.
  They give approximate motion in generic viewers that skin points; b2crig's players use the binding.
- **Primitive extension `B2CRIG_splat_cage`:**
  - `version`;
  - b2ctrain's own binding (`b2ctrain render --export-binding`), per splat:
    - `face`: uint32, a global cage face index; `0xFFFFFFFF` = unbound;
    - `barycentric`: VEC2;
    - `offset`: VEC3, in the canonical triangle frame divided by its size;
  - optionally the dual binding: `altSplat`, `altFace`, `altBarycentric`, `altOffset`, `altWeight`;
  - `posing`: the posing rule the binding is for (e.g. `"b2ctrain cage.cu pose_bound v1"`).
- **Per-splat state** (attributes on b2crig's primitive): `_B2CRIG_OPEN_R/G/B`, `_B2CRIG_OPEN_DOPACITY`,
  `_B2CRIG_CAGE_FILL`, `_B2CRIG_GATE_A/_B`.
- **`_B2CRIG_SEG_LABEL` / `_B2CRIG_SEG_CONF`**: the labels the binding was made with, so a consumer that binds again
  (b2ctrain) gets the same binding. They are b2crunner's `_SEG_*` accessors, referenced (R1a), or a retrained
  splat's own.
- **Optional document extension `B2CRIG_cage_app`.** The pose-dependent appearance MLP (b2ctrain's `.app`):
  `{version, params, latents, latentSize, ...scalars}`. `params` and `latents` are flat float accessors; `latents`
  holds `latentSize` values per cage vertex.

## 6. Clip files

Clips are **separate files**, so the subject file stays the size of the rig, and enhancing it never rewrites
animation data. A clip file is written by b2crig and owned by it entirely; R1–R8 apply to it on its own. b2crig
may refine this section; a change bumps `B2CRIG_clip`'s version.

- **Skeleton.** A clip file contains a copy of the subject's skeleton: a root with `W` and the joint nodes with the
  same names, order, parents and rest transforms. It needs a copy because glTF animations can only target nodes in
  their own file.
- **Animation.** One animation per clip: rotation + translation channels on every joint node, one key per frame,
  LINEAR.
- **Document extension `B2CRIG_clip`:**
  - `version`;
  - `name` (the clip's name) and `fps`;
  - `skeleton`: `{"root": <node>, "joints": [<node>, ...]}`, the copy's root and its joint nodes in MHR joint order;
  - `subject`: `{"id": <the subject file's asset.extras.b2c_id>, "uri": <relative path hint>}`;
  - `rig`: `{"cageSha256": <hex>, "cageVertexCount": <int>, "jointCount": <int>}`. `cageSha256` is a sha256 over:
    1. for each primitive of `b2crig_cage` in order: its `POSITION` bytes, its index bytes, then its
       `JOINTS_n` and `WEIGHTS_n` bytes (n ascending, JOINTS before WEIGHTS);
    2. then the skin's inverse bind matrices, every joint's rest rotation and every joint's rest translation, as
       float32 little-endian.

    A residual is only valid against exactly this cage, weighting and skeleton.
- **Animation.** Named `b2crig_<clip name>`. The time of frame i is i / fps.
- **Animation extension `B2CRIG_cage_residual`:** `{version, accessor, scale, frames, vertices}`. Per frame and
  cage vertex, b2crig's posed cage minus plain skinning, in the b2crunner frame. It is one normalised SHORT VEC3
  accessor with `frames × vertices` elements, frame-major, × `scale` metres. This one term holds pose correctives,
  expressions, hair dynamics and fit corrections. Plain skinning alone is off by centimetres (p99 27 mm on b24be4).
- **Animation extension `B2CRIG_open_gate`** (optional): `{version, accessor, frames, vertices, rangeDeg: 180}`.
  The per-frame, per-cage-vertex gate angle, as normalised UNSIGNED_SHORT × `rangeDeg`.
- **Animation extension `B2CRIG_motion`** (optional): the clip's MHR inputs, e.g. `body_params` [T,130],
  `expr` [T,72], `root_R` [T,3,3], `root_t` [T,3], `root_c` [3], `global_trans` [T,3]. Each is stored as
  `{"accessor": <float SCALAR, flat>, "shape": [...]}`, so a reader that has MHR can re-pose exactly.
- **Checks.** A player refuses a clip if:
  - its `subject.id` differs from the subject file's `b2c_id`;
  - its `rig` does not match the subject file's current cage;
  - its skeleton copy (parents, rest TRS) differs from the subject's.

These refinements (b2cgltf's first implementation, 2026-09-29) were made before any clip file was written, so
`B2CRIG_clip` stays at version 1.

## 7. Reading

### 7.1 The current splat

The current splat is the splat node that the default scene lists and that no other node supersedes. The package
provides this as one function.

### 7.2 A posed frame

For a clip frame, in the b2crunner frame:

1. skin the cage with `W⁻¹ · joint_world · IBM` (the joints relative to the skeleton root; the identity at bind
   pose);
2. add the residual;
3. pose the splats by the binding;
4. apply `W` once for display.

Plain glTF players skin with `joint_world · IBM` and pick up `W` from the joints. That is the same result without
the residual.

### 7.3 The MHR model

A reader that needs the model resolves `B2C_mhr.model` in this order:

1. `$B2C_MHR_MODEL_DIR/<basename of file>`;
2. otherwise the Hugging Face cache for `repo`/`file` (local files only).

The sha256 must match. Readers that only need the bind-pose mesh, skeleton and weights never load the model.

## 8. Package obligations and validation

- **Preservation.** Load → save of a file with every extension defined here, with `extras` and with unknown
  extensions must give equal JSON and an unchanged buffer prefix. This is tested.
- **One write path.** Every write goes through the package's `append_*` functions. They enforce owner prefixes,
  append-only indices, extension versions and R8.
- **Frame-rule test.** The package tests posing (7.2) with a non-identity `W`.
- **Validator.** Every written file passes the Khronos glTF-Validator with zero warnings. The only errors allowed are
  `MESH_PRIMITIVE_INVALID_ATTRIBUTE` on `KHR_gaussian_splatting:*` attributes, because validator 2.0.0-dev.3.10
  predates the extension. The allow-list is pinned to the validator version and removed once a release knows the
  extension. `UNSUPPORTED_EXTENSION` and `UNUSED_OBJECT` (R2a) are info level.
- **Round trip.** trainer PLY → subject file → PLY is bit-exact for positions, rotations and SH, and within
  float32 tolerance for scale and opacity.

## 9. Size (for reference)

| content | size |
|---|---|
| splat, 440k Gaussians at SH degree 3 | ~105 MB, the same as the PLY |
| rig (b24be4: cage + binding + preview skin) | ~10 MB |
| clip file: joint channels | ~1 MB per 360 frames |
| clip file: residual (22.5k cage vertices) | ~49 MB per 360 frames |

Compression (`EXT_meshopt_compression`, or an SPZ extension of `KHR_gaussian_splatting`) can come later without
changing this layout.

## References

- KHR_gaussian_splatting: https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md
- glTF 2.0 specification: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- Design discussion (the proposal this spec came from, with both sides' measurements): `~/Documents/gltf_proposal.md`, v0.5
