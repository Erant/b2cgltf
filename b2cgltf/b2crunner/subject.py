"""b2crunner's subject writer (SPEC section 4): creates the subject file from plain arrays.

The caller (b2crunner's `tools/export_glb.py`, or the pipeline's final step) gathers the inputs: the trainer PLY's
fields, the replayed MHR body and its record, the capture cameras and images. Nothing here imports b2crunner or MHR.

    path = write_subject(out, splat_fields, body, capture, tool="b2crunner export_glb", version=..., commit=...)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .. import splat as splat_enc
from ..document import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, UNSIGNED_INT, Document, RuleError
from . import convert

MHR_REPO, MHR_FILE = "facebook/sam-3d-body-dinov3", "assets/mhr_model.pt"


@dataclass
class Body:
    """The refitted MHR body in the b2crunner frame (SPEC 4.3)."""
    verts: np.ndarray                 # [V, 3], the replay at the refitted pose (the bind pose)
    faces: np.ndarray                 # [F, 3]
    skin_vertex: np.ndarray           # MHR's sparse skin triplets
    skin_joint: np.ndarray
    skin_weight: np.ndarray
    joint_parents: np.ndarray         # [J], -1 at the root
    joints: np.ndarray                # [J, 3], fitted joint positions
    global_rots: np.ndarray           # [J, 3, 3], record version 2 (the joints' frame)
    model_sha256: str
    world_from_raw: dict              # {"scale", "rotation" [3, 3], "translation" [3]}
    pose_params: dict                 # {name: array}, SAM-3D-Body's raw frame
    model_params: np.ndarray          # [204]
    hand_idx: np.ndarray
    source_record_version: int = 2
    joint_names: Optional[list] = None
    flip: tuple = (1.0, -1.0, -1.0)


@dataclass
class Cameras:
    """Camera-to-world, OpenGL axes, b2crunner frame (SPEC 2)."""
    rotation: np.ndarray              # [N, 3, 3]
    position: np.ndarray              # [N, 3]
    intrinsics: np.ndarray            # [N, 4] fx fy cx cy
    image_size: tuple                 # (w, h)
    names: Optional[list] = None
    dataset: Optional[str] = None


@dataclass
class Capture:
    """`B2C_orbit` (SPEC 4.4): only `final` is required."""
    final: Cameras
    orbit: Optional[Cameras] = None
    helix: Optional[dict] = None
    extension: Optional[dict] = None
    pass_frames: Optional[int] = None
    anchor_frame_index: Optional[int] = None
    extras: Optional[dict] = None
    prompt: Any = None
    settings: Optional[dict] = None
    images: dict = field(default_factory=dict)   # {"reference" | "anchor" | "front": PNG bytes}
    migrated: bool = False


def _camera_arrays(w, cams: Cameras, *, names_required: bool) -> dict:
    n = len(cams.position)
    rot = np.asarray(cams.rotation, np.float64).reshape(n, 3, 3)
    intr = np.asarray(cams.intrinsics, np.float64).reshape(n, 4)
    if names_required and (not cams.names or len(cams.names) != n):
        raise ValueError(f"final_cameras needs one name per camera ({n}), got {len(cams.names or [])}")
    out = {"rotation": w.add_accessor(rot.astype(np.float32), "MAT3"),
           "position": w.add_accessor(np.asarray(cams.position, np.float32).reshape(n, 3), "VEC3"),
           "intrinsics": w.add_accessor(intr.astype(np.float32), "VEC4"),
           "image_size": [int(v) for v in cams.image_size]}
    if cams.names:
        out["names"] = [str(s) for s in cams.names]
    if cams.dataset:
        out["dataset"] = cams.dataset
    return out


def _view_cameras(w, cams: Cameras, prefix: str) -> list[int]:
    """glTF perspective cameras for viewing (SPEC 4.4): a glTF camera looks down -Z with +Y up, which is the OpenGL
    camera-to-world rotation as it is."""
    iw, ih = (int(v) for v in cams.image_size)
    nodes = []
    for i in range(len(cams.position)):
        fy = float(cams.intrinsics[i][1])
        cam = w.add_camera({"name": f"{prefix}_{i:03d}", "type": "perspective",
                            "perspective": {"yfov": float(2 * np.arctan(ih / (2 * fy))), "aspectRatio": iw / ih,
                                            "znear": 0.01}})
        q = convert.quat_from_matrix(np.asarray(cams.rotation[i], np.float64).reshape(3, 3))
        nodes.append(w.add_node({"name": f"{prefix}_{i:03d}", "camera": cam,
                                 "translation": [float(v) for v in cams.position[i]],
                                 "rotation": [float(v) for v in q]}))
    return nodes


def _world_matrix(W: np.ndarray) -> list:
    W = np.asarray(W, np.float64).reshape(4, 4)
    r = W[:3, :3]
    if np.abs(W[3] - [0, 0, 0, 1]).max() > 1e-12 or abs(np.linalg.det(r) - 1) > 1e-6 or \
            np.abs(r @ r.T - np.eye(3)).max() > 1e-6:
        raise ValueError("W must be a rigid transform (rotation + translation); SPEC 2")
    return [float(v) for v in W.T.reshape(-1)]   # glTF matrices are column-major


def write_subject(path: str | Path, splat_fields, body: Body, capture: Capture, *, W: np.ndarray = np.eye(4),
                  tool: str = "b2crunner", version: str = "", commit: str = "", view_cameras: bool = True,
                  log=print) -> Path:
    """Create the subject file at `path` (atomically). Refuses inputs the spec does not allow."""
    doc, w = Document.new("b2crunner", tool=tool, version=version, commit=commit)
    matrix = _world_matrix(W)

    # 4.2 the splat, under b2c_world
    attrs = splat_enc.attributes(w, splat_fields, log=log)
    splat_mesh = w.add_mesh({"name": "b2c_splat", "primitives": [splat_enc.primitive(attrs)]})
    splat_node = w.add_node({"name": "b2c_splat", "mesh": splat_mesh})

    # 4.4 capture cameras (view nodes under b2c_world) + B2C_orbit
    cam_nodes = []
    if view_cameras:
        if capture.orbit is not None:
            cam_nodes += _view_cameras(w, capture.orbit, "b2c_cam_orbit")
        cam_nodes += _view_cameras(w, capture.final, "b2c_cam_final")
    world = w.add_node({"name": "b2c_world", "matrix": matrix, "children": [splat_node] + cam_nodes})

    # 4.3 the skeleton: joint nodes in MHR order, parented by joint_parents, under b2c_skeleton
    parents = np.asarray(body.joint_parents, np.int64).reshape(-1)
    n_joints = len(parents)
    names = body.joint_names or [f"b2c_joint_{i:03d}" for i in range(n_joints)]
    if len(names) != n_joints:
        raise ValueError(f"{len(names)} joint names for {n_joints} joints")
    jn = convert.joint_nodes(body.joints, body.global_rots, parents)
    log(f"b2cgltf.b2crunner: {n_joints} joints, float32 TRS composes to {jn['composed_error']:.2e} of the fit")
    first = len(doc.json.get("nodes", []))
    children = [[] for _ in range(n_joints)]
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(first + i)
    for i in range(n_joints):
        node = {"name": names[i], "translation": [float(v) for v in jn["translation"][i]],
                "rotation": [float(v) for v in jn["rotation"][i]]}
        if children[i]:
            node["children"] = children[i]
        if w.add_node(node, joint=True) != first + i:
            raise RuleError("joint nodes must be consecutive")
    roots = [first + i for i in range(n_joints) if parents[i] < 0]
    skeleton = w.add_node({"name": "b2c_skeleton", "matrix": matrix, "children": roots})

    # 4.3 the body: skinned mesh + skin + B2C_mhr
    verts = np.asarray(body.verts, np.float32).reshape(-1, 3)
    faces = np.asarray(body.faces, np.int64).reshape(-1, 3)
    if faces.min() < 0 or faces.max() >= len(verts):
        raise ValueError("body faces index outside the vertices")
    prim = {"mode": 4, "attributes": {
        "POSITION": w.add_accessor(verts, "VEC3", target=ARRAY_BUFFER, minmax=True)},
        "indices": w.add_accessor(faces.astype(np.uint32).reshape(-1), "SCALAR", component=UNSIGNED_INT,
                                  target=ELEMENT_ARRAY_BUFFER)}
    for s, (j, wt) in enumerate(convert.skin_sets(body.skin_vertex, body.skin_joint, body.skin_weight,
                                                  len(verts), n_joints)):
        prim["attributes"][f"JOINTS_{s}"] = w.add_accessor(j, "VEC4", target=ARRAY_BUFFER)
        prim["attributes"][f"WEIGHTS_{s}"] = w.add_accessor(wt, "VEC4", target=ARRAY_BUFFER)
    skin = w.add_skin({"name": "b2c_body", "joints": [first + i for i in range(n_joints)], "skeleton": skeleton,
                       "inverseBindMatrices": w.add_accessor(jn["inverse_bind"], "MAT4")})
    body_mesh = w.add_mesh({"name": "b2c_body", "primitives": [prim]})

    wfr = body.world_from_raw
    rot = np.asarray(wfr["rotation"], np.float32).reshape(1, 3, 3)
    mhr = {"version": 1,
           "model": {"repo": MHR_REPO, "file": MHR_FILE, "sha256": body.model_sha256},
           "sourceRecordVersion": int(body.source_record_version),
           "world_from_raw": {"scale": float(np.asarray(wfr["scale"]).reshape(-1)[0]),
                              "rotation": w.add_accessor(rot, "MAT3"),
                              "translation": [float(v) for v in np.asarray(wfr["translation"]).reshape(3)]},
           "flip": [float(v) for v in body.flip],
           "pose_params": {k: w.add_accessor(np.asarray(v, np.float32).reshape(-1), "SCALAR")
                           for k, v in body.pose_params.items()},
           "model_params": w.add_accessor(np.asarray(body.model_params, np.float32).reshape(-1), "SCALAR"),
           "hand_idx": w.add_accessor(np.asarray(body.hand_idx, np.uint32).reshape(-1), "SCALAR",
                                      component=UNSIGNED_INT)}
    if body.joint_names:
        mhr["jointNames"] = list(body.joint_names)
    body_node = w.add_node({"name": "b2c_body", "mesh": body_mesh, "skin": skin,
                            "extensions": {"B2C_mhr": mhr}})

    orbit: dict = {"version": 1, "migrated": bool(capture.migrated),
                   "final_cameras": _camera_arrays(w, capture.final, names_required=True)}
    if capture.orbit is not None:
        orbit["orbit_cameras"] = _camera_arrays(w, capture.orbit, names_required=False)
    for key in ("helix", "extension", "pass_frames", "anchor_frame_index", "extras", "prompt", "settings"):
        value = getattr(capture, key)
        if value is not None:
            orbit[key] = value
    if capture.images:
        unknown = set(capture.images) - {"reference", "anchor", "front"}
        if unknown:
            raise ValueError(f"B2C_orbit.images takes reference/anchor/front, not {sorted(unknown)}")
        orbit["images"] = {k: w.add_image(v, name=f"b2c_{k}") for k, v in capture.images.items()}
    w.set_extension("B2C_orbit", orbit)

    scene = w.add_scene({"name": "b2c", "nodes": [world, skeleton, body_node]})
    w.set_default_scene(scene)
    return w.save(path)
