"""A subject or clip file in memory, and the only write path into it (SPEC 3, 8).

    doc = Document.load(path)                       # anyone may read
    w = doc.writer("b2crig", tool="b2crig export", version="…", commit="…")
    a = w.add_accessor(array, "VEC3")               # appends only
    n = w.add_node({"name": "b2crig_cage", "mesh": m, "skin": 0})
    w.save(path)                                     # checks R1–R8, appends asset.extras.b2c_history

Every `add_*` appends; nothing edits an existing index. `save` compares every object that existed when the writer
was opened with its snapshot and refuses the file if one changed (R1, R2), so a bug in a writer fails loudly
instead of corrupting the other tool's data.
"""
from __future__ import annotations

import copy
import datetime as _dt
import uuid
from pathlib import Path

import numpy as np

from . import glb

# glTF component types
BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, UNSIGNED_INT, FLOAT = 5120, 5121, 5122, 5123, 5125, 5126
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
DTYPE = {BYTE: np.int8, UNSIGNED_BYTE: np.uint8, SHORT: np.int16, UNSIGNED_SHORT: np.uint16, UNSIGNED_INT: np.uint32,
         FLOAT: np.float32}
CTYPE = {np.dtype(v): k for k, v in DTYPE.items()}

OWNERS = {
    "b2crunner": {"name": ("b2c_", "b2c"), "ext": "B2C_", "attr": ("_B2C_", "_SEG_")},
    "b2crig": {"name": ("b2crig_", "b2crig"), "ext": "B2CRIG_", "attr": ("_B2CRIG_",)},
}
# Extension versions this package reads and writes (R6). A reader refuses others.
VERSIONS = {"B2C_mhr": 1, "B2C_orbit": 1, "B2CRIG_rig": 1, "B2CRIG_splat_cage": 1, "B2CRIG_cage_app": 1,
            "B2CRIG_clip": 1, "B2CRIG_cage_residual": 1, "B2CRIG_open_gate": 1, "B2CRIG_motion": 1}
ARRAYS = ("accessors", "animations", "bufferViews", "cameras", "images", "materials", "meshes", "nodes", "samplers",
          "scenes", "skins", "textures")


class RuleError(Exception):
    """A write that SPEC.md's rules forbid."""


def owner_of_name(name: str | None) -> str | None:
    if not name:
        return None
    if name == "b2crig" or name.startswith("b2crig_"):
        return "b2crig"
    if name == "b2c" or name.startswith("b2c_"):
        return "b2crunner"
    return None


def owner_of_ext(key: str) -> str | None:
    if key.startswith("B2CRIG_"):
        return "b2crig"
    if key.startswith("B2C_"):
        return "b2crunner"
    return None


def owner_of_attr(key: str) -> str | None:
    if key.startswith("_B2CRIG_"):
        return "b2crig"
    if key.startswith(("_B2C_", "_SEG_")):
        return "b2crunner"
    return None


class Document:
    def __init__(self, js: dict, bin_: bytearray, path: Path | None = None):
        self.json, self.bin, self.path = js, bin_, path

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path) -> "Document":
        js, b = glb.read(path)
        doc = cls(js, b, Path(path))
        doc.check_versions()
        return doc

    def check_versions(self) -> None:
        """R6: refuse an extension version this package does not know."""
        def visit(obj):
            if isinstance(obj, dict):
                for k, v in (obj.get("extensions") or {}).items():
                    if k in VERSIONS and isinstance(v, dict) and v.get("version") != VERSIONS[k]:
                        raise RuleError(f"{k} version {v.get('version')} is not supported (this package reads {VERSIONS[k]})")
                for v in obj.values():
                    visit(v)
            elif isinstance(obj, list):
                for v in obj:
                    visit(v)
        visit(self.json)

    @property
    def id(self) -> str | None:
        return self.json.get("asset", {}).get("extras", {}).get("b2c_id")

    @property
    def stripped(self) -> bool:
        return bool(self.json.get("asset", {}).get("extras", {}).get("b2c_stripped"))

    def extension(self, key: str, obj: dict | None = None):
        return ((self.json if obj is None else obj).get("extensions") or {}).get(key)

    # ------------------------------------------------------------------ reading
    def accessor(self, i: int, *, normalized: bool = False) -> np.ndarray:
        """Accessor `i` as a numpy array [count, ncomp] (or [count] for SCALAR, [count, 3, 3] / [count, 4, 4] for
        MAT3 / MAT4 in row-major order). `normalized=True` maps normalized integer accessors to floats."""
        a = self.json["accessors"][i]
        n, nc, dt = a["count"], NCOMP[a["type"]], np.dtype(DTYPE[a["componentType"]])
        if "bufferView" not in a:
            x = np.zeros(n * nc, dt)
        else:
            bv = self.json["bufferViews"][a["bufferView"]]
            if bv.get("buffer", 0) != 0:
                raise ValueError("only buffer 0 (the GLB BIN chunk) is supported (R3)")
            off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
            stride = bv.get("byteStride")
            if a["type"] == "MAT3" and dt.itemsize < 4:
                raise ValueError("MAT3 with 1- or 2-byte components needs column padding; not used here")
            if stride and stride != nc * dt.itemsize:
                el = nc * dt.itemsize
                raw = np.frombuffer(self.bin, np.uint8, (n - 1) * stride + el if n else 0, off)
                x = np.lib.stride_tricks.as_strided(raw, (n, el), (stride, 1)).copy().view(dt).reshape(-1)
            else:
                x = np.frombuffer(self.bin, dt, n * nc, off).copy()   # a copy: views would pin the growing buffer
        if normalized and a.get("normalized"):
            x = np.maximum(x.astype(np.float64) / np.iinfo(dt).max, -1.0)
        if a["type"] in ("MAT3", "MAT4"):
            k = 3 if a["type"] == "MAT3" else 4
            return np.swapaxes(x.reshape(n, k, k), 1, 2)   # glTF matrices are column-major
        return x.reshape(n, nc) if nc > 1 else x.reshape(n)

    def node_index(self, name: str) -> int:
        for i, nd in enumerate(self.json.get("nodes", [])):
            if nd.get("name") == name:
                return i
        raise KeyError(name)

    # ------------------------------------------------------------------ writing
    def writer(self, owner: str, *, tool: str, version: str = "", commit: str = "") -> "Writer":
        return Writer(self, owner, tool=tool, version=version, commit=commit)

    @classmethod
    def new(cls, owner: str, *, tool: str, version: str = "", commit: str = "", generator: str = "b2cgltf",
            subject: bool = True) -> tuple["Document", "Writer"]:
        """An empty document and its first writer. A subject file (`subject=True`) is created only by b2crunner and
        gets its `asset.extras.b2c_id` here (R5); clip files are created by b2crig."""
        if subject and owner != "b2crunner":
            raise RuleError("only b2crunner creates subject files (SPEC 1)")
        js = {"asset": {"version": "2.0", "generator": generator, "extras": {"b2c_history": []}}}
        if subject:
            js["asset"]["extras"]["b2c_id"] = str(uuid.uuid4())
        doc = cls(js, bytearray())
        w = Writer(doc, owner, tool=tool, version=version, commit=commit, _new=True)
        return doc, w


class Writer:
    """Appends to a Document on behalf of one owner and enforces R1–R8 when saving."""

    def __init__(self, doc: Document, owner: str, *, tool: str, version: str, commit: str, _new: bool = False):
        if owner not in OWNERS:
            raise RuleError(f"unknown owner {owner!r}")
        if doc.stripped:
            raise RuleError("a stripped (distribution) file is never enhanced (R8)")
        self.doc, self.owner, self.new = doc, owner, _new
        self.entry = {"tool": tool, "version": version, "commit": commit, "added": []}
        self.snap = copy.deepcopy(doc.json)
        self.joints: set[int] = set()
        self.bin0 = len(doc.bin)
        self.bin_prefix = bytes(doc.bin)

    # ------------------------------------------------------------------ checks while appending
    def _name_ok(self, obj: dict, what: str, *, allow_plain: bool = False) -> None:
        o = owner_of_name(obj.get("name"))
        if o is None and allow_plain:
            return
        if o != self.owner:
            raise RuleError(f"{what} name {obj.get('name')!r} does not carry {self.owner}'s prefix (R1)")

    def _ext_ok(self, key: str) -> None:
        o = owner_of_ext(key)
        if o is not None and o != self.owner:
            raise RuleError(f"extension {key} belongs to {o} (R1)")
        if key in VERSIONS:
            pass

    def _exts_ok(self, obj: dict) -> None:
        for k, v in (obj.get("extensions") or {}).items():
            self._ext_ok(k)
            if k in VERSIONS and (not isinstance(v, dict) or v.get("version") != VERSIONS[k]):
                raise RuleError(f"{k} needs \"version\": {VERSIONS[k]} (R6)")
            self.use_extension(k)

    def _append(self, key: str, obj: dict) -> int:
        arr = self.doc.json.setdefault(key, [])
        arr.append(obj)
        self.entry["added"].append(f"{key}/{len(arr) - 1}")
        return len(arr) - 1

    def use_extension(self, key: str) -> None:
        used = self.doc.json.setdefault("extensionsUsed", [])
        if key not in used:
            used.append(key)

    # ------------------------------------------------------------------ binary data
    def add_bytes(self, data: bytes, *, target: int | None = None, align: int = 4) -> int:
        """A new bufferView over bytes appended to buffer 0 (R3)."""
        b = self.doc.bin
        b.extend(b"\0" * (-len(b) % align))
        off = len(b)
        b.extend(data)
        bv = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target is not None:
            bv["target"] = target
        return self._append("bufferViews", bv)

    def add_accessor(self, array: np.ndarray, type_: str, *, component: int | None = None, normalized: bool = False,
                     target: int | None = None, minmax: bool = False) -> int:
        """`array` [count, ncomp] (MAT3/MAT4: [count, k, k] row-major; stored column-major). Component type from
        the dtype unless given."""
        a = np.asarray(array)
        if type_ in ("MAT3", "MAT4"):
            a = np.swapaxes(a, -1, -2)
        ct = component if component is not None else CTYPE.get(a.dtype)
        if ct is None:
            raise ValueError(f"no glTF component type for {a.dtype}")
        a = np.ascontiguousarray(a, DTYPE[ct])
        nc = NCOMP[type_]
        if a.size % nc:
            raise ValueError(f"{a.shape} is not a whole number of {type_}")
        if normalized and ct in (FLOAT, UNSIGNED_INT):
            raise ValueError("normalized FLOAT / UNSIGNED_INT accessors are not allowed")
        if type_ == "MAT3" and a.dtype.itemsize < 4:
            raise ValueError("1- and 2-byte MAT3 accessors need column padding; use float")
        el = nc * a.dtype.itemsize
        if target == ARRAY_BUFFER and el % 4:
            # glTF: every vertex-attribute element starts on a 4-byte boundary, so pad elements to a byte stride
            stride = el + (-el % 4)
            rows = np.zeros((a.size // nc, stride), np.uint8)
            rows[:, :el] = np.frombuffer(a.tobytes(), np.uint8).reshape(-1, el)
            bv = self.add_bytes(rows.tobytes(), target=target)
            self.doc.json["bufferViews"][bv]["byteStride"] = stride
        else:
            bv = self.add_bytes(a.tobytes(), target=target, align=max(4, a.dtype.itemsize))
        acc = {"bufferView": bv, "componentType": ct, "count": a.size // nc, "type": type_}
        if normalized:
            acc["normalized"] = True
        if minmax:
            flat = a.reshape(-1, nc)
            conv = (lambda x: float(x)) if ct == FLOAT else (lambda x: int(x))
            acc["min"] = [conv(v) for v in flat.min(0)]
            acc["max"] = [conv(v) for v in flat.max(0)]
        return self._append("accessors", acc)

    def add_image(self, data: bytes, mime: str = "image/png", *, name: str) -> int:
        img = {"name": name, "bufferView": self.add_bytes(data), "mimeType": mime}
        self._name_ok(img, "image")
        return self._append("images", img)

    # ------------------------------------------------------------------ objects
    def add_node(self, node: dict, *, joint: bool = False) -> int:
        """`joint=True`: a skeleton joint. Joints are named by MHR (or b2c_joint_NNN, SPEC 4.1) and a clip file's
        copy keeps the subject's names (SPEC 6), so their names are exempt from the prefix rule."""
        if not joint:
            self._name_ok(node, "node")
        self._exts_ok(node)
        i = self._append("nodes", node)
        if joint:
            self.joints.add(i)
        return i

    def add_mesh(self, mesh: dict) -> int:
        self._name_ok(mesh, "mesh")
        self._exts_ok(mesh)
        for p in mesh.get("primitives", []):
            self._exts_ok(p)
            for k in p.get("attributes", {}):
                o = owner_of_attr(k)
                if k.startswith("_") and o != self.owner:
                    raise RuleError(f"attribute {k} does not carry {self.owner}'s prefix (R1)")
        return self._append("meshes", mesh)

    def add_skin(self, skin: dict) -> int:
        self._name_ok(skin, "skin")
        return self._append("skins", skin)

    def add_scene(self, scene: dict) -> int:
        self._name_ok(scene, "scene")
        return self._append("scenes", scene)

    def add_animation(self, anim: dict) -> int:
        self._name_ok(anim, "animation")
        self._exts_ok(anim)
        return self._append("animations", anim)

    def add_camera(self, cam: dict) -> int:
        self._name_ok(cam, "camera")
        return self._append("cameras", cam)

    def set_extension(self, key: str, value: dict) -> None:
        """A document-level extension (e.g. B2C_orbit, B2CRIG_rig). An owner may replace its own."""
        self._ext_ok(key)
        if key in VERSIONS and value.get("version") != VERSIONS[key]:
            raise RuleError(f"{key} needs \"version\": {VERSIONS[key]} (R6)")
        self.doc.json.setdefault("extensions", {})[key] = value
        self.use_extension(key)
        self.entry["added"].append(f"extensions/{key}")

    def set_default_scene(self, i: int) -> None:
        """R1b: the one shared field."""
        self.doc.json["scene"] = i

    def set_asset_extra(self, key: str, value) -> None:
        if not key.startswith("b2c_") or key in ("b2c_id", "b2c_history", "b2c_stripped"):
            raise RuleError(f"asset.extras.{key} is not a free field")
        self.doc.json["asset"].setdefault("extras", {})[key] = value

    # ------------------------------------------------------------------ save
    def check(self) -> None:
        js, snap = self.doc.json, self.snap
        # R2: every object that existed is unchanged (buffer 0's length is the writer's to update)
        for key in ARRAYS:
            old, new = snap.get(key, []), js.get(key, [])
            if len(new) < len(old):
                raise RuleError(f"{key}: objects were removed (R2)")
            for i, o in enumerate(old):
                if new[i] != o:
                    raise RuleError(f"{key}/{i} was changed; writers only append (R1, R2)")
        if self.doc.bin[:self.bin0] != self.bin_prefix:
            raise RuleError("buffer 0's existing bytes changed (R3)")
        # document-level extensions: the other owner's are untouched
        for k, v in (snap.get("extensions") or {}).items():
            if owner_of_ext(k) != self.owner and (js.get("extensions") or {}).get(k) != v:
                raise RuleError(f"extensions/{k} belongs to {owner_of_ext(k) or 'nobody'} and was changed (R1)")
        # asset: only extras.b2c_history grows (this writer's entry is added on save), b2c_id never changes
        a_old, a_new = snap.get("asset", {}), js.get("asset", {})
        e_old, e_new = a_old.get("extras", {}), a_new.get("extras", {})
        if e_old.get("b2c_id") != e_new.get("b2c_id"):
            raise RuleError("asset.extras.b2c_id never changes (R5)")
        if e_new.get("b2c_history", [])[:len(e_old.get("b2c_history", []))] != e_old.get("b2c_history", []):
            raise RuleError("asset.extras.b2c_history is append-only (R5)")
        # R4
        req = [k for k in js.get("extensionsRequired", []) if owner_of_ext(k)]
        if req:
            raise RuleError(f"{req} must not be required (R4)")
        # ownership of everything new
        for key in ("nodes", "meshes", "skins", "scenes", "animations", "images", "cameras"):
            for i in range(len(snap.get(key, [])), len(js.get(key, []))):
                if key == "nodes" and i in self.joints:
                    continue
                nm = js[key][i].get("name")
                o = owner_of_name(nm)
                if o is not None and o != self.owner:
                    raise RuleError(f"{key}/{i} {nm!r} added by {self.owner} carries another owner's prefix (R1)")

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path or self.doc.path)
        js = self.doc.json
        js.setdefault("buffers", [{}]) if self.doc.bin else None
        if self.doc.bin:
            if len(js["buffers"]) != 1 or "uri" in js["buffers"][0]:
                raise RuleError("one buffer, embedded (R3)")
            js["buffers"][0]["byteLength"] = len(self.doc.bin)
            self.snap.setdefault("buffers", js["buffers"])
        self.check()
        entry = dict(self.entry, time=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
        js["asset"].setdefault("extras", {}).setdefault("b2c_history", []).append(entry)
        glb.write(path, js, self.doc.bin)
        self.doc.path = path
        # the next save of this writer starts from here
        self.snap, self.bin0, self.bin_prefix = copy.deepcopy(js), len(self.doc.bin), bytes(self.doc.bin)
        self.entry = dict(self.entry, added=[])
        return path


def load(path: str | Path) -> Document:
    return Document.load(path)
