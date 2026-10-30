"""The web forms of a rigged subject and its clips (SPEC 10): smaller files for publishing, played by b2cviewer.

    subject(scene.glb, out.sog[, level=i])       # SOG v2 splat + binding textures + rig.glb (runs splat-transform)
    subject_lods(scene.glb, out_dir)             # the LOD chain as a streamed SOG: lod-meta.json + <level>_0/ (10.3)
    clip(name.clip.glb, out.clip.zip, step_mm)   # joint channels + the residual quantised, frame-delta coded, deflated

    python -m b2cgltf.web subject SCENE.glb OUT.sog [--level I]
    python -m b2cgltf.web subject-lods SCENE.glb OUT_DIR
    python -m b2cgltf.web strip-lods SCENE.glb OUT.glb     # b2cgltf.lod.strip_lods: the file without its levels
    python -m b2cgltf.web clip NAME.clip.glb OUT.clip.zip [--step-mm 0.05]

All are lossy and derived: the glTF files stay the source. The subject needs Pillow, SciPy and splat-transform
(`pip install b2cgltf[web]`; splat-transform on PATH, or SPLAT_TRANSFORM naming it); the clip needs numpy only.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from . import glb, read
from .convert import splat_transform, write_ply
from .b2crig import clip as C
from .b2crig import rig as R
from .document import Document

WEB_VERSION = 1
POSING = "b2ctrain cage.cu pose_bound v1"
FACE_LIMIT = 0xFFFF            # bind_face.webp holds a uint16 face index; 0xFFFF = unbound
DEFAULT_STEP_MM = 0.05         # residual step: the posed splats stay within ~1 mm of the glTF clip's on b24be4


# ------------------------------------------------------------------ helpers

def _repack(js: dict, bin_: bytes, keep: list[int]) -> tuple[list[dict], list[dict], bytearray, dict[int, int]]:
    """Accessors `keep` (old indices) with their bytes copied into a fresh buffer 0: (accessors, bufferViews,
    buffer, old -> new accessor index)."""
    accs, views, blob, amap = [], [], bytearray(), {}
    for i in keep:
        if i in amap:
            continue
        a = copy.deepcopy(js["accessors"][i])
        if "bufferView" in a:
            v = js["bufferViews"][a["bufferView"]]
            blob.extend(b"\0" * (-len(blob) % 4))
            nv = {k: v[k] for k in ("byteStride", "target") if k in v}
            nv.update(buffer=0, byteOffset=len(blob), byteLength=v["byteLength"])
            blob.extend(bin_[v.get("byteOffset", 0): v.get("byteOffset", 0) + v["byteLength"]])
            views.append(nv)
            a["bufferView"] = len(views) - 1
        accs.append(a)
        amap[i] = len(accs) - 1
    blob.extend(b"\0" * (-len(blob) % 4))
    return accs, views, blob, amap


def _write_zip(path: Path, entries: list[tuple[str, bytes, bool]]) -> None:
    """entries: (name, data, deflate). Atomic, like glb.write."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data, deflate in entries:
            z.writestr(zipfile.ZipInfo(name), data, zipfile.ZIP_DEFLATED if deflate else zipfile.ZIP_STORED,
                       compresslevel=9 if deflate else None)
    glb.write_bytes(path, buf.getvalue())


def _webp(rgba: np.ndarray, width: int, height: int) -> bytes:
    """Lossless WebP of [n, 4] uint8 texels in the SOG's texture layout (row-major, padded to width x height)."""
    from PIL import Image
    img = np.zeros((width * height, 4), np.uint8)
    img[:len(rgba)] = rgba
    out = io.BytesIO()
    # exact: keep RGB under alpha 0 (every channel is data)
    Image.fromarray(img.reshape(height, width, 4), "RGBA").save(out, "WEBP", lossless=True, quality=100, method=6, exact=True)
    return out.getvalue()


def _texels(data: bytes, n: int) -> tuple[np.ndarray, int, int]:
    from PIL import Image
    im = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))
    return im.reshape(-1, 4)[:n].astype(np.uint32), im.shape[1], im.shape[0]


# ------------------------------------------------------------------ the subject

def rig_glb(doc: Document) -> bytes:
    """The subject stripped to what posing needs: the skeleton (b2c_skeleton and its joints), the skin and the cage
    (b2crig_cage, every layer), with B2CRIG_rig. No splat, body mesh, cameras, images or MHR. The cage and skeleton
    bytes are copied unchanged, so the rig hash (SPEC 6) is the subject's."""
    js, rig = doc.json, doc.extension(R.RIG)
    skin = js["skins"][0]
    nodes, stack = [], [skin["skeleton"]]
    while stack:
        i = stack.pop(0)
        nodes.append(i); stack += js["nodes"][i].get("children", [])
    nodes.append(rig["cageNode"])
    nmap = {old: new for new, old in enumerate(nodes)}
    cage_mesh = copy.deepcopy(js["meshes"][js["nodes"][rig["cageNode"]]["mesh"]])
    keep = [skin["inverseBindMatrices"]]
    for p in cage_mesh["primitives"]:
        keep += list(p["attributes"].values()) + [p["indices"]]
    accs, views, blob, amap = _repack(js, doc.bin, keep)
    for p in cage_mesh["primitives"]:
        p["attributes"] = {k: amap[v] for k, v in p["attributes"].items()}
        p["indices"] = amap[p["indices"]]
    out_nodes = []
    for old in nodes:
        nd = copy.deepcopy(js["nodes"][old])
        nd.pop("extras", None)
        if "children" in nd:
            nd["children"] = [nmap[c] for c in nd["children"] if c in nmap]
        if "mesh" in nd:
            nd["mesh"] = 0
        if "skin" in nd:
            nd["skin"] = 0
        out_nodes.append(nd)
    sk = copy.deepcopy(skin)
    sk.update(joints=[nmap[j] for j in skin["joints"]], skeleton=nmap[skin["skeleton"]], inverseBindMatrices=amap[skin["inverseBindMatrices"]])
    r2 = copy.deepcopy(rig)
    r2.update(cageNode=nmap[rig["cageNode"]], splatNode=None)
    out = {"asset": copy.deepcopy(js["asset"]), "scene": 0,
           "scenes": [{"name": "b2crig", "nodes": [nmap[skin["skeleton"]], nmap[rig["cageNode"]]]}],
           "nodes": out_nodes, "meshes": [cage_mesh], "skins": [sk], "accessors": accs, "bufferViews": views,
           "buffers": [{"byteLength": len(blob)}], "extensionsUsed": [R.RIG], "extensions": {R.RIG: r2}}
    return glb.dumps(out, blob)


def _sog_order(meta: dict, files: dict[str, bytes], pos: np.ndarray) -> np.ndarray:
    """The SOG's splat order (SOG index -> source index), recovered by matching its decoded centres to `pos`:
    splat-transform sorts splats into Morton order and does not report it. Splats closer together than the 16-bit
    quantisation can claim the same source; the nearer keeps it and the other takes its nearest unclaimed one."""
    from scipy.spatial import cKDTree
    n = meta["count"]
    lo, _, _ = _texels(files[meta["means"]["files"][0]], n)
    hi, _, _ = _texels(files[meta["means"]["files"][1]], n)
    mn, mx = np.array(meta["means"]["mins"]), np.array(meta["means"]["maxs"])
    t = mn + ((hi[:, :3] << 8) | lo[:, :3]) / 65535.0 * (mx - mn)
    dec = np.sign(t) * np.expm1(np.abs(t))
    tree = cKDTree(pos)
    _, order = tree.query(dec)
    vals, counts = np.unique(order, return_counts=True)
    free = set(np.setdiff1d(np.arange(n), order).tolist())
    for v in vals[counts > 1]:
        claim = np.nonzero(order == v)[0]
        keep = claim[np.argmin(np.linalg.norm(dec[claim] - pos[v], axis=1))]
        for c in claim[claim != keep]:
            cand = [x for x in tree.query(dec[c], k=min(32, n))[1] if x in free]
            if not cand:
                raise ValueError("could not recover the SOG's splat order")
            order[c] = cand[0]; free.discard(cand[0])
    if len(np.unique(order)) != n:
        raise ValueError("could not recover the SOG's splat order")
    err = np.linalg.norm(dec - pos[order], axis=1).max()
    if err > 1e-3:
        raise ValueError(f"the SOG's centres are up to {err * 1000:.2f} mm from the splat's: not this splat")
    return order


def _st_fields(doc: Document, node: int) -> dict:
    """A splat's trainer-PLY fields as splat-transform gets them: without seg_* (and any ev_*), so every LOD level has
    the same columns (splat-transform: "stackLods: available-layer mismatch" otherwise)."""
    f = read.to_trainer_ply_fields(read.splat(doc, node))
    return {k: v for k, v in f.items() if not k.startswith(("seg_", "ev_"))}


def _run_st(args: list[str], out: Path, exe: str | None) -> None:
    run = subprocess.run([exe or splat_transform(), "-w", *args], capture_output=True, text=True)
    if run.returncode != 0 or not out.exists():
        raise RuntimeError(f"splat-transform failed:\n{run.stdout[-2000:]}\n{run.stderr[-2000:]}")


def _binding(doc: Document, node: int, meta: dict, files: dict[str, bytes], rig_path: str, src) -> tuple[dict, dict, list]:
    """Splat `node`'s binding as textures in its SOG's order (`meta`, `files`: the SOG as written): (the b2c block
    for meta.json, {texture name: bytes}, notes)."""
    n = meta["count"]
    s, b = read.splat(doc, node), R.binding(doc, node)
    if len(s["pos"]) != n:
        raise ValueError(f"node {node}: the SOG holds {n} splats, the splat {len(s['pos'])}")
    notes = []
    if b.alt is not None or any(k.startswith("alt") for k in s["primitive"]["extensions"][R.BIND]):
        notes.append("the dual binding (alt*) is not carried: b2cviewer does not use it")
    nf = len(R.cage(doc).faces)
    if nf >= FACE_LIMIT:
        raise ValueError(f"{src}: the cage has {nf} faces; bind_face.webp holds at most {FACE_LIMIT - 1}")
    _, W, H = _texels(files[meta["means"]["files"][0]], n)
    order = _sog_order(meta, files, s["pos"].astype(np.float64))
    face = b.face[order]
    bary = np.asarray(b.barycentric, np.float64)[order]
    off = np.asarray(b.offset, np.float64)[order]
    f16 = np.where(face < 0, FACE_LIMIT, face).astype(np.uint32)
    b8 = np.round(np.clip(bary, 0, 1) * 255).astype(np.uint32)
    olo, ohi = off.min(0), off.max(0)
    o16 = np.round((off - olo) / np.maximum(ohi - olo, 1e-12) * 65535).astype(np.uint32)
    ones = np.full(n, 255, np.uint32)
    tex = {
        "bind_face.webp": _webp(np.stack([f16 & 255, f16 >> 8, b8[:, 0], b8[:, 1]], 1), W, H),
        "bind_off_l.webp": _webp(np.stack([o16[:, 0] & 255, o16[:, 1] & 255, o16[:, 2] & 255, ones], 1), W, H),
        "bind_off_u.webp": _webp(np.stack([o16[:, 0] >> 8, o16[:, 1] >> 8, o16[:, 2] >> 8, ones], 1), W, H),
    }
    block = {
        "version": WEB_VERSION,
        "rig": rig_path,
        "binding": {"posing": POSING,
                    "face": {"file": "bind_face.webp"},
                    "offset": {"mins": olo.tolist(), "maxs": ohi.tolist(), "files": ["bind_off_l.webp", "bind_off_u.webp"]}},
    }
    return block, tex, notes


def _rigged_chain(doc: Document, src) -> list[tuple[int, int, float | None]]:
    rig = doc.extension(R.RIG)
    if rig is not None and read.current_splat(doc) != rig["splatNode"]:
        raise ValueError(f"{src}: the current splat is not b2crig's rigged splat")
    return read.lods(doc)


def subject(src: str | Path, dst: str | Path, *, level: int = 0, exe: str | None = None) -> dict:
    """SPEC 10.1: the web form of a rigged subject file, one .sog. `level`: which member of the current splat's LOD
    chain (SPEC 4.7, read.lods) is the splat, with its own binding; 0 = the rigged splat itself."""
    doc = Document.load(src)
    if doc.extension(R.RIG) is None:
        raise ValueError(f"{src}: not rigged (no B2CRIG_rig); a plain splat needs no b2c web form, use splat-transform")
    chain = _rigged_chain(doc, src)
    if not 0 <= level < len(chain):
        raise ValueError(f"{src}: no level {level} (the chain has {len(chain)} members)")
    node, _, height = chain[level]
    with tempfile.TemporaryDirectory() as tmp:
        ply, sog = Path(tmp) / "splat.ply", Path(tmp) / "splat.sog"
        write_ply(ply, _st_fields(doc, node))
        _run_st([str(ply), str(sog)], sog, exe)
        with zipfile.ZipFile(sog) as z:
            files = {i.filename: z.read(i) for i in z.infolist()}
    meta = json.loads(files["meta.json"])
    block, tex, notes = _binding(doc, node, meta, files, "rig.glb", src)
    if level:
        block["level"] = {"index": level, "maxScreenHeightPx": height}
    meta["b2c"] = block
    tex["rig.glb"] = rig_glb(doc)
    entries = [("meta.json", json.dumps(meta, separators=(",", ":")).encode(), True)]
    entries += [(k, v, False) for k, v in files.items() if k != "meta.json"]
    entries += [(k, v, k == "rig.glb") for k, v in tex.items()]
    _write_zip(Path(dst), entries)
    return {"splats": meta["count"], "bytes": Path(dst).stat().st_size, "notes": notes}


def _one_file_per_level(lm: dict, counts: list[int]) -> bool:
    """lod-meta.json keeps level i whole in file `i_0/`: the tree's nodes (spatial cells, splat-transform splits once
    the levels together pass chunkGaussians) only index ranges of it, which tile it exactly. splat-transform may list
    the files in any order (`filenames`; e.g. 0_0, 2_0, 1_0): a range's `file` indexes that list."""
    names = lm.get("filenames") or []
    if sorted(names) != sorted(f"{i}_0/meta.json" for i in range(len(counts))):
        return False
    spans = {i: [] for i in range(len(counts))}
    stack = [lm.get("tree", {})]
    while stack:
        nd = stack.pop()
        stack += nd.get("children", [])
        for k, r in nd.get("lods", {}).items():
            if int(k) not in spans or not 0 <= r["file"] < len(names) or names[r["file"]] != f"{k}_0/meta.json":
                return False
            spans[int(k)].append((r["offset"], r["count"]))
    for i, c in enumerate(counts):
        at = 0
        for off, n in sorted(spans[i]):
            if off != at:
                return False
            at += n
        if at != c:
            return False
    return True


def subject_lods(src: str | Path, out_dir: str | Path, *, exe: str | None = None) -> dict:
    """SPEC 10.3: the current splat's LOD chain (read.lods) as one PlayCanvas streamed SOG: out_dir/lod-meta.json and
    a `<level>_0/` SOG folder per level (0 = the base), written by splat-transform from one `-l N` tagged input per
    level. A rigged subject also gets each level's binding textures in that level's SOG order (its meta.json's `b2c`
    block, as in `subject`) and one rig.glb for all. lod-meta.json gets a `b2c` block with each level's count and
    maxScreenHeightPx, which a player picks the level by (lod-meta.json holds no screen heights)."""
    doc = Document.load(src)
    rigged = doc.extension(R.RIG) is not None
    chain = _rigged_chain(doc, src)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not (out / "lod-meta.json").exists():
        raise ValueError(f"{out}: not empty and not a streamed SOG")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        args = []
        for i, (node, _, _) in enumerate(chain):
            ply = Path(tmp) / f"l{i}.ply"
            write_ply(ply, _st_fields(doc, node))
            args += [str(ply), "-l", str(i)]
        stage = Path(tmp) / "out"
        _run_st([*args, str(stage / "lod-meta.json")], stage / "lod-meta.json", exe)
        lm = json.loads((stage / "lod-meta.json").read_text())
        if not _one_file_per_level(lm, [c for _, c, _ in chain]):
            raise NotImplementedError("splat-transform split a level over several files (more than "
                                      f"{lm.get('asset', {}).get('chunkGaussians')} splats in it, or a larger extent): "
                                      "b2c.binding is written for one file per level")
        levels, notes = [], []
        for i, (node, count, height) in enumerate(chain):
            d = stage / f"{i}_0"
            files = {f.name: f.read_bytes() for f in d.iterdir()}
            meta = json.loads(files["meta.json"])
            if rigged:
                block, tex, nt = _binding(doc, node, meta, files, "../rig.glb", src)
                notes += [n for n in nt if n not in notes]
                block["level"] = {"index": i, "maxScreenHeightPx": height}
                meta["b2c"] = block
                for k, v in tex.items():
                    (d / k).write_bytes(v)
                (d / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
            levels.append({"count": count, "maxScreenHeightPx": height})
        lm["b2c"] = {"version": WEB_VERSION, "levels": levels, **({"rig": "rig.glb"} if rigged else {})}
        (stage / "lod-meta.json").write_text(json.dumps(lm, separators=(",", ":")))
        if rigged:
            (stage / "rig.glb").write_bytes(rig_glb(doc))
        # replace a previous output only once this one is complete
        for old in [*out.glob("*_*/"), out / "lod-meta.json", out / "rig.glb"]:
            if old.is_dir():
                shutil.rmtree(old)
            elif old.exists():
                old.unlink()
        for f in stage.iterdir():
            shutil.move(str(f), out / f.name)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    return {"levels": levels, "bytes": size, "rigged": rigged, "notes": notes}


def read_subject(path: str | Path) -> dict:
    """A web subject read back (for checks): the SOG centres, the binding in SOG order and the rig document. `path`:
    a .sog, or one level folder (`<level>_0/`) of a streamed SOG (subject_lods)."""
    path = Path(path)
    if path.is_dir():
        files = {f.name: f.read_bytes() for f in path.iterdir() if f.is_file()}
    else:
        with zipfile.ZipFile(path) as z:
            files = {i.filename: z.read(i) for i in z.infolist()}
    meta = json.loads(files["meta.json"])
    n, B = meta["count"], meta["b2c"]
    if B["rig"] not in files:   # a streamed SOG's level: the shared rig.glb beside the level folders
        files[B["rig"]] = (path / B["rig"]).read_bytes()
    lo, _, _ = _texels(files[meta["means"]["files"][0]], n)
    hi, _, _ = _texels(files[meta["means"]["files"][1]], n)
    mn, mx = np.array(meta["means"]["mins"]), np.array(meta["means"]["maxs"])
    t = mn + ((hi[:, :3] << 8) | lo[:, :3]) / 65535.0 * (mx - mn)
    bf, _, _ = _texels(files[B["binding"]["face"]["file"]], n)
    bl, _, _ = _texels(files[B["binding"]["offset"]["files"][0]], n)
    bu, _, _ = _texels(files[B["binding"]["offset"]["files"][1]], n)
    face = (bf[:, 0] | (bf[:, 1] << 8)).astype(np.int64)
    face[face == FACE_LIMIT] = -1
    olo, ohi = np.array(B["binding"]["offset"]["mins"]), np.array(B["binding"]["offset"]["maxs"])
    js, bin_ = _glb_loads(files[B["rig"]])
    return {"meta": meta, "pos": np.sign(t) * np.expm1(np.abs(t)), "face": face, "barycentric": bf[:, 2:4] / 255.0,
            "offset": olo + ((bu[:, :3] << 8) | bl[:, :3]) / 65535.0 * (ohi - olo), "rig": Document(js, bin_)}


def _glb_loads(data: bytes) -> tuple[dict, bytearray]:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.glb"
        p.write_bytes(data)
        return glb.read(p)


# ------------------------------------------------------------------ clips

def clip(src: str | Path, dst: str | Path, step_mm: float = DEFAULT_STEP_MM) -> dict:
    """SPEC 10.2: the web form of a clip file. The MHR inputs (B2CRIG_motion) and the open gate are left out."""
    js, bin_ = glb.read(src)
    if C.CLIP not in js.get("extensions", {}):
        raise ValueError(f"{src}: not a clip file (no {C.CLIP})")
    anim = js["animations"][0]
    ext = anim.get("extensions", {})
    keep = [i for s in anim["samplers"] for i in (s["input"], s["output"])]
    accs, views, blob, amap = _repack(js, bin_, keep)
    out = copy.deepcopy(js)
    out.update(accessors=accs, bufferViews=views, buffers=[{"byteLength": len(blob)}])
    an = out["animations"][0]
    for s in an["samplers"]:
        s["input"], s["output"] = amap[s["input"]], amap[s["output"]]
    an.pop("extensions", None)
    entries, err = [], 0.0
    if C.RES in ext:
        rx = ext[C.RES]
        a, v = js["accessors"][rx["accessor"]], js["bufferViews"][js["accessors"][rx["accessor"]]["bufferView"]]
        T, NV = rx["frames"], rx["vertices"]
        r = np.frombuffer(bytes(bin_), "<i2", count=T * NV * 3, offset=v.get("byteOffset", 0) + a.get("byteOffset", 0))
        res = np.maximum(r.astype(np.float64), -32767) * rx["scale"] / 32767   # normalised SHORT
        q = np.round(res / (step_mm / 1000)).astype(np.int64)
        d = np.diff(q.reshape(T, NV * 3), axis=0, prepend=0)
        if np.abs(d).max() > 32767:
            raise ValueError(f"a frame-to-frame residual change exceeds int16 at {step_mm} mm steps: use a larger step")
        planes = d.astype("<i2").view(np.uint8).reshape(-1, 2)
        entries.append(("residual.bin", np.concatenate([planes[:, 0], planes[:, 1]]).tobytes(), True))
        err = float(np.abs(q * (step_mm / 1000) - res).max())
        an["extensions"] = {C.RES: {"version": rx["version"], "file": "residual.bin", "frames": T, "vertices": NV,
                                    "step": step_mm / 1000}}
    used = {C.CLIP} | ({C.RES} if C.RES in ext else set())
    out["extensionsUsed"] = [e for e in js.get("extensionsUsed", []) if e in used or e not in (C.MOTION, C.GATE)]
    _write_zip(Path(dst), [("clip.glb", glb.dumps(out, blob), True)] + entries)
    return {"bytes": Path(dst).stat().st_size, "max_residual_error_m": err,
            "dropped": sorted(k for k in ext if k in (C.MOTION, C.GATE))}


def read_clip_residual(path: str | Path) -> np.ndarray:
    """A web clip's residual [T, nv, 3] in metres (for checks)."""
    with zipfile.ZipFile(path) as z:
        js, _ = _glb_loads(z.read("clip.glb"))
        rx = js["animations"][0]["extensions"][C.RES]
        raw = np.frombuffer(z.read(rx["file"]), np.uint8)
    h = len(raw) // 2
    d = (raw[:h].astype(np.uint16) | (raw[h:].astype(np.uint16) << 8)).view(np.int16)
    return np.cumsum(d.reshape(rx["frames"], -1).astype(np.int64), 0).reshape(rx["frames"], rx["vertices"], 3) * rx["step"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("subject", help="a rigged subject .glb -> its web form (.sog)")
    a.add_argument("src", type=Path); a.add_argument("dst", type=Path)
    a.add_argument("--level", type=int, default=0, help="LOD level of the current splat (0 = the rigged splat)")
    sl = sub.add_parser("subject-lods", help="a subject .glb -> its LOD chain as a streamed SOG (lod-meta.json)")
    sl.add_argument("src", type=Path); sl.add_argument("dst", type=Path)
    st = sub.add_parser("strip-lods", help="a subject .glb -> a copy without its LOD levels")
    st.add_argument("src", type=Path); st.add_argument("dst", type=Path)
    c = sub.add_parser("clip", help="a .clip.glb -> its web form (.clip.zip)")
    c.add_argument("src", type=Path); c.add_argument("dst", type=Path)
    c.add_argument("--step-mm", type=float, default=DEFAULT_STEP_MM, help=f"residual quantisation step (default {DEFAULT_STEP_MM})")
    args = ap.parse_args(argv)
    if args.cmd == "subject":
        info = subject(args.src, args.dst, level=args.level)
        print(f"{args.dst}: {info['splats']} splats, {info['bytes'] / 1e6:.2f} MB")
    elif args.cmd == "subject-lods":
        info = subject_lods(args.src, args.dst)
        print(f"{args.dst}: {len(info['levels'])} levels "
              + ", ".join(f"{lv['count']}" + (f" <= {lv['maxScreenHeightPx']:g} px" if lv["maxScreenHeightPx"] else "")
                          for lv in info["levels"]) + f", {info['bytes'] / 1e6:.2f} MB")
    elif args.cmd == "strip-lods":
        from .lod import strip_lods
        info = strip_lods(args.src, args.dst)
        print(f"{args.dst}: removed {', '.join(info['levels'])}; {info['bytes_before'] / 1e6:.1f} -> "
              f"{info['bytes_after'] / 1e6:.1f} MB of data")
    else:
        info = clip(args.src, args.dst, args.step_mm)
        print(f"{args.dst}: {info['bytes'] / 1e6:.2f} MB, residual within {info['max_residual_error_m'] * 1000:.3f} mm"
              + (f"; left out {', '.join(info['dropped'])}" if info["dropped"] else ""))
    for note in info.get("notes", []):
        print(f"note: {note}")


if __name__ == "__main__":
    main()
