"""Splat LOD levels (SPEC 4.7): a chain of lower-detail splats behind a base splat node.

    lvl = lod.add_level_splat(w, fields, "b2c_splat_lod1")         # b2crunner: an unskinned level (4.1)
    lod.add_lods(w, base, [{"node": lvl, "method": "..."}, ...])    # MSFT_lod + B2C_lod / B2CRIG_lod on the base
    lod.chain(doc)                                                   # [(node, count, maxScreenHeightPx)], base first

b2crig's skinned levels are made by `b2crig.rig.add_level`; the chain is written the same way. The base node may be
one this writer added or one its owner wrote earlier (Writer.extend_node, R2b); the level nodes are in no scene.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import splat as S
from .document import Document, RuleError, Writer, owner_of_name

MSFT = "MSFT_lod"
RECORD = {"b2crunner": "B2C_lod", "b2crig": "B2CRIG_lod"}
DEFAULT_HEIGHTS = (960, 480)        # px, the levels judged on b2crunner's crew-* and b2crig's b24be4 (2026-10-30)
B2CRIG_STATE = ("_B2CRIG_OPEN_R", "_B2CRIG_OPEN_G", "_B2CRIG_OPEN_B", "_B2CRIG_OPEN_DOPACITY", "_B2CRIG_CAGE_FILL",
                "_B2CRIG_GATE_A", "_B2CRIG_GATE_B", "_B2CRIG_SEG_LABEL", "_B2CRIG_SEG_CONF")


def _prim(doc: Document, node: int) -> dict:
    js = doc.json
    return js["meshes"][js["nodes"][node]["mesh"]]["primitives"][0]


def _degree(at: dict) -> int:
    d = 0
    while f"{S.KHR}:SH_DEGREE_{d + 1}_COEF_0" in at:
        d += 1
    return d


def source_hash(doc: Document, node: int) -> str:
    """SPEC 4.7 `source`: one SHA-256 over the splat's POSITION, ROTATION, SCALE, OPACITY and SH_DEGREE_d_COEF_n
    (d = 0..degree, n = 0..2d) accessors, in that order, as the stored display values packed as little-endian float32
    without stride padding. "sha256:" + lowercase hex."""
    at = _prim(doc, node)["attributes"]
    keys = ["POSITION", f"{S.KHR}:ROTATION", f"{S.KHR}:SCALE", f"{S.KHR}:OPACITY"]
    keys += [f"{S.KHR}:SH_DEGREE_{d}_COEF_{n}" for d in range(_degree(at) + 1) for n in range(2 * d + 1)]
    h = hashlib.sha256()
    for k in keys:
        h.update(np.ascontiguousarray(doc.accessor(at[k]), "<f4").tobytes())
    return "sha256:" + h.hexdigest()


def _count(doc: Document, node: int) -> int:
    return doc.json["accessors"][_prim(doc, node)["attributes"]["POSITION"]]["count"]


def _in_scenes(doc: Document) -> set[int]:
    js, out = doc.json, set()
    for sc in js.get("scenes", []):
        stack = list(sc.get("nodes", []))
        while stack:
            n = stack.pop()
            if n not in out:
                out.add(n); stack += js["nodes"][n].get("children", [])
    return out


def screen_coverage(doc: Document, base: int, heights: list[float], viewport=(1920, 1080)) -> list[float]:
    """MSFT_screencoverage derived from the levels' maxScreenHeightPx (4.7): the share of a `viewport` [w, h] that the
    base's front-on bounding box covers when the subject is a level's height tall. One value per chain member: the
    base's is the first level's threshold, each level's the next one's, the last level's 0."""
    lo, hi = (np.asarray(v, np.float64) for v in (doc.json["accessors"][_prim(doc, base)["attributes"]["POSITION"]][k]
                                                  for k in ("min", "max")))
    ext = hi - lo
    aspect = max(ext[0], ext[2]) / max(ext[1], 1e-9)
    cov = [float((h / viewport[1]) ** 2 * aspect * viewport[1] / viewport[0]) for h in heights]
    return cov + [0.0]


def add_level_splat(w: Writer, fields, name: str) -> int:
    """b2crunner's level node (4.1): the 4.2 encoding of trainer-PLY `fields`, no transform, no skin, in no scene."""
    mesh = w.add_mesh({"name": name, "primitives": [S.primitive(S.attributes(w, fields))]})
    return w.add_node({"name": name, "mesh": mesh})


def add_lods(w: Writer, base: int, levels: list[dict], *, coverage: bool = False, viewport=(1920, 1080)) -> dict:
    """Chain `levels` behind the splat node `base` (SPEC 4.7): MSFT_lod {ids} on the base, and the owner's level
    record (B2C_lod / B2CRIG_lod). Each level is {"node", "method", "maxScreenHeightPx" (default 960, 480, then
    halving)}; `count` and `source` are filled in here. `coverage`: also write the derived MSFT_screencoverage.
    The chain is checked (SPEC 8) before anything is written. Returns the level record."""
    doc = w.doc
    if owner_of_name(doc.json["nodes"][base].get("name")) != w.owner:
        raise RuleError(f"{w.owner} chains levels only behind its own splat (R1)")
    src = source_hash(doc, base)
    recs = []
    for i, lv in enumerate(levels):
        h = lv.get("maxScreenHeightPx", DEFAULT_HEIGHTS[i] if i < len(DEFAULT_HEIGHTS) else DEFAULT_HEIGHTS[-1] / 2 ** (i - 1))
        recs.append({"node": int(lv["node"]), "count": _count(doc, lv["node"]), "maxScreenHeightPx": float(h),
                     "method": str(lv.get("method", "")), "source": src})
    rec = {"version": 1, "levels": recs}
    problems = _check(doc, base, [r["node"] for r in recs], rec)
    if problems:
        raise RuleError("LOD chain: " + "; ".join(problems))
    extras = {"MSFT_screencoverage": screen_coverage(doc, base, [r["maxScreenHeightPx"] for r in recs], viewport)} if coverage else None
    w.extend_node(base, extensions={MSFT: {"ids": [r["node"] for r in recs]}, RECORD[w.owner]: rec}, extras=extras)
    return rec


def _record(doc: Document, base: int) -> dict | None:
    ex = doc.json["nodes"][base].get("extensions") or {}
    return ex.get(RECORD["b2crig"]) or ex.get(RECORD["b2crunner"])


def _check(doc: Document, base: int, ids: list[int], rec: dict | None) -> list[str]:
    js, out = doc.json, []
    scened = _in_scenes(doc)
    bat = _prim(doc, base)["attributes"]
    bext = _prim(doc, base).get("extensions", {}).get(S.KHR, {})
    prev = _count(doc, base)
    for n in ids:
        nd = js["nodes"][n]
        if n in scened:
            out.append(f"level node {n} is listed in a scene")
        prim = _prim(doc, n) if "mesh" in nd else None
        if prim is None or S.KHR not in (prim.get("extensions") or {}):
            out.append(f"level node {n} is not a splat"); continue
        at = prim["attributes"]
        if any(k in nd for k in ("matrix", "translation", "rotation", "scale")):
            out.append(f"level node {n} has a transform")
        if "skin" in nd and owner_of_name(nd.get("name")) == "b2crunner":
            out.append(f"level node {n} is b2crunner's and skinned")
        if _degree(at) != _degree(bat):
            out.append(f"level node {n}: SH degree {_degree(at)}, base {_degree(bat)}")
        lext = prim["extensions"][S.KHR]
        for k in ("kernel", "colorSpace"):
            if lext.get(k) != bext.get(k):
                out.append(f"level node {n}: {k} {lext.get(k)!r}, base {bext.get(k)!r}")
        for k in ("_SEG_LABEL", "_SEG_CONF") + B2CRIG_STATE:
            if k in bat and k not in at:
                out.append(f"level node {n} lacks {k}, which the base has")
        c = _count(doc, n)
        if c >= prev:
            out.append(f"level node {n}: {c} splats, not fewer than the {prev} before it")
        prev = c
    if rec is not None:
        if [r["node"] for r in rec.get("levels", [])] != list(ids):
            out.append("the level record's nodes do not match MSFT_lod.ids")
        hs = [r.get("maxScreenHeightPx") for r in rec.get("levels", [])]
        if any(h is None or h <= 0 for h in hs) or any(b >= a for a, b in zip(hs, hs[1:])):
            out.append(f"maxScreenHeightPx {hs} does not strictly decrease")
        for r in rec.get("levels", []):
            if r.get("node") in ids and r.get("count") != _count(doc, r["node"]):
                out.append(f"level node {r['node']}: record count {r.get('count')}, splat {_count(doc, r['node'])}")
    cov = (js["nodes"][base].get("extras") or {}).get("MSFT_screencoverage")
    if cov is not None and (len(cov) != len(ids) + 1 or any(b > a for a, b in zip(cov, cov[1:]))):
        out.append(f"MSFT_screencoverage {cov} is not one non-increasing value per chain member")
    return out


def check(doc: Document, base: int | None = None) -> list[str]:
    """SPEC 8's chain checks for the splat node `base` (default: the current splat); [] when it has no chain or a
    sound one. A stale `source` is reported too (a warning: read.lods then drops the chain)."""
    from .read import current_splat
    base = current_splat(doc) if base is None else base
    ext = doc.json["nodes"][base].get("extensions") or {}
    if MSFT not in ext:
        return []
    rec = _record(doc, base)
    if rec is None:
        return [f"node {base} has MSFT_lod but no B2C_lod / B2CRIG_lod record"]
    out = _check(doc, base, list(ext[MSFT].get("ids", [])), rec)
    src = source_hash(doc, base)
    if any(r.get("source") != src for r in rec.get("levels", [])):
        out.append(f"stale chain: a level's source is not the base's {src}")
    return out


def chain(doc: Document, base: int | None = None, *, log=print) -> list[tuple[int, int, float | None]]:
    """SPEC 7.1: the current splat's LOD chain, [(node, count, maxScreenHeightPx)], most detailed first; the base's
    height is None (no upper bound). The base alone when it has no chain, or when the chain is stale or broken."""
    from .read import current_splat
    base = current_splat(doc) if base is None else base
    out = [(base, _count(doc, base), None)]
    ext = doc.json["nodes"][base].get("extensions") or {}
    if MSFT not in ext:
        return out
    problems = check(doc, base)
    if problems:
        log(f"b2cgltf.lod: node {base}'s LOD chain is not used: " + "; ".join(problems))
        return out
    return out + [(r["node"], r["count"], r["maxScreenHeightPx"]) for r in _record(doc, base)["levels"]]
