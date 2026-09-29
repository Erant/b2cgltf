"""The GLB container: a JSON chunk and one BIN chunk (glTF 2.0, section 4.4). The JSON stays a plain dict, so a
load -> save round trip reproduces it exactly (SPEC 8, preservation)."""
from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path

MAGIC, VERSION, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 2, 0x4E4F534A, 0x004E4942


def read(path: str | Path) -> tuple[dict, bytearray]:
    raw = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<III", raw, 0)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"{path}: not a glTF 2.0 binary")
    o, js, bin_ = 12, None, bytearray()
    while o < length:
        clen, ctype = struct.unpack_from("<II", raw, o)
        data = raw[o + 8:o + 8 + clen]
        if ctype == JSON_CHUNK and js is None:
            js = json.loads(data.decode("utf-8"))
        elif ctype == BIN_CHUNK and not bin_:
            bin_ = bytearray(data)
        o += 8 + clen
    if js is None:
        raise ValueError(f"{path}: no JSON chunk")
    blen = js.get("buffers", [{}])[0].get("byteLength") if js.get("buffers") else 0
    if blen is not None:
        del bin_[blen:]   # BIN chunk padding is not part of buffer 0
    return js, bin_


def write(path: str | Path, js: dict, bin_: bytes | bytearray) -> None:
    """Atomically (temporary file in the same directory, then rename)."""
    jb = json.dumps(js, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    jb += b" " * (-len(jb) % 4)
    bb = bytes(bin_) + b"\0" * (-len(bin_) % 4)
    total = 12 + 8 + len(jb) + (8 + len(bb) if bb else 0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(struct.pack("<III", MAGIC, VERSION, total))
            f.write(struct.pack("<II", len(jb), JSON_CHUNK)); f.write(jb)
            if bb:
                f.write(struct.pack("<II", len(bb), BIN_CHUNK)); f.write(bb)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
