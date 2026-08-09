from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

_SCALARS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
_STRING = 8
_ARRAY = 9


class GGUFError(Exception):
    """The file is not a readable GGUF."""


def read_metadata(path: Path) -> dict[str, object]:
    """Key-value metadata only. Returns before the tensor data, so cost
    is independent of the model's size."""
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise GGUFError(f"{path} is not a GGUF file")
        struct.unpack("<I", handle.read(4))          # version, unused
        struct.unpack("<Q", handle.read(8))          # tensor count, unused
        count = struct.unpack("<Q", handle.read(8))[0]
        return {_string(handle): _value(handle, _u32(handle)) for _ in range(count)}


def _u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", handle.read(4))[0]


def _u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", handle.read(8))[0]


def _string(handle: BinaryIO) -> str:
    return handle.read(_u64(handle)).decode("utf-8", errors="replace")


def _value(handle: BinaryIO, kind: int) -> object:
    if kind == _STRING:
        return _string(handle)
    if kind == _ARRAY:
        element = _u32(handle)
        return [_value(handle, element) for _ in range(_u64(handle))]
    if kind in _SCALARS:
        fmt, size = _SCALARS[kind]
        return struct.unpack(fmt, handle.read(size))[0]
    raise GGUFError(f"unknown GGUF value type {kind}")
