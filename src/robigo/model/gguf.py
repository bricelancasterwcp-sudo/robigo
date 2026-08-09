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

_MAX_READ = 64 * 1024 * 1024
"""No single GGUF field approaches this. The largest metadata values are the
tokenizer arrays, and those are read element by element, so no individual
read is big. A claimed length above this means a corrupt length field, and
honouring it would allocate that much memory."""


class GGUFError(Exception):
    """The file is not a readable GGUF."""


def _read_exactly(handle: BinaryIO, n: int, what: str) -> bytes:
    """Read exactly `n` bytes or raise. `read` returns a short buffer at EOF
    rather than raising, which reaches `struct.unpack` as `struct.error` and
    reaches `bytes.decode` as a silently truncated string."""
    if not 0 <= n <= _MAX_READ:
        raise GGUFError(
            f"{what} claims {n} bytes, beyond anything a real GGUF field "
            f"contains; the file is corrupt."
        )
    data = handle.read(n)
    if len(data) != n:
        raise GGUFError(
            f"file ends mid-{what}: wanted {n} bytes, got {len(data)}; the "
            f"file is truncated."
        )
    return data


def read_metadata(path: Path) -> dict[str, object]:
    """Key-value metadata only. Returns before the tensor data, so cost
    is independent of the model's size."""
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise GGUFError(f"{path} is not a GGUF file")
        _read_exactly(handle, 4, "version")          # version, unused
        _read_exactly(handle, 8, "tensor_count")     # tensor count, unused
        count = struct.unpack("<Q", _read_exactly(handle, 8, "kv_count"))[0]
        return {_string(handle): _value(handle, _u32(handle)) for _ in range(count)}


def _u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", _read_exactly(handle, 4, "u32"))[0]


def _u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exactly(handle, 8, "u64"))[0]


def _string(handle: BinaryIO, what: str = "string") -> str:
    raw = _read_exactly(handle, _u64(handle), what)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GGUFError(
            f"{what} is not valid UTF-8 ({raw[:24]!r}); the file is corrupt."
        ) from exc


def _value(handle: BinaryIO, kind: int) -> object:
    if kind == _STRING:
        return _string(handle)
    if kind == _ARRAY:
        element = _u32(handle)
        if element == _ARRAY:
            raise GGUFError(
                "nested arrays are not supported; no real GGUF model uses "
                "them, and honouring them would let a crafted file recurse "
                "until the stack is exhausted."
            )
        return [_value(handle, element) for _ in range(_u64(handle))]
    if kind in _SCALARS:
        fmt, size = _SCALARS[kind]
        return struct.unpack(fmt, _read_exactly(handle, size, f"scalar_type_{kind}"))[0]
    raise GGUFError(f"unknown GGUF value type {kind}")
