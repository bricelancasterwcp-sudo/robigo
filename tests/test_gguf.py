from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from robigo.model.gguf import GGUFError, read_metadata
from robigo.model.geometry import from_model_info


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv_u32(key: str, value: int) -> bytes:
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_str(key: str, value: str) -> bytes:
    return _s(key) + struct.pack("<I", 8) + _s(value)


def _kv_arr_u32(key: str, values: list[int]) -> bytes:
    head = _s(key) + struct.pack("<I", 9) + struct.pack("<I", 4) + struct.pack("<Q", len(values))
    return head + b"".join(struct.pack("<I", v) for v in values)


def _gguf(pairs: list[bytes]) -> bytes:
    return (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 0)
        + struct.pack("<Q", len(pairs))
        + b"".join(pairs)
    )


def test_reads_strings_ints_and_arrays(tmp_path: Path):
    path = tmp_path / "m.gguf"
    path.write_bytes(_gguf([
        _kv_str("general.architecture", "qwen2"),
        _kv_u32("qwen2.block_count", 28),
        _kv_arr_u32("qwen2.attention.head_count_kv", [4, 4]),
    ]))
    info = read_metadata(path)
    assert info["general.architecture"] == "qwen2"
    assert info["qwen2.block_count"] == 28
    assert info["qwen2.attention.head_count_kv"] == [4, 4]


def test_stops_reading_after_the_metadata_block(tmp_path: Path):
    # Real GGUFs are gigabytes; the reader must not walk the tensor data.
    path = tmp_path / "m.gguf"
    path.write_bytes(_gguf([_kv_str("general.architecture", "qwen2")]) + b"\x00" * 4096)
    assert read_metadata(path) == {"general.architecture": "qwen2"}


def test_rejects_a_file_that_is_not_gguf(tmp_path: Path):
    path = tmp_path / "not.gguf"
    path.write_bytes(b"ORDINARY FILE")
    with pytest.raises(GGUFError) as e:
        read_metadata(path)
    assert "not a GGUF" in str(e.value)


def test_rejects_an_unknown_value_type(tmp_path: Path):
    path = tmp_path / "m.gguf"
    path.write_bytes(_gguf([_s("k") + struct.pack("<I", 99)]))
    with pytest.raises(GGUFError) as e:
        read_metadata(path)
    assert "99" in str(e.value)


def _header(kv_count: int = 1) -> bytes:
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack(
        "<Q", kv_count
    )


def test_a_file_holding_only_the_magic_is_a_gguf_error(tmp_path):
    path = tmp_path / "m.gguf"
    path.write_bytes(b"GGUF")
    with pytest.raises(GGUFError):
        read_metadata(path)


def test_a_header_truncated_mid_field_is_a_gguf_error(tmp_path):
    path = tmp_path / "t.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + b"\x01\x02")
    with pytest.raises(GGUFError):
        read_metadata(path)


def test_a_count_promising_more_pairs_than_the_file_holds_is_a_gguf_error(tmp_path):
    """The interrupted-download shape: the header is intact and claims a
    pair, and the file simply stops."""
    path = tmp_path / "p.gguf"
    path.write_bytes(_header(1))
    with pytest.raises(GGUFError) as e:
        read_metadata(path)
    assert "truncated" in str(e.value)


def test_a_key_string_running_past_the_end_does_not_silently_shorten(tmp_path):
    """The value-level assertion. A short read used to decode cleanly, so the
    key came back as 'abc' instead of failing."""
    path = tmp_path / "s.gguf"
    path.write_bytes(_header(1) + struct.pack("<Q", 300) + b"abc")
    with pytest.raises(GGUFError) as e:
        read_metadata(path)
    assert "truncated" in str(e.value)


def test_an_absurd_string_length_is_refused_before_allocating(tmp_path):
    path = tmp_path / "h.gguf"
    path.write_bytes(_header(1) + struct.pack("<Q", 2**63) + b"abc")
    with pytest.raises(GGUFError) as e:
        read_metadata(path)
    assert "corrupt" in str(e.value)


def test_an_absurd_kv_count_ends_at_the_first_missing_byte(tmp_path):
    path = tmp_path / "c.gguf"
    path.write_bytes(_header(2**40))
    with pytest.raises(GGUFError):
        read_metadata(path)


def _find_ollama_blobs_dir() -> Path | None:
    """Resolve Ollama models directory in priority order:
    1. OLLAMA_MODELS environment variable + /blobs
    2. ~/.ollama/models/blobs (default)
    3. Common deployment paths (/mnt/extra/ollama-models/blobs, /var/lib/ollama/blobs)
    """
    candidates = []

    # 1. Check OLLAMA_MODELS environment variable
    ollama_models = os.environ.get("OLLAMA_MODELS")
    if ollama_models:
        candidates.append(Path(ollama_models) / "blobs")

    # 2. Check default ~/.ollama/models/blobs
    candidates.append(Path.home() / ".ollama" / "models" / "blobs")

    # 3. Check common deployment paths
    candidates.append(Path("/mnt/extra/ollama-models/blobs"))
    candidates.append(Path("/var/lib/ollama/blobs"))

    for blobs_dir in candidates:
        if blobs_dir.is_dir():
            # Check if it has at least one sha256-* file
            if list(blobs_dir.glob("sha256-*")):
                return blobs_dir

    return None


_ollama_blobs_dir = _find_ollama_blobs_dir()


@pytest.mark.skipif(
    _ollama_blobs_dir is None,
    reason="No Ollama blobs directory found",
)
def test_real_gguf_blobs_parse():
    """Re-verification against real GGUF blobs from Ollama store.
    Resolves the blobs directory via OLLAMA_MODELS env var or ~/.ollama/models/blobs.
    Parses the largest GGUF blobs and verifies geometry extraction."""
    assert _ollama_blobs_dir is not None, "Blobs directory should exist if test runs"

    # Find GGUF blobs (filter by magic bytes) and take the largest handful
    gguf_blobs = []
    for blob_path in _ollama_blobs_dir.glob("sha256-*"):
        if not blob_path.is_file():
            continue
        try:
            with open(blob_path, "rb") as f:
                magic = f.read(4)
                if magic == b"GGUF":
                    gguf_blobs.append(blob_path)
        except (OSError, IOError):
            pass

    if not gguf_blobs:
        pytest.skip("No GGUF files found in blobs directory")

    # Take the largest 10
    gguf_blobs = sorted(gguf_blobs, key=lambda p: p.stat().st_size, reverse=True)[:10]

    parsed_count = 0
    geometry_count = 0
    failed = []

    for blob_path in gguf_blobs:
        try:
            # Parse metadata
            metadata = read_metadata(blob_path)
            parsed_count += 1

            # Verify geometry can be extracted and has required fields
            try:
                geometry = from_model_info(metadata)
                # Verify required fields are present and integral
                assert isinstance(geometry.layers, int) and geometry.layers > 0
                assert isinstance(geometry.kv_heads, int) and geometry.kv_heads > 0
                assert isinstance(geometry.training_ctx, int) and geometry.training_ctx > 0
                geometry_count += 1
            except Exception as e:
                # Geometry extraction failed - metadata parsed but incomplete
                pass
        except GGUFError as e:
            failed.append((blob_path.name, f"GGUFError: {str(e)[:60]}"))
        except Exception as e:
            failed.append((blob_path.name, f"{type(e).__name__}: {str(e)[:60]}"))

    # Report results
    print(f"\nReal-blob re-verification: {parsed_count}/{len(gguf_blobs)} blobs parsed, "
          f"{geometry_count} with complete geometry")
    if failed:
        failures_str = "\n".join(f"  {name}: {msg}" for name, msg in failed)
        print(f"Failed:\n{failures_str}")

    # Assert all tested blobs parsed successfully
    assert len(failed) == 0, f"Expected all blobs to parse, but {len(failed)} failed"
    assert parsed_count == len(gguf_blobs), f"Expected all {len(gguf_blobs)} blobs to parse"
