import hashlib
import json
import struct
from compression import zstd

import pytest

from gf_common.group_pack import PackMember, canonical_json, decode_pack, encode_pack


def test_exact_members_all_kinds_and_arbitrary_sizes() -> None:
    kinds = ("journal", "replay", "candidates", "result", "http-log", "simulator", "diagnostic")
    for count in (0, 2, 6, 11):
        members = tuple(
            PackMember(kinds[i % 7], f"group/{i:02}.bin", bytes([255, i]))
            for i in reversed(range(count))
        )
        assert decode_pack(encode_pack(members)) == tuple(sorted(members, key=lambda m: m.name))
    members = tuple(
        PackMember(kind, f"{i}.bin", b"\xff\x00" if i else b"") for i, kind in enumerate(kinds)
    )
    assert decode_pack(encode_pack(members)) == members


def raw_pack(index: bytes, payload: bytes = b"") -> bytes:
    return zstd.compress(b"GFBPACK1" + struct.pack("<Q", len(index)) + index + payload, level=3)


def test_binary_index_contract() -> None:
    packed = encode_pack((PackMember("journal", "z", b"x"), PackMember("result", "a", b"")))
    raw = zstd.decompress(packed)
    assert raw[:8] == b"GFBPACK1"
    length = struct.unpack("<Q", raw[8:16])[0]
    index = json.loads(raw[16 : 16 + length])
    assert index == {
        "format": "group-pack-v1",
        "files": [
            {
                "kind": "result",
                "name": "a",
                "offset": 0,
                "length": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            },
            {
                "kind": "journal",
                "name": "z",
                "offset": 0,
                "length": 1,
                "sha256": hashlib.sha256(b"x").hexdigest(),
            },
        ],
    }
    assert raw[16 + length :] == b"x"


@pytest.mark.parametrize("name", ["", "/a", "a//b", "./a", "a/../b", "a\\b", "a\x00b", "a\nb"])
def test_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError):
        encode_pack((PackMember("journal", name, b""),))


def test_duplicate_names_unknown_kind_and_bounds() -> None:
    with pytest.raises(ValueError):
        encode_pack((PackMember("journal", "x", b""), PackMember("replay", "x", b"")))
    with pytest.raises(ValueError):
        encode_pack((PackMember("unknown", "x", b""),))
    with pytest.raises(ValueError):
        encode_pack((PackMember("journal", "x", b"x" * 4096),), max_uncompressed_bytes=100)
    packed = encode_pack((PackMember("journal", "x", b"x" * 4096),))
    with pytest.raises(ValueError):
        decode_pack(packed, max_uncompressed_bytes=100)
    with pytest.raises(ValueError):
        decode_pack(b"x" * (1024 * 1024 + 101), max_uncompressed_bytes=100)


def test_corruption_frames_and_duplicate_json_keys() -> None:
    valid = encode_pack((PackMember("journal", "x", b"123"),))
    for data in (
        valid[:-1],
        valid + b"x",
        valid + valid,
        b"garbage",
        raw_pack(b'{"format":"group-pack-v1","format":"group-pack-v1","files":[]}'),
    ):
        with pytest.raises(ValueError):
            decode_pack(data)
    raw = zstd.decompress(valid)
    for data in (raw[:-1], raw + b"x", raw[:-1] + b"4"):
        with pytest.raises(ValueError):
            decode_pack(zstd.compress(data))
    length = struct.unpack("<Q", raw[8:16])[0]
    index = json.loads(raw[16 : 16 + length])
    for mutation in ({"offset": True}, {"offset": 1}, {"length": -1}, {"extra": 1}):
        bad = json.loads(json.dumps(index))
        bad["files"][0].update(mutation)
        reason = "invalid Pack entry" if "extra" in mutation else "invalid Pack range or order"
        with pytest.raises(ValueError, match=reason):
            decode_pack(raw_pack(canonical_json(bad), b"123"))


def test_noncanonical_index_is_a_separate_refusal() -> None:
    value = {"format": "group-pack-v1", "files": []}
    with pytest.raises(ValueError, match="noncanonical JSON"):
        decode_pack(raw_pack(json.dumps(value).encode()))
