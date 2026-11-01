"""Fixed raw sidecar layout, exact reconstruction and parent-only publication."""
from __future__ import annotations

import copy
import hashlib
import struct
from contextlib import contextmanager

import pytest

from gf_common.candidate_columns import (
    MAX_COLUMN_BYTES,
    column_layout,
    decode_candidate_columns,
    encode_candidate_columns,
)
from gf_common.domain import ContractError, digest
from gf_common.structured import CandidateTable
from gf_training.candidate_storage import (
    candidate_basename,
    publish_candidates,
    read_candidates,
    sidecar_inventory,
)
from tests.test_structured_records import frame_for


def table_fixture(make_input):
    frame, _ = frame_for(make_input(hand=("exchange", "sell", "buy", "steel-armor"), learned=("fireball",)))
    actions = [{"type": "pray"},
               *[{"type": "play_sequence", "ordered_elements": [frame.references[4]] * 18,
                  **({} if target == "omitted" else {"target": target})} for target in ("omitted", None, 1, 2)],
               {"type": "exchange", "artifact": 1, "hp": -5, "mp": 0, "money": 2**32 - 1},
               {"type": "sell", "artifact": 2, "sold_artifact": 4, "target": 2},
               {"type": "buy", "artifact": 3, "target": 1},
               {"type": "discard", "instance_ids": [4]}]
    return frame, CandidateTable.from_actions(frame, actions), actions


def test_fixed_little_endian_layout_and_raw_zero_based_codes(make_input):
    frame, table, actions = table_fixture(make_input)
    payload, descriptor = encode_candidate_columns(table, "player-1-decision-0000000000.columns")
    columns = {item["name"]: item for item in descriptor["columns"]}
    assert [item["name"] for item in descriptor["columns"]] == ["kinds", "targets", "numerics", "presence", "offsets", "slots"]
    assert payload[0] == 0  # pray rawcode0; no token PAD/MISSING offsets.
    targets = columns["targets"]["offset"]
    assert payload[targets + 1:targets + 5] == bytes((0, 1, 2, 3))
    assert columns["numerics"]["offset"] % 8 == columns["offsets"]["offset"] % 8 == 0
    start = columns["numerics"]["offset"] + 5 * 4 * 8
    assert payload[start:start + 8] == struct.pack("<q", -5)
    assert payload[start + 16:start + 24] == struct.pack("<q", 2**32 - 1)
    decoded = decode_candidate_columns(payload, descriptor, expected_frame_hash=frame.frame_hash)
    decoded.validate(frame)
    assert decoded == table
    assert [decoded.to_action(index, frame) for index in range(len(actions))] == actions


@pytest.mark.parametrize("length", [18, 128])
def test_ordered_long_repeated_learned_references_and_active_capacity(make_input, length):
    frame, _ = frame_for(make_input(learned=("fireball",)))
    table = CandidateTable.from_actions(frame, [{"type": "play_sequence", "ordered_elements": [frame.references[-1]] * length}])
    payload, descriptor = encode_candidate_columns(table, "owned.columns", max_positions=length)
    assert decode_candidate_columns(payload, descriptor, max_positions=length) == table
    with pytest.raises(ContractError, match="capacity"):
        decode_candidate_columns(payload, descriptor, max_positions=length - 1)
    with pytest.raises(ContractError, match="capacity"):
        encode_candidate_columns(table, "owned.columns", max_positions=length - 1)


def rehashed(payload, descriptor):
    value = copy.deepcopy(descriptor)
    value["sha256"] = hashlib.sha256(payload).hexdigest()
    return value


@pytest.mark.parametrize("corruption", ["padding", "kind", "target", "presence", "slot", "offset", "absent_numeric", "trailing"])
def test_rehashed_corruption_rejects_before_use(make_input, corruption):
    frame, _ = frame_for(make_input())
    table = CandidateTable.from_actions(frame, [{"type": "pray"}])
    raw, descriptor = encode_candidate_columns(table, "owned.columns")
    payload = bytearray(raw)
    columns = {item["name"]: item for item in descriptor["columns"]}
    if corruption == "padding":
        payload[2] = 1
    elif corruption in ("kind", "target", "presence"):
        name = {"kind": "kinds", "target": "targets", "presence": "presence"}[corruption]
        payload[columns[name]["offset"]] = 255
    elif corruption == "slot":
        payload.append(18)
        descriptor["elements"] = 1
        descriptor["columns"], descriptor["bytes"] = column_layout(1, 1)
        struct.pack_into("<q", payload, columns["offsets"]["offset"] + 8, 1)
    elif corruption == "offset":
        struct.pack_into("<q", payload, columns["offsets"]["offset"], -1)
    elif corruption == "absent_numeric":
        struct.pack_into("<q", payload, columns["numerics"]["offset"], 1)
    else:
        payload += b"trailing"
    owned = bytes(payload)
    with pytest.raises(ContractError):
        decode_candidate_columns(owned, rehashed(owned, descriptor))


@pytest.mark.parametrize("changes", [{"codec": "unknown"}, {"schema_version": True}, {"count": True},
    {"file": "../other.columns"}, {"bytes": 1}, {"unknown": 1}, {"frame_hash": "b" * 64}, {"table_hash": "b" * 64}])
def test_closed_descriptor_and_original_frame_binding(make_input, changes):
    frame, table, _ = table_fixture(make_input)
    payload, descriptor = encode_candidate_columns(table, "owned.columns")
    with pytest.raises(ContractError):
        decode_candidate_columns(payload, {**descriptor, **changes}, expected_frame_hash=frame.frame_hash)
    altered = copy.deepcopy(descriptor)
    altered["columns"][2]["offset"] += 1
    with pytest.raises(ContractError, match="layout"):
        decode_candidate_columns(payload, altered)


def test_limits_are_admitted_before_bytearray_allocation(make_input, monkeypatch):
    import gf_common.candidate_columns as module
    _, table, _ = table_fixture(make_input)
    _, exact = column_layout(len(table.kinds), len(table.slots))
    def forbidden(*args, **kwargs):
        pytest.fail("payload allocation preceded byte admission")
    monkeypatch.setattr(module, "bytearray", forbidden, raising=False)
    with pytest.raises(ContractError, match="limit"):
        encode_candidate_columns(table, "owned.columns", max_bytes=exact - 1)
    _, maximum = column_layout(65536, 65536 * 128)
    assert maximum == 11403272 and maximum < MAX_COLUMN_BYTES
    with pytest.raises(ContractError):
        column_layout(65537, 0)
    with pytest.raises(ContractError):
        column_layout(1, 129)


def test_parent_publication_reconstruction_inventory_and_immutable_collision(make_input, tmp_path):
    frame, table, _ = table_fixture(make_input)
    descriptor = publish_candidates(tmp_path, frame, table, 0)
    assert descriptor["file"] == candidate_basename(1, 0)
    assert not list(tmp_path.glob("*.tmp"))
    assert read_candidates(tmp_path, descriptor, frame, 0) == table
    inventory = sidecar_inventory(tmp_path, descriptor, frame, 0)
    assert inventory["sha256"] == descriptor["sha256"] and inventory["table_hash"] == table.table_hash
    with pytest.raises(ContractError, match="exists"):
        publish_candidates(tmp_path, frame, table, 0)
    with pytest.raises(ContractError, match="foreign"):
        read_candidates(tmp_path, descriptor, frame, 1)
    path = tmp_path / descriptor["file"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ContractError, match="size"):
        read_candidates(tmp_path, descriptor, frame, 0)


def test_parent_refuses_dangling_sidecars_and_preserves_unpublished_temporary(make_input, tmp_path, monkeypatch):
    import gf_training.candidate_storage as module
    frame, table, _ = table_fixture(make_input)
    outside = tmp_path / "outside"
    final = tmp_path / candidate_basename(1, 0)
    final.symlink_to(outside)
    with pytest.raises(ContractError, match="exists"):
        publish_candidates(tmp_path, frame, table, 0)
    assert not outside.exists()
    final.unlink()
    def failed_link(*args, **kwargs):
        raise OSError("publication failed")
    monkeypatch.setattr(module.os, "link", failed_link)
    with pytest.raises(OSError, match="publication"):
        publish_candidates(tmp_path, frame, table, 0)
    assert not final.exists() and (tmp_path / (final.name + ".tmp")).is_file()


def test_reader_refuses_missing_symlink_or_foreign_own_reference(make_input, tmp_path):
    frame, table, _ = table_fixture(make_input)
    descriptor = publish_candidates(tmp_path, frame, table, 0)
    path = tmp_path / descriptor["file"]
    payload = path.read_bytes()
    path.unlink()
    with pytest.raises(ContractError, match="missing"):
        read_candidates(tmp_path, descriptor, frame, 0)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_bytes(payload)
    path.symlink_to(elsewhere)
    with pytest.raises(ContractError, match="unsafe"):
        read_candidates(tmp_path, descriptor, frame, 0)
    path.unlink()
    fields = table._content()
    fields["slots"] = (17, *table.slots[1:])
    forged = CandidateTable(**fields, table_hash=digest(fields))
    # Pure structural inspection cannot certify original own references.
    payload, description = encode_candidate_columns(forged, path.name)
    path.write_bytes(payload)
    with pytest.raises(ContractError, match="slot"):
        read_candidates(tmp_path, description, frame, 0)


@pytest.mark.parametrize("mutation", ["overwrite", "replacement"])
def test_inventory_never_pairs_old_hash_with_refreshed_unverified_identity(make_input, tmp_path, monkeypatch, mutation):
    import gf_training.candidate_storage as module
    frame, table, _ = table_fixture(make_input)
    descriptor = publish_candidates(tmp_path, frame, table, 0)
    original_read = module._verified_read
    path = tmp_path / descriptor["file"]
    @contextmanager
    def read_then_change(*args, **kwargs):
        with original_read(*args, **kwargs) as result:
            if mutation == "overwrite":
                altered = bytearray(path.read_bytes())
                altered[0] = 255
                path.write_bytes(altered)
            else:
                replacement = tmp_path / "replacement"
                replacement.write_bytes(path.read_bytes())
                replacement.replace(path)
            yield result
    monkeypatch.setattr(module, "_verified_read", read_then_change)
    with pytest.raises(ContractError, match="changed after verified read"):
        sidecar_inventory(tmp_path, descriptor, frame, 0)


def test_inventory_hash_bytes_and_identity_come_from_one_verified_snapshot(make_input, tmp_path):
    import gf_training.candidate_storage as module
    frame, table, _ = table_fixture(make_input)
    descriptor = publish_candidates(tmp_path, frame, table, 0)
    inventory = sidecar_inventory(tmp_path, descriptor, frame, 0)
    path = tmp_path / descriptor["file"]
    assert inventory["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert inventory["identity"] == list(module._identity(path.stat()))
    assert inventory["bytes"] == path.stat().st_size


@pytest.mark.parametrize("mutation", ["symlink", "replacement", "inplace"])
def test_publication_checks_actual_published_object_against_open_owned_fd(make_input, tmp_path, monkeypatch, mutation):
    import gf_training.candidate_storage as module
    frame, table, _ = table_fixture(make_input)
    outside = tmp_path / "outside"
    outside.write_bytes(b"must remain unchanged")
    original_link = module.os.link
    def mutate_link(source, destination, **kwargs):
        source = type(tmp_path)(source)
        if mutation == "symlink":
            source.unlink()
            source.symlink_to(outside)
        elif mutation == "replacement":
            payload = source.read_bytes()
            source.unlink()
            source.write_bytes(payload)
        else:
            with source.open("r+b") as output:
                output.write(b"\xff")
        return original_link(source, destination, **kwargs)
    monkeypatch.setattr(module.os, "link", mutate_link)
    with pytest.raises(ContractError):
        publish_candidates(tmp_path, frame, table, 0)
    assert outside.read_bytes() == b"must remain unchanged"
    # A failed publish never returns a descriptor and preserves all orphan paths.
    assert (tmp_path / (candidate_basename(1, 0) + ".tmp")).exists() or (tmp_path / (candidate_basename(1, 0) + ".tmp")).is_symlink()


def test_fifo_sidecar_open_is_nonblocking_and_probe_child_is_reaped(make_input, tmp_path):
    import json
    import os
    import subprocess
    import sys

    from gf_common.structured_wire import frame_dict
    frame, table, _ = table_fixture(make_input)
    descriptor = publish_candidates(tmp_path, frame, table, 0)
    path = tmp_path / descriptor["file"]
    path.unlink()
    os.mkfifo(path)
    code = ('import json,sys; from pathlib import Path; from gf_common.domain import ContractError; '
            'from gf_common.structured_wire import parse_frame; from gf_training.candidate_storage import read_candidates; '
            'directory=Path(sys.argv[1]); descriptor=json.loads(sys.argv[2]); frame=parse_frame(json.loads(sys.argv[3])); '
            '\ntry: read_candidates(directory,descriptor,frame,0)\n'
            'except ContractError: print("rejected")\nelse: raise AssertionError("FIFO accepted")\n')
    completed = subprocess.run([sys.executable, "-c", code, str(tmp_path), json.dumps(descriptor), json.dumps(frame_dict(frame))],
                               timeout=2, check=True, capture_output=True, text=True)
    assert completed.stdout.strip() == "rejected" and completed.returncode == 0


@pytest.mark.parametrize("replacement", ["fifo", "symlink"])
def test_json_actual_fd_guard_rejects_swap_before_open_in_bounded_child(tmp_path, replacement):
    import subprocess
    import sys
    path = tmp_path / "manifest.json"
    path.write_text('{"value":1}')
    outside = tmp_path / "outside.json"
    outside.write_text('{"outside":"unchanged"}')
    code = '''
import os,sys
from pathlib import Path
from gf_common.domain import ContractError
from gf_models.sharded_arrays import read_json
path,outside=Path(sys.argv[1]),Path(sys.argv[2])
kind=sys.argv[3]
actual_os_open=os.open
actual_path_open=Path.open
swapped=False
def swap():
    global swapped
    if not swapped:
        swapped=True
        path.unlink()
        if kind=='fifo': os.mkfifo(path)
        else: path.symlink_to(outside)
def os_open(target,*args,**kwargs):
    if Path(target)==path: swap()
    return actual_os_open(target,*args,**kwargs)
def path_open(target,*args,**kwargs):
    if target==path: swap()
    return actual_path_open(target,*args,**kwargs)
os.open=os_open
Path.open=path_open
try: read_json(path)
except ContractError: print('rejected')
else: raise AssertionError('swapped manifest accepted')
'''
    completed = subprocess.run([sys.executable, "-c", code, str(path), str(outside), replacement],
                               timeout=2, check=True, capture_output=True, text=True)
    assert completed.stdout.strip() == "rejected" and completed.returncode == 0
    assert outside.read_text() == '{"outside":"unchanged"}'


def test_publication_rejects_truncation_at_final_hash_eof_without_refreshing_baseline(make_input, tmp_path, monkeypatch):
    import gf_training.candidate_storage as module
    frame, table, _ = table_fixture(make_input)
    payload, _ = encode_candidate_columns(table, candidate_basename(1, 0))
    original_pread = module.os.pread
    eof_probes = 0
    def truncate_after_final_eof(fd, size, offset):
        nonlocal eof_probes
        result = original_pread(fd, size, offset)
        if size == 1 and offset == len(payload):
            eof_probes += 1
            if eof_probes == 2:
                (tmp_path / candidate_basename(1, 0)).write_bytes(b"")
        return result
    monkeypatch.setattr(module.os, "pread", truncate_after_final_eof)
    with pytest.raises(ContractError, match="changed during verification"):
        publish_candidates(tmp_path, frame, table, 0)
    assert eof_probes == 2
    assert (tmp_path / candidate_basename(1, 0)).stat().st_size == 0
