"""Exact SQLite contract and physical sealing verification, without Source authority."""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from gf_common.group_pack import (
    FORMAT,
    Json,
    StorageSettings,
    canonical_json,
    decode_pack,
    digest,
    member_index,
    parse_json,
    safe_name,
    unpack_raw,
)
from gf_training.group_pack_resources import OwnedDescriptor, remaining, retain_cleanup

DATABASE = "collection.sqlite3"
SCHEMA_VERSION = 1
SCHEMA = {
    "metadata": "CREATE TABLE metadata(key TEXT PRIMARY KEY NOT NULL, value BLOB NOT NULL)",
    "groups": (
        "CREATE TABLE groups(run_id TEXT NOT NULL, group_id TEXT NOT NULL, "
        "pack BLOB NOT NULL, sha256 TEXT NOT NULL, raw_bytes INTEGER NOT NULL, "
        "metadata BLOB NOT NULL, PRIMARY KEY(run_id,group_id))"
    ),
    "episodes": (
        "CREATE TABLE episodes(run_id TEXT NOT NULL, episode_id TEXT NOT NULL, "
        "group_id TEXT NOT NULL, game_id TEXT NULL, metadata BLOB NOT NULL, "
        "PRIMARY KEY(run_id,episode_id), FOREIGN KEY(run_id,group_id) "
        "REFERENCES groups(run_id,group_id))"
    ),
    "members": (
        "CREATE TABLE members(name TEXT PRIMARY KEY NOT NULL, run_id TEXT NOT NULL, "
        "group_id TEXT NOT NULL, kind TEXT NOT NULL, offset INTEGER NOT NULL, "
        "length INTEGER NOT NULL, sha256 TEXT NOT NULL, FOREIGN KEY(run_id,group_id) "
        "REFERENCES groups(run_id,group_id))"
    ),
    "controls": (
        "CREATE TABLE controls(name TEXT PRIMARY KEY NOT NULL, data BLOB NOT NULL, "
        "sha256 TEXT NOT NULL, bytes INTEGER NOT NULL)"
    ),
}
CATALOG_SQL = (
    "CREATE TABLE collections(database TEXT PRIMARY KEY NOT NULL, run_id TEXT NOT NULL, "
    "state TEXT NOT NULL, seal_id TEXT NULL, descriptor TEXT NULL, "
    "descriptor_sha256 TEXT NULL)"
)
DESCRIPTOR_KEYS = {
    "format",
    "schema_version",
    "run_id",
    "seal_id",
    "database",
    "database_bytes",
    "database_sha256",
    "controls",
    "publication",
}


def identity_text(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("invalid storage identity")
    value.encode("utf-8")
    return value


def configure(db: sqlite3.Connection, deadline: float) -> None:
    db.execute(f"PRAGMA busy_timeout={min(5000, max(1, int(remaining(deadline) * 1000)))}")
    db.execute("PRAGMA foreign_keys=ON")
    db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)


def new_database(path: Path, run_id: str, deadline: float, original_fd: int) -> sqlite3.Connection:
    identity_text(run_id)
    # The child exclusively created this held inode before entering SQLite.
    held, current = os.fstat(original_fd), path.lstat()
    if (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino) or held.st_size != 0:
        raise ValueError("new database original identity mismatch")
    db = sqlite3.connect(
        f"file:/proc/self/fd/{original_fd}?mode=rw", uri=True, timeout=min(5.0, remaining(deadline))
    )
    try:
        configure(db, deadline)
        if db.execute("PRAGMA journal_mode=WAL").fetchone() != ("wal",):
            raise RuntimeError("cannot enter WAL mode")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("PRAGMA wal_autocheckpoint=8192")
        db.execute("PRAGMA cache_size=-32768")
        with db:
            db.execute("PRAGMA user_version=1")
            for sql in SCHEMA.values():
                db.execute(sql)
            db.executemany(
                "INSERT INTO metadata VALUES (?,?)",
                [("format", canonical_json(FORMAT)), ("run_id", canonical_json(run_id))],
            )
        return db
    except BaseException as error:
        try:
            db.close()
        except BaseException as cleanup:  # noqa: BLE001 -- preserve initialization failure
            error.__dict__["cleanup_owner"] = db
            error.add_note(f"SQLite cleanup remains owned ({type(cleanup).__name__})")
        raise


def check_schema(db: sqlite3.Connection) -> None:
    if db.execute("PRAGMA user_version").fetchone() != (SCHEMA_VERSION,):
        raise ValueError("unsupported database schema")
    objects = db.execute("SELECT type,name,sql FROM sqlite_master WHERE sql IS NOT NULL").fetchall()
    if {name: sql for kind, name, sql in objects if kind == "table"} != SCHEMA:
        raise ValueError("database schema mismatch")
    if any(kind != "table" for kind, _, _ in objects):
        raise ValueError("unexpected database object")


def metadata(db: sqlite3.Connection, *, sealed: bool) -> dict[str, Json]:
    count, largest = db.execute(
        "SELECT count(*),coalesce(max(length(value)),0) FROM metadata"
    ).fetchone()
    if count != (3 if sealed else 2) or largest > 1024 * 1024:
        raise ValueError("database metadata exceeds admission")
    entries = dict(db.execute("SELECT key,value FROM metadata"))
    if any(type(v) is not bytes for v in entries.values()):
        raise ValueError("database metadata must be original canonical bytes")
    if set(entries) != ({"format", "run_id", "seal"} if sealed else {"format", "run_id"}):
        raise ValueError("database metadata keys mismatch")
    result = {key: parse_json(value) for key, value in entries.items()}
    if result["format"] != FORMAT:
        raise ValueError("database format mismatch")
    identity_text(result["run_id"])
    return result


def submit_group(
    db: sqlite3.Connection,
    run_id: str,
    group_id: str,
    pack: bytes,
    *,
    raw_bytes: int,
    record: Json,
    episodes: list[Json],
    max_pack_bytes: int,
) -> None:
    identity_text(group_id)
    if type(raw_bytes) is not int or raw_bytes < 16 or raw_bytes > max_pack_bytes:
        raise ValueError("invalid uncompressed size")
    if len(unpack_raw(pack, raw_bytes)) != raw_bytes:
        raise ValueError("Pack size mismatch")
    members = member_index(decode_pack(pack, raw_bytes))
    record_bytes = canonical_json(record, limit=1024 * 1024)
    rows = []
    if not isinstance(episodes, list):
        raise TypeError("invalid episode list")
    seen = set()
    for episode in episodes:
        if not isinstance(episode, dict) or set(episode) != {"episode_id", "game_id", "record"}:
            raise ValueError("invalid episode envelope")
        episode_id = identity_text(episode["episode_id"])
        game_id = episode["game_id"]
        if game_id is not None:
            identity_text(game_id)
        if episode_id in seen:
            raise ValueError("duplicate episode ID")
        seen.add(episode_id)
        rows.append(
            (
                run_id,
                episode_id,
                group_id,
                game_id,
                canonical_json(episode["record"], limit=1024 * 1024),
            )
        )
    proposed = (run_id, group_id, pack, digest(pack), raw_bytes, record_bytes)
    with db:
        metadata(db, sealed=False)
        old = db.execute(
            "SELECT * FROM groups WHERE run_id=? AND group_id=?", (run_id, group_id)
        ).fetchone()
        if old is not None:
            old_episodes = db.execute(
                "SELECT * FROM episodes WHERE run_id=? AND group_id=? ORDER BY episode_id",
                (run_id, group_id),
            ).fetchall()
            if old != proposed or old_episodes != sorted(rows, key=lambda r: r[1]):
                raise ValueError("conflicting group resend")
            return
        db.execute("INSERT INTO groups VALUES (?,?,?,?,?,?)", proposed)
        for member in members:
            if db.execute("SELECT 1 FROM controls WHERE name=?", (member["name"],)).fetchone():
                raise ValueError("duplicate logical name")
            db.execute(
                "INSERT INTO members VALUES (?,?,?,?,?,?,?)",
                (
                    member["name"],
                    run_id,
                    group_id,
                    member["kind"],
                    member["offset"],
                    member["length"],
                    member["sha256"],
                ),
            )
        db.executemany("INSERT INTO episodes VALUES (?,?,?,?,?)", rows)


def store_control(db: sqlite3.Connection, name: str, data: bytes) -> None:
    safe_name(name)
    proposed = (name, data, digest(data), len(data))
    with db:
        metadata(db, sealed=False)
        if db.execute("SELECT 1 FROM members WHERE name=?", (name,)).fetchone():
            raise ValueError("duplicate logical name")
        old = db.execute("SELECT * FROM controls WHERE name=?", (name,)).fetchone()
        if old is not None:
            if old != proposed:
                raise ValueError("conflicting control resend")
            return
        db.execute("INSERT INTO controls VALUES (?,?,?,?)", proposed)


def logical_index(db: sqlite3.Connection, run_id: str) -> dict[str, Json]:
    """Cloned JSON index, never BLOBs or a learning receipt."""
    return {
        "format": FORMAT,
        "schema_version": 1,
        "run_id": run_id,
        "groups": [
            {"group_id": g, "sha256": h, "raw_bytes": n, "record": parse_json(m)}
            for g, h, n, m in db.execute(
                "SELECT group_id,sha256,raw_bytes,metadata FROM groups ORDER BY group_id"
            )
        ],
        "episodes": [
            {"episode_id": e, "group_id": g, "game_id": game, "record": parse_json(m)}
            for e, g, game, m in db.execute(
                "SELECT episode_id,group_id,game_id,metadata FROM episodes ORDER BY episode_id"
            )
        ],
        "members": [
            {"name": n, "group_id": g, "kind": k, "offset": o, "length": l, "sha256": h}
            for n, g, k, o, l, h in db.execute(
                "SELECT name,group_id,kind,offset,length,sha256 FROM members ORDER BY name"
            )
        ],
        "controls": [
            {"name": n, "bytes": s, "sha256": h}
            for n, s, h in db.execute("SELECT name,bytes,sha256 FROM controls ORDER BY name")
        ],
    }


def index_admission(db: sqlite3.Connection, max_owned_bytes: int) -> tuple[int, int]:
    """Admit SQL-measured row/index metadata before materializing Python objects."""
    lengths = db.execute(
        "SELECT coalesce(sum(length(metadata)+length(group_id)+128),0) FROM groups"
    ).fetchone()[0]
    lengths += db.execute(
        "SELECT coalesce(sum(length(metadata)+length(episode_id)+"
        "length(group_id)+coalesce(length(game_id),0)+128),0) "
        "FROM episodes"
    ).fetchone()[0]
    lengths += db.execute(
        "SELECT coalesce(sum(length(name)+length(group_id)+length(kind)+128),0) FROM members"
    ).fetchone()[0]
    lengths += db.execute("SELECT coalesce(sum(length(name)+128),0) FROM controls").fetchone()[0]
    control_count = db.execute("SELECT count(*) FROM controls").fetchone()[0]
    metadata_charge = 32 * lengths + 65536 * (1 + control_count)
    if metadata_charge > max_owned_bytes:
        raise ValueError("database index exceeds admission")
    return lengths, metadata_charge


def seal_transaction(
    db: sqlite3.Connection, run_id: str, *, max_owned_bytes: int
) -> dict[str, Json]:
    metadata(db, sealed=False)
    _, charge = index_admission(db, max_owned_bytes)
    seal = {
        "schema_version": 1,
        "seal_id": str(uuid4()),
        "run_id": run_id,
        "logical_sha256": digest(canonical_json(logical_index(db, run_id), limit=charge)),
    }
    with db:
        db.execute("INSERT INTO metadata VALUES (?,?)", ("seal", canonical_json(seal)))
    return seal


def validate_seal_object(seal: Json, run_id: str) -> None:
    if (
        not isinstance(seal, dict)
        or set(seal) != {"schema_version", "seal_id", "run_id", "logical_sha256"}
        or type(seal["schema_version"]) is not int
        or seal["schema_version"] != 1
        or seal["run_id"] != run_id
        or not is_digest(seal["logical_sha256"])
    ):
        raise ValueError("invalid seal")
    try:
        value = UUID(seal["seal_id"])
        if value.version != 4 or str(value) != seal["seal_id"]:
            raise ValueError("invalid seal identity")
    except (TypeError, AttributeError) as exc:
        raise ValueError("invalid seal identity") from exc


def is_digest(value: Json) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def admit_database(
    db: sqlite3.Connection, *, deadline: float, max_pack_bytes: int, max_owned_bytes: int
) -> dict[str, Json]:
    """Bounded metadata admission: never fetch a Pack or run whole-DB pragmas."""
    from gf_common.group_pack import KINDS

    configure(db, deadline)
    check_schema(db)
    if db.execute("PRAGMA journal_mode").fetchone() != ("delete",):
        raise ValueError("sealed database must use DELETE journal mode")
    meta = metadata(db, sealed=True)
    run_id, seal = meta["run_id"], meta["seal"]
    validate_seal_object(seal, run_id)
    lengths, metadata_charge = index_admission(db, max_owned_bytes)
    checks = {
        "groups": "typeof(run_id)!='text' OR typeof(group_id)!='text' OR typeof(pack)!='blob' OR typeof(sha256)!='text' OR typeof(raw_bytes)!='integer' OR typeof(metadata)!='blob'",
        "episodes": "typeof(run_id)!='text' OR typeof(episode_id)!='text' OR typeof(group_id)!='text' OR (game_id IS NOT NULL AND typeof(game_id)!='text') OR typeof(metadata)!='blob'",
        "members": "typeof(name)!='text' OR typeof(run_id)!='text' OR typeof(group_id)!='text' OR typeof(kind)!='text' OR typeof(offset)!='integer' OR typeof(length)!='integer' OR typeof(sha256)!='text'",
        "controls": "typeof(name)!='text' OR typeof(data)!='blob' OR typeof(sha256)!='text' OR typeof(bytes)!='integer'",
    }
    for table, condition in checks.items():
        if db.execute(f"SELECT 1 FROM {table} WHERE {condition} LIMIT 1").fetchone():
            raise ValueError("database row type mismatch")
    index = logical_index(db, run_id)
    for table in ("groups", "episodes", "members"):
        if db.execute(f"SELECT 1 FROM {table} WHERE run_id != ? LIMIT 1", (run_id,)).fetchone():
            raise ValueError("database contains another run")
    groups = {g['group_id']: g for g in index['groups']}
    compressed = dict(db.execute("SELECT group_id,length(pack) FROM groups"))
    for group_id, group in groups.items():
        remaining(deadline)
        identity_text(group_id)
        raw_size, size = group['raw_bytes'], compressed[group_id]
        if (not is_digest(group['sha256']) or raw_size < 16 or raw_size > max_pack_bytes
                or size < 1 or size > max_pack_bytes + 1024 * 1024
                or 4 * raw_size + 2 * size + metadata_charge > max_owned_bytes):
            raise ValueError("group exceeds admission")
    cursors = dict.fromkeys(groups, 0)
    seen: set[str] = set()
    for member in index['members']:
        remaining(deadline)
        name, group_id = safe_name(member['name']), member['group_id']
        if (name in seen or group_id not in groups or member['kind'] not in KINDS
                or not is_digest(member['sha256']) or member['offset'] != cursors.get(group_id)
                or member['length'] < 0
                or member['offset'] + member['length'] > groups[group_id]['raw_bytes'] - 16):
            raise ValueError("member lookup range/reference mismatch")
        cursors[group_id] += member['length']
        seen.add(name)
    for episode in index['episodes']:
        identity_text(episode['episode_id'])
        if episode['group_id'] not in groups:
            raise ValueError("episode group reference mismatch")
        if episode['game_id'] is not None:
            identity_text(episode['game_id'])
    for name, size, packed_size, expected_hash in db.execute(
        "SELECT name,bytes,length(data),sha256 FROM controls ORDER BY name"
    ):
        remaining(deadline)
        safe_name(name)
        if (name in seen or size != packed_size or size < 0
                or not is_digest(expected_hash)
                or size > min(max_pack_bytes, 64 * 1024 * 1024)
                or metadata_charge + size * 2 > max_owned_bytes):
            raise ValueError("control exceeds admission or duplicates name")
        data = db.execute("SELECT data FROM controls WHERE name=?", (name,)).fetchone()[0]
        if digest(data) != expected_hash:
            raise ValueError("control BLOB mismatch")
    if digest(canonical_json(index, limit=metadata_charge)) != seal["logical_sha256"]:
        raise ValueError("logical seal digest mismatch")
    return {"seal": seal, "index": index, "metadata_bytes": lengths,
            "metadata_charge": metadata_charge, "compressed_sizes": compressed}


def verified_pack(db: sqlite3.Connection, run_id: str, group: Json, index: Json,
                  compressed_size: int) -> tuple:
    """Fetch a single admitted Pack and decode it once, checking the complete index."""
    from gf_common.group_pack import PACK_FORMAT

    row = db.execute("SELECT pack FROM groups WHERE run_id=? AND group_id=?",
                     (run_id, group['group_id'])).fetchone()
    if (row is None or type(row[0]) is not bytes or len(row[0]) != compressed_size
            or digest(row[0]) != group['sha256']):
        raise ValueError("group BLOB mismatch")
    members = decode_pack(row[0], group['raw_bytes'])
    actual = member_index(members)
    expected = [{k: m[k] for k in ('name', 'kind', 'offset', 'length', 'sha256')}
                for m in index['members'] if m['group_id'] == group['group_id']]
    if actual != expected:
        raise ValueError("member lookup mismatch")
    raw_size = 16 + len(canonical_json({'format': PACK_FORMAT, 'files': actual})) + sum(len(m.data) for m in members)
    if raw_size != group['raw_bytes']:
        raise ValueError("group uncompressed size mismatch")
    return members


def validate_database(
    db: sqlite3.Connection, *, deadline: float, max_pack_bytes: int, max_owned_bytes: int
) -> dict[str, Json]:
    """Explicit full readonly audit, streaming one complete Pack at a time."""
    admitted = admit_database(db, deadline=deadline, max_pack_bytes=max_pack_bytes,
                              max_owned_bytes=max_owned_bytes)
    if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise ValueError("database integrity failure")
    if db.execute("PRAGMA foreign_key_check").fetchall():
        raise ValueError("database foreign key failure")
    for group in admitted['index']['groups']:
        remaining(deadline)
        members = verified_pack(db, admitted['seal']['run_id'], group, admitted['index'],
                                admitted['compressed_sizes'][group['group_id']])
        del members
    remaining(deadline)
    return admitted


def finish_wal(db: sqlite3.Connection) -> dict[str, float]:
    checkpoint_started = time.perf_counter()
    checkpoint = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if (
        checkpoint is None
        or len(checkpoint) != 3
        or checkpoint[0] != 0
        or checkpoint[1] != checkpoint[2]
    ):
        raise RuntimeError("WAL checkpoint incomplete")
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    delete_started = time.perf_counter()
    if db.execute("PRAGMA journal_mode=DELETE").fetchone() != ("delete",):
        raise RuntimeError("DELETE journal transition failed")
    return {"checkpoint_seconds": checkpoint_seconds,
            "delete_mode_seconds": time.perf_counter() - delete_started}


def fsync_directory(directory: Path) -> None:
    owner = OwnedDescriptor(os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW))
    try:
        os.fsync(owner.fd)
    except BaseException as error:
        retain_cleanup(owner, error)
        try:
            owner.close()
        except BaseException as cleanup:  # noqa: BLE001 -- preserve sync failure
            error.add_note(f"directory descriptor cleanup uncertain ({type(cleanup).__name__})")
        raise
    else:
        owner.close()


class CatalogConflictError(ValueError):
    """This transaction could not create its requested fresh catalog row."""


def catalog_update(
    settings: StorageSettings,
    database: Path,
    run_id: str,
    state: str,
    deadline: float,
    descriptor: Json = None,
    *,
    create_only: bool = False,
) -> None:
    if state not in ("writing", "sealed", "failed"):
        raise ValueError("invalid catalog state")
    if create_only and state != "sealed":
        raise ValueError("exclusive catalog creation requires a complete seal")
    settings.db_root.mkdir(parents=True, exist_ok=True)
    path = settings.db_root / "catalog.sqlite3"
    if path.is_symlink():
        raise ValueError("symlink catalog")
    with closing(sqlite3.connect(path, timeout=min(5.0, remaining(deadline)))) as db:
        configure(db, deadline)
        db.execute("PRAGMA synchronous=FULL")
        with db:
            db.execute("BEGIN IMMEDIATE")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("unsupported catalog schema")
            db.execute("PRAGMA user_version=1")
            if version == 0:
                db.execute(CATALOG_SQL)
            objects = db.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL").fetchall()
            if objects != [(CATALOG_SQL,)]:
                raise ValueError("catalog schema mismatch")
            key = str(database.resolve())
            old = db.execute(
                "SELECT run_id,state FROM collections WHERE database=?", (key,)
            ).fetchone()
            if create_only and old is not None:
                raise CatalogConflictError("catalog identity conflict")
            if old is not None and (
                old[0] != run_id or (state == "writing" and old[1] == "sealed")
            ):
                raise ValueError("catalog identity conflict")
            seal_id = descriptor["seal_id"] if descriptor else None
            descriptor_path = (
                str(database.parent / descriptor_name(descriptor["publication"]))
                if descriptor
                else None
            )
            descriptor_hash = digest(canonical_json(descriptor)) if descriptor else None
            if state == "failed" and old is not None:
                db.execute("UPDATE collections SET state=? WHERE database=?", (state, key))
            else:
                db.execute(
                    "INSERT INTO collections VALUES (?,?,?,?,?,?) ON CONFLICT(database) "
                    "DO UPDATE SET state=excluded.state,seal_id=excluded.seal_id,"
                    "descriptor=excluded.descriptor,descriptor_sha256=excluded.descriptor_sha256",
                    (key, run_id, state, seal_id, descriptor_path, descriptor_hash),
                )


def descriptor_name(publication: str) -> str:
    if publication not in ("published", "pending"):
        raise ValueError("invalid descriptor publication")
    return f"collection.{publication}.json"


def validate_descriptor(value: Json) -> dict[str, Json]:
    if (
        not isinstance(value, dict)
        or set(value) != DESCRIPTOR_KEYS
        or value["format"] != FORMAT
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(value["database_bytes"]) is not int
        or value["database_bytes"] <= 0
        or not is_digest(value["database_sha256"])
        or not isinstance(value["controls"], list)
    ):
        raise ValueError("invalid physical descriptor")
    identity_text(value["run_id"])
    validate_seal_object(
        {
            "schema_version": 1,
            "seal_id": value["seal_id"],
            "run_id": value["run_id"],
            "logical_sha256": "0" * 64,
        },
        value["run_id"],
    )
    if safe_name(value["database"]) != DATABASE:
        raise ValueError("invalid database basename")
    descriptor_name(value["publication"])
    previous = ""
    for control in value["controls"]:
        if (
            not isinstance(control, dict)
            or set(control) != {"name", "bytes", "sha256"}
            or type(control["bytes"]) is not int
            or control["bytes"] < 0
            or not is_digest(control["sha256"])
            or safe_name(control["name"]) <= previous
        ):
            raise ValueError("invalid descriptor controls")
        previous = control["name"]
    return value


def reconcile_collection(
    directory: Path, *, deadline: float, max_owned_bytes: int
) -> dict[str, Json]:
    """Inspect originals, never repair DBs or turn catalog rows into authority."""
    from gf_training.group_pack_source import inspect_packed

    descriptors = [
        directory / descriptor_name(p)
        for p in ("published", "pending")
        if (directory / descriptor_name(p)).exists()
    ]
    if len(descriptors) > 1:
        raise ValueError("ambiguous descriptor namespace")
    if descriptors:
        value = inspect_packed(
            descriptors[0], deadline=deadline, max_owned_bytes=max_owned_bytes, allow_pending=True
        )
        return {"state": value["descriptor"]["publication"], "descriptor": value["descriptor"]}
    path = directory / DATABASE
    if not path.exists():
        return {"state": "failed/unsealed"}
    # Diagnostic seal/index/control observation only: no Pack/full audit or Source authority.
    try:
        value = inspect_packed(
            path,
            deadline=deadline,
            max_owned_bytes=max_owned_bytes,
            allow_pending=True,
            without_descriptor=True,
        )
    except (ValueError, sqlite3.DatabaseError) as error:
        remaining(deadline)
        owner = error.__dict__.get('cleanup_owner')
        if owner is not None and not getattr(owner, '_closed', False):
            raise
        return {"state": "failed/unsealed"}
    remaining(deadline)
    return {"state": "sealed-without-descriptor", "seal": value["validated"]["seal"]}
