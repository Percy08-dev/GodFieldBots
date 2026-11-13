"""Durable campaign records and measured destination admission; no GPU work."""
import hashlib
from dataclasses import replace

import pytest

from gf_cli.hpo_config import HPOStorageConfig
from gf_cli.hpo_storage import (
    CampaignBusy,
    CampaignStore,
    check_storage,
    mount_facts,
    windows_drive,
)
from gf_common.domain import ContractError


def test_campaign_records_are_owned_immutable_and_require_the_held_lock(tmp_path):
    directory = tmp_path / "campaign"
    value = {"schema_version":1,"kind":"hpo-campaign"}
    with CampaignStore(directory,mode="fresh",max_owned_bytes=1024**2) as store:
        expected = hashlib.sha256(b'{"kind":"hpo-campaign","schema_version":1}\n').hexdigest()
        assert store.put("campaign.json",value) == expected
        assert store.put("campaign.json",value) == expected
        assert store.get("campaign.json") == (value,expected)
        assert store.records() == ("campaign.json",)
        with pytest.raises(CampaignBusy): CampaignStore(directory,mode="resume",max_owned_bytes=1024**2)
        with pytest.raises(ContractError): store.put("campaign.json",{**value,"kind":"other"})
    with pytest.raises(ContractError): store.put("other.json",value)
    with CampaignStore(directory,mode="resume",max_owned_bytes=1024**2) as reopened:
        assert reopened.get("campaign.json") == (value,expected)
    with pytest.raises(FileExistsError): CampaignStore(directory,mode="fresh",max_owned_bytes=1024**2)


@pytest.mark.parametrize("name", ["../escape.json","/absolute.json","a/../b.json","a//b.json","a/./b.json"])
def test_campaign_records_cannot_escape_owned_namespace(tmp_path,name):
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=1024**2) as store, pytest.raises(ContractError):
        store.put(name,{"schema_version":1})


@pytest.mark.parametrize("raw", [b'{"x":1e999}\n',b'{"x":-1e999}\n',b'{"x":1,"x":2}\n',b'{ "x":1}\n'])
def test_saved_record_corruption_cannot_be_reinterpreted_or_overwritten(tmp_path,raw):
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=1024**2) as store:
        path = store.directory / "record.json"
        path.write_bytes(raw)
        with pytest.raises(ContractError): store.get("record.json")
        with pytest.raises(ContractError): store.put("record.json",{"x":1})
        assert path.read_bytes() == raw


def test_fsync_failure_poisoned_owner_never_reports_a_saved_proposal(tmp_path,monkeypatch):
    from gf_cli import hpo_storage as storage

    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=1024**2) as store:
        store.put("campaign.json",{"schema_version":1})
        sync = storage.os.fsync
        monkeypatch.setattr(storage.os,"fsync",lambda fd: (_ for _ in ()).throw(OSError("fsync fault")))
        with pytest.raises(OSError,match="fsync fault"): store.put("proposal.json",{"value":1})
        monkeypatch.setattr(storage.os,"fsync",sync)
        with pytest.raises(ContractError): store.put("proposal.json",{"value":1})
        assert not (store.directory / "proposal.json").exists()


def test_symlink_or_oversized_record_has_no_owned_receipt(tmp_path):
    with CampaignStore(tmp_path / "campaign",mode="fresh",max_owned_bytes=1024**2) as store:
        target = tmp_path / "foreign"
        target.write_text("{}")
        (store.directory / "foreign.json").symlink_to(target)
        with pytest.raises(ContractError): store.get("foreign.json")
        with pytest.raises(ContractError): store.put("foreign.json",{})
        with pytest.raises(ContractError): store.put("too-large.json",{"x":"a" * 100_000})
        assert not (store.directory / "too-large.json").exists()


def test_actual_mount_facts_identify_drive_letters_not_mountpoint_labels():
    facts = mount_facts("25 1 8:1 / / rw - ext4 /dev/sda rw\n"
        "26 25 0:1 / /mnt/c rw - 9p C:\\134 rw,aname=drvfs;path=C:\\134;uid=1000\n"
        "27 25 0:2 / /mnt/d rw - 9p D:\\134 rw,aname=drvfs;path=D:\\134;uid=1000\n"
        "28 25 8:3 / /fake/c rw - ext4 /dev/sdc rw\n")
    assert windows_drive(facts[1]) == "C"
    assert windows_drive(facts[2]) == "D"
    assert windows_drive(facts[3]) is None
    assert facts[1].source == "C:\\"


def admitted_storage(tmp_path,monkeypatch):
    from gf_cli import hpo_storage as storage

    controls = tmp_path / "controls"
    windows = tmp_path / "windows-c"
    originals = tmp_path / "windows-d"
    for path in (controls,windows,originals): path.mkdir()
    facts = mount_facts(f"25 1 8:1 / {tmp_path} rw - ext4 /dev/sda rw\n"
        f"26 25 0:1 / {windows} rw - 9p C:\\134 rw,aname=drvfs;path=C:\\134\n"
        f"27 25 0:2 / {originals} rw - 9p D:\\134 rw,aname=drvfs;path=D:\\134\n")
    monkeypatch.setattr(storage,"_mounts",lambda: facts)
    monkeypatch.setattr(storage,"_free",lambda path: 1024**3)
    limits = HPOStorageConfig(parent_root=str(controls),numeric_root=str(controls),
        optuna_path=str(controls / "optuna.sqlite3"),originals_root=str(originals),windows_c_root=str(windows),
        destination_capacity_bytes=1024**3,admission_reservation_bytes=1024,windows_c_min_free_bytes=1024)
    return limits,controls,windows,originals


def test_storage_probe_records_actual_volumes_and_usage_before_creation(tmp_path,monkeypatch):
    limits,controls,windows,originals = admitted_storage(tmp_path,monkeypatch)
    (controls / "existing").write_bytes(b"a" * 17)
    receipt = check_storage(limits,campaign_dir=controls / "campaign",originals_root=originals,deadline=float("inf"))
    assert receipt["windows_c"]["drive"] == "C"
    assert receipt["destinations"][0]["used_bytes"] == 17
    assert not (controls / "campaign").exists()
    with pytest.raises(ContractError):
        check_storage(replace(limits,windows_c_root=str(originals)),campaign_dir=controls / "campaign",originals_root=originals,deadline=float("inf"))
    with pytest.raises(ContractError):
        check_storage(replace(limits,numeric_root=str(windows)),campaign_dir=controls / "campaign",originals_root=originals,deadline=float("inf"))


def test_actual_c_volume_missing_or_unknown_never_falls_back_to_parent_free_space(tmp_path,monkeypatch):
    limits,controls,windows,originals = admitted_storage(tmp_path,monkeypatch)
    with pytest.raises(ContractError):
        check_storage(replace(limits,windows_c_root=str(windows / "missing")),campaign_dir=controls / "campaign",originals_root=originals,deadline=float("inf"))
    with pytest.raises(ContractError):
        check_storage(replace(limits,windows_c_root=str(controls)),campaign_dir=controls / "campaign",originals_root=originals,deadline=float("inf"))


def test_low_capacity_and_free_space_refuse_without_directory_or_process_work(tmp_path,monkeypatch):
    from gf_cli import hpo_storage as storage

    limits,controls,_,originals = admitted_storage(tmp_path,monkeypatch)
    (controls / "data").write_bytes(b"a" * 100)
    with pytest.raises(ContractError):
        check_storage(replace(limits,destination_capacity_bytes=110),campaign_dir=controls / "campaign",originals_root=originals,deadline=float("inf"))
    monkeypatch.setattr(storage,"_free",lambda path: 1)
    with pytest.raises(ContractError):
        check_storage(limits,campaign_dir=controls / "campaign",originals_root=originals,deadline=float("inf"))
    assert not (controls / "campaign").exists()
