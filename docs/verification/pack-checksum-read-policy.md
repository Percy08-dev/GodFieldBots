# Pack checksum read policy verification

Status: implementation pending. Design: [ADR0035](../adr/0035-pack-checksum-read-policy.md)
and [accepted specification](../design/pack-checksum-read-policy.md).

## Scope and sources

Baseline Bots `f59a2abc360e7e3d587841edcd55ecd839ab3629`.
Simulator remains read-only at `91d5926a7572db327eaea3751e7c24f428120996`;
its current 07§21.1 was read, with no integration/policy changes proposed.
CPython3.14.10 GIL-enabled, SQLite3.54.1. Persisted SQLite1/Pack1/descriptor1,
proof2/Run4/Update4/Dataset4/provider1/plan4/wire5 remain unchanged.
No formal campaign, HPO, final-test, qualified real-data fit, migration or deletion.

## Design gate

Independent `bot_architecture_reviewer` READY before implementation on2026-11-11;
no blocking findings. Final correctness/conformance remain pending.

## Baseline checks

PASS: `.venv/bin/python -m pytest -q tests/test_group_pack_storage.py tests/test_group_pack_game_source.py tests/test_group_pack_dataset_contract.py`
35tests in8.26s, before code changes.

## Pending acceptance

Reader/audit/custody/native/Dataset tests, source-matched native build and checks,
old/current same-original HDD measurement, bounded two-update pause/resume,
controlled Dataset/Tree and independent final reviews remain NOT RUN.
There are no eligible genuine formal parents/public final64 sources for a
qualified Dataset4 fit; that scope remains NOT RUN. Existing DrvFS compound
cleanup uncertainty is not treated as healthy retirement.
