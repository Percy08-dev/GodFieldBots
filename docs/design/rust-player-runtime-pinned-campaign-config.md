# Reconstruct campaign CONFIG from its pinned canonical snapshot

Status: accepted bounded parser/I-O seam; independent architecture READY on
proposal0a1cb8468d226f4cb0b9249cfae7604e1d170dc3ddeb6a8498ef56081df3f6d4. Owner: root. Applies the already accepted loaded-campaign1
bootstrap record without changing its fields or adding a live capability.

The existing campaign_config parser unconditionally calls load_config on the
original base TOML. That is correct for ordinary initial loading, but a restarted
control host must consume the canonical snapshot already pinned by its actual
HostControl/bootstrap, without late unpinned source-file reads. A normalized
campaign asdict also contains derived base_config_sha256 and normalized search
objects, so passing it directly as raw TOML data is not supported.

Extend ONLY campaign_config with a private optional keyword for an explicitly
supplied LoadedConfig base (default None). With no supplied base, preserve the
existing load_config path, call ordering, accepted schemas/defaults/unknown-key
checks and all legacy behavior exactly. With supplied base, require the exact
LoadedConfig DATA type and its canonical asdict/effective/digest invariant, then
run all existing base compatibility checks. This is CONFIG injection, not a
run/Source/HostControl/admission or checkpoint object. No new parser framework.

The new CLI snapshot decoder first applies existing validate_loaded_campaign
and reconstructs base_effective with existing _normalized_config, with exact
canonical asdict equality and base SHA. It converts ONLY normalized search
representation into the already supported raw search_space input keys; removes
derived base_config_sha256 only for invoking the existing campaign parser with
the pinned base; and requires FULL final canonical campaign asdict equality,
base SHA and existing combined campaign SHA. Every dropped derived/default/
normalized field must therefore be reproduced exactly; unknown/missing/tampered
fields refuse, rather than being silently ignored. Existing model, recipe,
operating, runtime and search validation retain ownership of their checks.

Provenance: loaded-campaign1 stores one full original input-receipt sequence,
not separate base inputs. Preserve that full sequence and ordering in the final
LoadedHPOConfig.inputs and in the reconstructed base's provenance-only DATA.
Do not claim to have recovered an original base-only subset or infer it from a
filename. Current campaign_config returns base.inputs and load_hpo_config then
appends the campaign receipt; materialize_hpo_run consumes loaded_campaign.inputs.
The decoder sets final inputs to the recorded full sequence without appending
new or duplicated receipts. Inputs never grant live authority; no schema change.

Dependency direction stays CLI config-to-common/model/training pure CONFIG
helpers as already established. Training never imports CLI. The actual caller
obtains snapshot bytes through existing owned Store/HostControl checks; this
standalone pure decoder can return typed CONFIG DATA only. It performs no
Torch/device/game/native-control work and never reads original TOML contents.

Ownership extension only src/gf_cli/hpo_config.py plus focused config/snapshot
tests, with CLI decoder in the already assigned campaign/entry files. Verify
roundtrip after removing/changing temporary original TOMLs, exact full provenance,
wrong base/config/derived-search/default/hash/unknown-key refusal, explicit timed
native6/mixed cases, and unchanged default legacy file-loading path. No mocks
of actual Source/Parent/HostControl authority, no tests/process/GPU budgets here.
