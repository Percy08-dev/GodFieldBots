# 0006: Preserve the existing public history information in structured inputs

Status: correction design READY; implementation/reverification pending (2026-11-02).

The implementation plan requires the current interval event GRU and the existing
recent128-event aggregate. A late root audit finds that the initial structured
62-field aggregate loses information present in the unchanged v1 reference
(`gf_models/encoder.py:184–196`): event definition-ID counts, window truncation and
prioritized-actor-specific numeric sums/counts. Role marginals and global numeric
sums cannot recover the latter. Raw persisted events and current-interval ID
tokens do not restore older-interval features consumed by MLP or Tree2.

The root-requested verifier confirms the exact field difference. The advisor
recommends a mandatory correction: retain the62fields and append exact definition
counts and the old window/actor statistics, without32-bucket policy features.
See [ADR0023§24](../adr/0023-structured-encoders-and-diverse-search.md#24-public-history-preservation-correction)
for the chosen two-descriptor compatibility boundary and required checks.

Existing lifecycle, generation safety, proposal and raw-storage results retain
their original scopes. Numerical/performance results for62fields cannot qualify
the corrected model. No real model2 game, fit or qualified Source exists to
reinterpret. Old model1/data meanings remain unchanged. Preserve actual existing
component artifacts for historical inspection; active inference/training/resume
must reject the incomplete recipe. Do not solve this omission by reducing the
plan's input or verification requirements.
