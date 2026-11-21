# 0051: Estimate the CUDA cost of a resident inference participant by role

Status: accepted design for the existing homogeneous Model4 HPO training pool.
This refines the active actor part of ADR0050. See [the design](../design/ppo-shared-inference.md)
and [calibration](../verification/ppo-shared-inference.md).

A normal participant scores one current decision under no_grad. Its GPU
reservation previously included a mixed host/learner envelope and inactive
temporal tensors. In the new derived hpo-resident-actor-envelope-v3 recipe,
the active actor owns one weight set and position buffer. The temporal state
is empty for MLP, a layered hidden vector for GRU, or two prefix-capacity
buffers for Transformer. Named maxima for one decision cover frame and hand
encoding, interval GRU, temporal step, full candidate features, chunked
scoring, global normalization and return. Native context/workspace remain
the explicitly configured allowances; these estimates are not hard device
limits. Cache adds (K−1) copies of weights **and** position buffer on CUDA;
the separate host reservation still follows ADR0050. Concurrent BEGIN and
serial SYNC retain their existing reserved owners.

Use this amount in resident training preflight, CLI reporting, actual pool
leases and per-game admission after the original two role bindings have
been validated. The collector uses the actual pool and learner and therefore
includes any admitted experimental extras. Changing the shared NumericCosts
or scalar input guards would alter search candidates. Changing the evaluation
pool would require accounting for its separate set of frozen identities.
Keep both boundaries intact. Historical evidence is not recosted.

Revisit the configured native reserves with evidence on a changed platform,
backend or new inference implementation; do not use a single measured
allocator peak as a universal upper bound. This decision does not authorize
a multi-participant inference service.
