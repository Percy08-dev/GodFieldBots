# 0058 Four pytest workers with file scheduling

Status — accepted.

## Context

The matched CPU probe covered eight files with 210 passes and 12 skips
under every worker count. Sequential wall time was 127.96 s. Two four-worker
runs had median wall 46.31 s and median sampled process-tree peak PSS 4.41 GiB.
Two eight-worker runs had 48.59 s and 5.40 GiB. The longest file occupied
roughly 43 to 45 s. The user selected four workers.

## Decision

Use `pytest-xdist==3.8.0` from the development dependency group and lock its
resolved execnet version. Pytest reads `-n 4 --dist=loadfile` from the project
configuration. Each file's tests execute on the same worker. Session fixtures
keep private resources per worker under
[ADR0025](0025-session-shared-pytest-ppo-sources.md), including the existing
fresh CPU subprocess template ownership. Use `uv run pytest -n 0` for GPU,
timing or other checks that require serial scheduling.

A wide preflight found a time-generated UUID v1 in a parameterized HTTP test.
A fixed canonical UUID v1 makes the node ID identical across workers. The case
still exercises rejection of a UUID with the wrong version.

## Alternatives and consequences

Sequential execution uses less peak RAM. Eight workers raised peak memory
and increased wall time in the eight-file probe. File-based scheduling
preserves setup reuse within each file. Session setup may repeat on separate
workers, and file duration constrains balancing. Revisit the width or
scheduler when the suite or available build resources change.

## Verification

The locked CPython 3.14.10 environment uses pytest 9.1.1, pytest-xdist 3.8.0
and execnet 2.1.2. Default and explicit serial HTTP runs passed 23 cases each.
An independent verbose check observed four workers and LoadFileScheduling.
The serial actual CUDA inference and learning run passed 145 cases. The
lock check and Ruff passed.

The broad run used the default worker settings on commit `077f452`, with
CUDA hidden, two OMP threads and one BLAS thread in the test environment.
It passed 8,942 cases with 585 skips in 651.16 s outer wall on the 16-logical-CPU
host. Sampled peak process-tree PSS was 6.88 GiB. The verification command
explicitly excluded the same 30 cases that depend on unavailable archived
Tree artifacts as the prior integration check. The eight-file probe supplies
the matched speed comparison.

The primary checkout retains the measurement inputs and outputs under
`runs/ordinary-rust-ppo-update/pytest-parallel-bbf5e8f` and
`runs/ordinary-rust-ppo-update/pytest-default-four-bbf5e8f`.
