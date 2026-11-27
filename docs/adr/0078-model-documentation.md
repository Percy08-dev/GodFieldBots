# 0078: Model catalog and generated network diagrams

Status: Accepted. The architecture review approved the design before implementation.

## Context

`model_id` identifies a model family. Structured models in that family also
have an input contract, a model version, and configurable encoders and heads.
Readers need a stable name that identifies the structure they are comparing
and a way to inspect the shapes used by a representative execution.

## Decision

Maintain a [model catalog](../models/README.md) with one Markdown page per
documented structure. The initial catalog covers 12 neural configurations,
four LightGBM configurations, and three rule or sampling policies. Catalog
names are documentation identifiers. Runtime model and artifact identities
come from the existing configuration and manifest contracts.

Use Mermaid for the model data flow. `tools/model_docs.py` generates a
checked-in torchview SVG for each neural configuration. It instantiates the
existing factory on CPU with a fixed seed and synthetic operands. Model1
uses its normal forward. Model2 through Model4 use existing packed tensor
operations with a documentation-only adapter that exposes tensor operands to
torchview. The MLP diagram uses the explicit indexed frame path and the
shared heads. The GRU and Transformer diagrams use the packed forward.
The pages identify the path and representative shapes. The adapter's log
probabilities and value are checked against the ordinary forward on the
same inputs.

The command validates the index, pages and generated diagram fingerprints.
Each fingerprint covers the graph source and its model/profile identity.
Graphviz's layout output is an SVG for reading and can vary with the drawing
environment. The Python packages for drawing are isolated in an opt-in uv
group. Graphviz `dot` is used for rendering. A failed check reports the
affected entry and how to regenerate its diagram. This workflow uses no
checkpoint, collected match, simulator process or GPU.

## Alternatives and consequences

A single hand-maintained description becomes difficult to navigate as models
gain independent encoder and scorer options. Generated graphs alone make it
difficult to see a model's policy-input boundary and compatibility rules.
Keeping the overview and the generated graph together gives readers those
details and the representative tensor shapes. The repository carries the SVG
files so the diagrams are visible without an installed rendering environment.

## Verification

Check the full catalog, successful graph generation, repeatable fingerprints,
packed and ordinary forward agreement, and diagnostics for invalid input or
missing and stale artifacts. The checks exercise synthetic CPU inputs.

The catalog check passed for all 19 pages and 12 diagrams. All 12 SVGs
matched byte-for-byte when regenerated from the committed model code.
The model-docs, structured-network, and structured-config test selection
passed 107 cases, including packed and ordinary forward agreement for all
nine structured neural models. The independent correctness and architecture
reviews accepted the implementation.
