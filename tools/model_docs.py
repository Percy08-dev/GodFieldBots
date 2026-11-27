"""Render and verify the CPU-only architecture diagrams in docs/models."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

if TYPE_CHECKING:
    from gf_common.structured_layout import StructuredLayout
    from gf_common.structured_tokens import CandidateTokens, FrameTokens
    from gf_models.structured_config import ModelConfigV2, ModelConfigV3, ModelConfigV4

CATALOG_DIR = Path(__file__).resolve().parents[1] / "docs" / "models"
NEURAL_MODELS = {
    f"{kind}-v{version}" + ("-event-gru" if version > 1 else ""): (kind, version)
    for version in range(1, 5)
    for kind in ("mlp", "gru", "transformer")
}
OTHER_MODELS = (
    "lightgbm-v1-observation",
    "lightgbm-v1-history",
    "lightgbm-v2",
    "lightgbm-v3",
    "random-v1",
    "heuristic-v1",
    "heuristic-refresh-v1",
)
SEED = 0
SVG = "{http://www.w3.org/2000/svg}"
FINGERPRINT_ID = "godfieldbots-model-fingerprint"
INSTALL_PYTHON = "uv sync --group model-docs --extra torch"
RUN_TOOL = "uv run --locked --group model-docs --extra torch python tools/model_docs.py"


class ModelDocsError(Exception):
    """A catalog, tracing, or rendering failure to report to the caller."""


@dataclass(frozen=True)
class Trace:
    module: Any
    inputs: tuple[Any, ...]
    profile: dict[str, Any]


def _v1_trace(torch: Any, kind: str) -> Trace:
    from gf_models.config import GRUModelConfig, MLPModelConfig, TransformerModelConfig
    from gf_models.encoder import feature_names, signature
    from gf_models.features import critic_indices

    config = {
        "mlp": MLPModelConfig,
        "gru": GRUModelConfig,
        "transformer": TransformerModelConfig,
    }[kind]()
    network = import_module(f"gf_models.{kind}.model").make_network(config, SEED)
    d = len(critic_indices(config.encoder))
    a = len(feature_names(config.encoder))
    state_shape = (1, d) if kind == "mlp" else (1, 2, d)
    actor = torch.zeros((1, 3, a), dtype=torch.float32, device="cpu")
    state = torch.zeros(state_shape, dtype=torch.float32, device="cpu")
    mask = torch.ones((1, 3), dtype=torch.bool, device="cpu")
    inputs = (actor, state, mask) if kind == "mlp" else (state, actor, mask)
    return Trace(
        network.eval(),
        inputs,
        {
            "path": "forward",
            "architecture": asdict(config),
            "encoder_signature": signature(config.encoder),
            "tensors": {
                "actor": _tensor_profile(actor),
                "state": _tensor_profile(state),
                "mask": _tensor_profile(mask),
            },
        },
    )


def _structured_inputs(
    layout: StructuredLayout,
    config: ModelConfigV2 | ModelConfigV3 | ModelConfigV4,
    length: int,
) -> tuple[list[FrameTokens], CandidateTokens]:
    from gf_common.structured_tokens import MISSING, PAD, CandidateTokens, FrameTokens, category
    from gf_models.structured_config import recorded_token_signature

    slots, occupied, interval = 18, 2, 2
    card_width = len(layout.card_numeric_fields)
    card_categories = len(layout.card_categories)
    event_width = len(layout.event_numeric_fields)
    state_width = len(layout.state_numeric_fields)
    recipe = recorded_token_signature(config, layout=layout)
    # Real layout dimensions, synthetic field values. Occupied rows use the
    # layout's missing categories and unoccupied slots use padding.
    frames = [
        FrameTokens(
            frame_hash=f"synthetic-frame-{index}",
            recipe_signature=recipe,
            aggregate_window=config.event.aggregate_window,
            actor=1,
            episode="synthetic-model-docs",
            revision=index,
            event_start=index * interval,
            event_end=(index + 1) * interval,
            state_numerics=(0,) * state_width,
            state_present=(True,) * state_width,
            state_categories=(MISSING,) * len(layout.state_categories),
            state_category_present=(False,) * len(layout.state_categories),
            card_numerics=((0,) * card_width,) * slots,
            card_present=((False,) * card_width,) * slots,
            card_categories=(
                ((MISSING,) * card_categories,) * occupied
                + ((PAD,) * card_categories,) * (slots - occupied)
            ),
            card_mask=(True,) * occupied + (False,) * (slots - occupied),
            opponent_learned=(PAD,) * slots,
            interval_numerics=((0,) * event_width,) * interval,
            interval_present=((False,) * event_width,) * interval,
            interval_categories=((MISSING,) * len(layout.event_categories),) * interval,
            history=(0,) * layout.history_width,
        )
        for index in range(length)
    ]
    candidates = CandidateTokens(
        table_hash="synthetic-candidates",
        frame_hash=frames[-1].frame_hash,
        recipe_signature=recipe,
        kinds=tuple(
            category(kind, layout.action_kinds)
            for kind in ("pray", "play_sequence", "play_sequence")
        ),
        targets=tuple(
            category(target, layout.targets) for target in ("omitted", "opponent", "self")
        ),
        numerics=((0, 0, 0, 0),) * 3,
        present=((False, False, False, False),) * 3,
        offsets=(0, 0, 1, 3),
        slots=(0, 1, 0),
    )
    return frames, candidates


def _structured_trace(torch: Any, entry: tuple[str, int]) -> Trace:
    from gf_common.structured_layout import CORRECTED_CARD_INSTANCE_LAYOUT, LEGACY_LAYOUT
    from gf_models.structured_binding import model_defaults_v2, model_defaults_v3, model_defaults_v4
    from gf_models.structured_learning import pack_window
    from gf_models.structured_network import make_network

    kind, version = entry
    layout = LEGACY_LAYOUT if version == 2 else CORRECTED_CARD_INSTANCE_LAYOUT
    config: ModelConfigV2 | ModelConfigV3 | ModelConfigV4
    if version == 2:
        config = model_defaults_v2(kind).architecture()
    elif version == 3:
        config = model_defaults_v3(kind).architecture(layout=layout)
    else:
        config = model_defaults_v4(kind).architecture(layout=layout)
    network = make_network(config, SEED, layout=layout)
    length = 1 if kind == "mlp" else 2
    frames, candidates = _structured_inputs(layout, config, length)
    window = pack_window(network, frames, [candidates], [(0, 0.0, 0.0, 0.0, 0.0)])

    base: Any = torch.nn.Module

    class PackedForward(base):
        def __init__(self) -> None:
            super().__init__()
            self.network = network

        def forward(self, operands: dict[str, Any]) -> tuple[Any, Any]:
            # Keep the packer's metadata and substitute the traced tensor views.
            packed = replace(window, operands=operands)
            if kind == "mlp":
                # Trace the indexed CPU path explicitly across dispatch optimizations.
                context, cards = self.network._learning_frames(packed, [0])
                return self.network._learning_heads(packed.operands, context, cards)
            return self.network._learning_forward(packed, [length], length)

    return Trace(
        PackedForward().eval(),
        (window.operands,),
        {
            "path": "_learning_frames -> _learning_heads" if kind == "mlp" else "_learning_forward",
            "architecture": asdict(config),
            "token_signature": candidates.recipe_signature,
            "frames": length,
            "event_lengths": window.event_lengths,
            "occupied_cards": sum(frames[-1].card_mask),
            "tensors": {name: _tensor_profile(tensor) for name, tensor in window.operands.items()},
        },
    )


def _tensor_profile(tensor: Any) -> dict[str, Any]:
    return {"shape": tuple(tensor.shape), "dtype": str(tensor.dtype)}


def _load_graph_tools() -> tuple[Any, Any]:
    packages = {}
    for name in ("torch", "graphviz", "torchview"):
        try:
            packages[name] = import_module(name)
        except ImportError as exc:
            raise ModelDocsError(
                f"Cannot import model-docs package {name!r} ({exc}). Run `{INSTALL_PYTHON}`."
            ) from exc
    return packages["torch"], packages["torchview"].draw_graph


def _diagram(slug: str, torch: Any, draw_graph: Any) -> tuple[Any, str]:
    kind, version = NEURAL_MODELS[slug]
    settings = {
        "device": "cpu",
        "mode": "eval",
        "depth": 3,
        "expand_nested": False,
        "roll": True,
    }
    try:
        trace = (
            _v1_trace(torch, kind) if version == 1 else _structured_trace(torch, (kind, version))
        )
        graph = draw_graph(
            trace.module, input_data=list(trace.inputs), graph_name=slug, **settings
        ).visual_graph
    except (ImportError, RuntimeError, ValueError) as exc:
        detail = exc.__cause__ or exc
        raise ModelDocsError(f"{slug}: unable to trace the network ({detail}).") from exc
    profile = {
        "slug": slug,
        "seed": SEED,
        "settings": settings,
        **trace.profile,
        "dot": graph.source,
    }
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    return graph, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _svg_fingerprint(root: ElementTree.Element) -> str:
    if root.tag != SVG + "svg":
        raise ModelDocsError("is invalid (expected an SVG document)")
    if not any(element.get("class") == "node" for element in root.iter(SVG + "g")):
        raise ModelDocsError("is empty (no graph nodes)")
    fingerprint = root.findtext(f"{SVG}metadata[@id='{FINGERPRINT_ID}']")
    if fingerprint is None or re.fullmatch("[0-9a-f]{64}", fingerprint) is None:
        raise ModelDocsError("is stale (fingerprint missing or invalid)")
    return fingerprint


def _render(graph: Any, fingerprint: str) -> bytes:
    output = graph.pipe(format="svg")
    # Add metadata while retaining Graphviz's XML header and SVG serialization.
    start, closing, suffix = output.rpartition(b"</svg>")
    metadata = f'<metadata id="{FINGERPRINT_ID}">{fingerprint}</metadata>\n'.encode()
    svg = start + metadata + closing + suffix
    _svg_fingerprint(ElementTree.fromstring(svg))
    return svg


def _generate(slugs: Sequence[str], directory: Path) -> None:
    for slug in slugs:
        if slug in OTHER_MODELS:
            raise ModelDocsError(f"{slug!r} is a non-neural entry and has no generated SVG.")
        if slug not in NEURAL_MODELS:
            raise ModelDocsError(
                f"Unknown model slug {slug!r}. Neural entries: {', '.join(NEURAL_MODELS)}."
            )
    torch, draw_graph = _load_graph_tools()
    if shutil.which("dot") is None:
        raise ModelDocsError("Graphviz `dot` is missing. Install Graphviz and add `dot` to PATH.")
    directory.mkdir(parents=True, exist_ok=True)
    for slug in dict.fromkeys(slugs):
        graph, fingerprint = _diagram(slug, torch, draw_graph)
        path = directory / f"{slug}.svg"
        try:
            path.write_bytes(_render(graph, fingerprint))
        except (
            OSError,
            subprocess.CalledProcessError,
            ElementTree.ParseError,
            ModelDocsError,
        ) as exc:
            raise ModelDocsError(f"Unable to write {path}: {exc}") from exc
        print(f"Generated {path}")


def _visible_markdown(source: str) -> str:
    """Ignore catalog comments and fenced examples before matching inline links."""
    fence_pattern = re.compile(r" {0,3}(`{3,}|~{3,})(.*)")
    fence = ""
    in_comment = False
    visible = []
    for line in source.splitlines():
        marker = fence_pattern.fullmatch(line)
        if fence:
            if marker and marker[1].startswith(fence) and not marker[2].strip():
                fence = ""
            continue
        if not in_comment and marker and (marker[1][0] == "~" or "`" not in marker[2]):
            fence = marker[1]
            continue

        parts = []
        while line:
            if in_comment:
                _, closing, line = line.partition("-->")
                in_comment = not closing
            else:
                text, opening, line = line.partition("<!--")
                parts.append(text)
                in_comment = bool(opening)
        visible.append(" ".join(parts))
    return "\n".join(visible)


def _markdown_link(source: str, target: str, *, image: bool = False) -> bool:
    prefix = "!" if image else "(?<!!)"
    return re.search(rf"{prefix}\[[^\]\n]*\]\((?:\./)?{re.escape(target)}\)", source) is not None


def _regenerate(slug: str) -> str:
    return f"Run `{RUN_TOOL} generate {slug}`."


def _check(directory: Path) -> list[str]:
    errors: list[str] = []
    index_path = directory / "README.md"
    index = None
    try:
        index = _visible_markdown(index_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"{index_path} is missing. Add the model catalog index.")

    fingerprints = {}
    for slug in (*NEURAL_MODELS, *OTHER_MODELS):
        page = directory / f"{slug}.md"
        if index is not None and not _markdown_link(index, page.name):
            errors.append(f"{index_path} is missing a link to {page.name}.")
        source = None
        try:
            source = page.read_text(encoding="utf-8")
        except FileNotFoundError:
            errors.append(f"{page} is missing.")
        if slug not in NEURAL_MODELS:
            continue
        svg = directory / f"{slug}.svg"
        if source is not None and not _markdown_link(
            _visible_markdown(source), svg.name, image=True
        ):
            errors.append(f"{page} must embed {svg.name} with a relative Markdown image link.")
        try:
            fingerprints[slug] = _svg_fingerprint(ElementTree.parse(svg).getroot())
        except FileNotFoundError:
            errors.append(f"{svg} is missing. {_regenerate(slug)}")
        except (ElementTree.ParseError, LookupError, ValueError) as exc:
            errors.append(f"{svg} is invalid ({exc}). {_regenerate(slug)}")
        except ModelDocsError as exc:
            errors.append(f"{svg} {exc}. {_regenerate(slug)}")

    if fingerprints:
        try:
            torch, draw_graph = _load_graph_tools()
        except ModelDocsError as exc:
            return [*errors, str(exc)]
        for slug, fingerprint in fingerprints.items():
            try:
                _, expected = _diagram(slug, torch, draw_graph)
            except ModelDocsError as exc:
                errors.append(str(exc))
                continue
            if fingerprint != expected:
                svg = directory / f"{slug}.svg"
                errors.append(f"{svg} is stale (fingerprint mismatch). {_regenerate(slug)}")
    return errors


def main(argv: Sequence[str] | None = None, *, catalog_dir: Path = CATALOG_DIR) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="render neural diagrams with Graphviz")
    generate.add_argument("slugs", metavar="slug", nargs="*", help="neural slugs (default: all)")
    commands.add_parser("check", help="verify model pages and current neural diagrams")
    args = parser.parse_args(argv)

    try:
        if args.command == "generate":
            _generate(args.slugs or tuple(NEURAL_MODELS), catalog_dir)
            return 0
        errors = _check(catalog_dir)
    except (ModelDocsError, OSError, UnicodeError) as exc:
        errors = [str(exc)]

    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(
        f"Checked {len(NEURAL_MODELS) + len(OTHER_MODELS)} model pages and "
        f"{len(NEURAL_MODELS)} neural diagrams."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
