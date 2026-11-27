from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/model_docs.py"
NEURAL_SLUGS = (
    "mlp-v1",
    "gru-v1",
    "transformer-v1",
    "mlp-v2-event-gru",
    "gru-v2-event-gru",
    "transformer-v2-event-gru",
    "mlp-v3-event-gru",
    "gru-v3-event-gru",
    "transformer-v3-event-gru",
    "mlp-v4-event-gru",
    "gru-v4-event-gru",
    "transformer-v4-event-gru",
)
OTHER_SLUGS = (
    "lightgbm-v1-observation",
    "lightgbm-v1-history",
    "lightgbm-v2",
    "lightgbm-v3",
    "random-v1",
    "heuristic-v1",
    "heuristic-refresh-v1",
)
SVG_GROUP = "{http://www.w3.org/2000/svg}g"
SVG_METADATA = "{http://www.w3.org/2000/svg}metadata"


def run_script(*args: str, no_site: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *(["-S"] if no_site else []), str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(("slug", "reason"), [("made-up", "unknown"), ("random-v1", "neural")])
def test_generate_rejects_invalid_slugs_before_loading_optional_packages(
    slug: str, reason: str
) -> None:
    result = run_script("generate", slug, no_site=True)
    assert result.returncode != 0
    assert slug in result.stderr
    assert reason in result.stderr.lower()


def test_help_works_without_optional_packages() -> None:
    result = run_script("--help", no_site=True)
    assert result.returncode == 0, result.stderr
    assert "generate" in result.stdout
    assert "check" in result.stdout


def test_generate_explains_how_to_install_missing_python_packages() -> None:
    result = run_script("generate", "mlp-v1", no_site=True)
    assert result.returncode != 0
    assert "torch" in result.stderr
    assert "uv sync" in result.stderr
    assert "--group model-docs --extra torch" in result.stderr
    assert "Traceback" not in result.stderr


def test_generate_explains_how_to_install_dot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tools import model_docs

    for name in ("torch", "torchview", "graphviz"):
        pytest.importorskip(name)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert model_docs.main(["generate", "mlp-v1"], catalog_dir=tmp_path) != 0
    result = capsys.readouterr()
    assert "Graphviz" in result.err
    assert "`dot`" in result.err
    assert "install" in result.err.lower()
    assert not (tmp_path / "mlp-v1.svg").exists()


@pytest.fixture
def catalog(tmp_path: Path) -> Path:
    slugs = NEURAL_SLUGS + OTHER_SLUGS
    (tmp_path / "README.md").write_text(
        "\n".join(f"- [{slug}]({slug}.md)" for slug in slugs), encoding="utf-8"
    )
    for slug in slugs:
        diagram = f"![Architecture]({slug}.svg)\n" if slug in NEURAL_SLUGS else ""
        (tmp_path / f"{slug}.md").write_text(f"# {slug}\n\n{diagram}", encoding="utf-8")
    return tmp_path


def test_check_reports_broken_index_missing_page_and_missing_embed(
    catalog: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tools import model_docs

    index = catalog / "README.md"
    index.write_text(
        index.read_text(encoding="utf-8").replace("(random-v1.md)", "(bad.md)"),
        encoding="utf-8",
    )
    (catalog / "heuristic-v1.md").unlink()
    (catalog / "mlp-v1.md").write_text("# MLP\n\n[Diagram](mlp-v1.svg)\n", encoding="utf-8")

    assert model_docs.main(["check"], catalog_dir=catalog) != 0
    errors = capsys.readouterr().err.splitlines()
    assert any("README.md" in line and "random-v1.md" in line for line in errors)
    assert any("heuristic-v1.md" in line and "missing" in line for line in errors)
    assert any("mlp-v1.md" in line and "embed" in line for line in errors)


@pytest.mark.parametrize(
    "hidden",
    [
        "<!-- {} -->",
        "<!--\n{}\n-->",
        "~~~markdown\n<!--\n{}\n~~~",
        "````markdown\n```\n{}\n`````",
    ],
)
def test_check_ignores_links_and_images_in_markdown_examples(
    catalog: Path, capsys: pytest.CaptureFixture[str], hidden: str
) -> None:
    from tools import model_docs

    index = catalog / "README.md"
    link = "- [random-v1](random-v1.md)"
    index.write_text(
        index.read_text(encoding="utf-8").replace(link, "\n" + hidden.format(link) + "\n"),
        encoding="utf-8",
    )
    pages = catalog / "mlp-v1.md", catalog / "gru-v1.md"
    for page in pages:
        image = f"![Architecture]({page.stem}.svg)"
        source = f"# Model\n\n{hidden.format(image)}\n"
        if page.stem == "gru-v1":
            source += f"\n{image}\n"
        page.write_text(source, encoding="utf-8")

    assert model_docs.main(["check"], catalog_dir=catalog) != 0
    errors = capsys.readouterr().err.splitlines()
    assert any("README.md" in line and "random-v1.md" in line for line in errors)
    assert any("mlp-v1.md" in line and "embed" in line for line in errors)
    assert all("heuristic-v1.md" not in line for line in errors)
    assert all("gru-v1.md" not in line for line in errors)


@pytest.mark.parametrize(
    ("svg", "reason"),
    [
        (None, "missing"),
        ("<svg", "invalid"),
        ('<?xml version="1.0" encoding="fictional"?><svg/>', "invalid"),
        ('<?xml version="1.0" encoding="utf-7"?><svg/>', "invalid"),
        ('<svg xmlns="http://www.w3.org/2000/svg"/>', "empty"),
    ],
)
def test_check_identifies_unusable_svg_with_regeneration_command(
    catalog: Path, capsys: pytest.CaptureFixture[str], svg: str | None, reason: str
) -> None:
    from tools import model_docs

    path = catalog / "gru-v2-event-gru.svg"
    if svg is not None:
        path.write_text(svg, encoding="utf-8")
    assert model_docs.main(["check"], catalog_dir=catalog) != 0
    errors = capsys.readouterr().err.splitlines()
    assert any(path.name in line and reason in line.lower() for line in errors)
    assert "generate gru-v2-event-gru" in "\n".join(errors)


@pytest.fixture
def graph_tools() -> None:
    for name in ("torch", "torchview", "graphviz"):
        pytest.importorskip(name)
    if shutil.which("dot") is None:
        pytest.skip("Graphviz dot is needed to test SVG rendering")


def test_generate_writes_only_requested_diagrams(catalog: Path, graph_tools: None) -> None:
    from tools import model_docs

    assert model_docs.main(["generate", "mlp-v1"], catalog_dir=catalog) == 0
    assert {path.name for path in catalog.glob("*.svg")} == {"mlp-v1.svg"}


def test_all_diagrams_render_and_check_without_dot(
    catalog: Path,
    graph_tools: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tools import model_docs

    assert model_docs.main(["generate"], catalog_dir=catalog) == 0
    assert {path.stem for path in catalog.glob("*.svg")} == set(NEURAL_SLUGS)
    for slug in NEURAL_SLUGS:
        svg = ElementTree.parse(catalog / f"{slug}.svg")
        assert any(group.get("class") == "node" for group in svg.iter(SVG_GROUP))
        fingerprint = svg.find(SVG_METADATA)
        assert fingerprint is not None
        assert fingerprint.text is not None and re.fullmatch("[0-9a-f]{64}", fingerprint.text)

    monkeypatch.setenv("PATH", str(catalog))
    assert model_docs.main(["check"], catalog_dir=catalog) == 0, capsys.readouterr().err

    path = catalog / "gru-v3-event-gru.svg"
    svg_text = path.read_text(encoding="utf-8")
    fingerprint = ElementTree.fromstring(svg_text).find(SVG_METADATA)
    assert fingerprint is not None and fingerprint.text is not None
    stale = "0" * 64 if fingerprint.text != "0" * 64 else "1" * 64
    path.write_text(svg_text.replace(fingerprint.text, stale, 1), encoding="utf-8")
    assert model_docs.main(["check"], catalog_dir=catalog) != 0
    err = capsys.readouterr().err
    assert f"{path.name} is stale" in err
    assert "generate gru-v3-event-gru" in err


@pytest.mark.parametrize("slug", NEURAL_SLUGS[3:])
def test_structured_trace_adapter_agrees_with_public_forward(
    slug: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    from tools import model_docs

    trace = model_docs._structured_trace(torch, model_docs.NEURAL_MODELS[slug])
    network = trace.module.network
    length = 1 if slug.startswith("mlp-") else 2
    if length == 1:

        def unavailable(*_args: object) -> None:
            pytest.fail("The structured MLP trace should use the indexed CPU path explicitly")

        monkeypatch.setattr(network, "_learning_forward", unavailable)
    frames, candidates = model_docs._structured_inputs(network.layout, network.config, length)
    with torch.no_grad():
        packed_log_probs, packed_value = trace.module(*trace.inputs)
        log_probs, value, _ = network(frames, candidates)
    assert trace.inputs[0]["event_lengths"].tolist() == [2] * length
    assert trace.inputs[0]["card_mask"].sum(dim=-1).tolist() == [2] * length
    assert packed_log_probs.shape == (1, 3)
    assert packed_value.shape == (1,)
    torch.testing.assert_close(packed_log_probs[0], log_probs)
    torch.testing.assert_close(packed_value[0], value)
