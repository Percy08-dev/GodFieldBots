"""Synthetic collector integration identities; never executable Simulator code."""
from pathlib import Path


def synthetic_server_identity(tmp_path: Path) -> dict[str, str | None]:
    """Keep synthetic coordinators independent of mutable external source/docs."""
    binary = tmp_path / 'synthetic-simulator.identity'
    binary.write_bytes(b'synthetic collector identity only; not an executable engine')
    return {'source_dir': None, 'binary': str(binary)}
