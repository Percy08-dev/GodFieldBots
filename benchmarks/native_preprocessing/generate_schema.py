"""Compatibility launcher for the native-owned typed Journal schema generator."""
import runpy
from pathlib import Path

_GENERATOR = Path(__file__).resolve().parents[2] / 'native/src/typed_journal/dto/generate_schema.py'
_MODULE = runpy.run_path(str(_GENERATOR))
OUT = _MODULE['OUT']
BORROWED_OUT = _MODULE['BORROWED_OUT']
generate = _MODULE['generate']

if __name__ == '__main__':
    runpy.run_path(str(_GENERATOR), run_name='__main__')
