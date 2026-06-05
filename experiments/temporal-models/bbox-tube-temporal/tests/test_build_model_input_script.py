"""Unit tests for build_model_input.py's pure helpers."""

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_model_input.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_model_input_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_model_input_script"] = module
    spec.loader.exec_module(module)
    return module


def test_to_bool_parses_dvc_strings():
    mod = _load()
    assert mod._to_bool("true") is True
    assert mod._to_bool("True") is True
    assert mod._to_bool("false") is False
    assert mod._to_bool("False") is False
    assert mod._to_bool("0") is False
