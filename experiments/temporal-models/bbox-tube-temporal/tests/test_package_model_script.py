"""Unit tests for ``scripts/package_model.py``'s pure helpers."""

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_model.py"


def _load_script_module(alias: str):
    spec = importlib.util.spec_from_file_location(alias, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def test_tubes_config_includes_merge_keys_when_all_present():
    mod = _load_script_module("package_model_script_all_present")
    all_params = {
        "tubes": {
            "iou_threshold": 0.2,
            "max_misses": 2,
            "merge_iomin": 0.3,
            "merge_prox_factor": 1.0,
            "merge_max_gap": 10,
        },
        "build_tubes": {"min_tube_length": 4, "min_detected_entries": 2},
        "package": {"infer_min_tube_length": 2},
    }
    cfg = mod._tubes_config(all_params)
    assert cfg["merge_iomin"] == 0.3
    assert cfg["merge_prox_factor"] == 1.0
    assert cfg["merge_max_gap"] == 10


def test_tubes_config_omits_merge_keys_when_any_missing():
    mod = _load_script_module("package_model_script_some_missing")
    all_params = {
        "tubes": {"iou_threshold": 0.2, "max_misses": 2, "merge_iomin": 0.3},
        "build_tubes": {"min_tube_length": 4, "min_detected_entries": 2},
        "package": {"infer_min_tube_length": 2},
    }
    cfg = mod._tubes_config(all_params)
    assert "merge_iomin" not in cfg
    assert "merge_prox_factor" not in cfg
    assert "merge_max_gap" not in cfg


def test_model_input_config_carries_stabilize_default_false():
    mod = _load_script_module("package_model_script_mi_default")
    all_params = {"model_input": {"context_factor": 1.5, "patch_size": 224}}
    cfg = mod._model_input_config(all_params)
    assert cfg["context_factor"] == 1.5
    assert cfg["patch_size"] == 224
    assert cfg["stabilize"] is False


def test_model_input_config_carries_stabilize_true():
    mod = _load_script_module("package_model_script_mi_true")
    all_params = {
        "model_input": {"context_factor": 1.5, "patch_size": 224, "stabilize": True}
    }
    cfg = mod._model_input_config(all_params)
    assert cfg["stabilize"] is True


def test_model_input_config_stabilize_override_true():
    mod = _load_script_module("package_model_script_override_true")
    all_params = {
        "model_input": {"context_factor": 1.5, "patch_size": 224, "stabilize": False}
    }
    cfg = mod._model_input_config(all_params, stabilize=True)
    assert cfg["stabilize"] is True


def test_model_input_config_stabilize_override_none_uses_param():
    mod = _load_script_module("package_model_script_override_none")
    all_params = {
        "model_input": {"context_factor": 1.5, "patch_size": 224, "stabilize": True}
    }
    cfg = mod._model_input_config(all_params, stabilize=None)
    assert cfg["stabilize"] is True


def test_to_bool_parses_dvc_strings():
    mod = _load_script_module("package_model_script_to_bool")
    assert mod._to_bool("true") is True
    assert mod._to_bool("false") is False


def test_apply_infer_overrides_sets_values():
    mod = _load_script_module("pkg_overrides_set")
    pkg = {"infer": {"pad_to_min_frames": 20, "pad_strategy": "symmetric"}}
    mod._apply_infer_overrides(pkg, pad_to_min_frames=8, pad_strategy="uniform")
    assert pkg["infer"]["pad_to_min_frames"] == 8
    assert pkg["infer"]["pad_strategy"] == "uniform"


def test_apply_infer_overrides_none_is_noop():
    mod = _load_script_module("pkg_overrides_noop")
    pkg = {"infer": {"pad_to_min_frames": 20, "pad_strategy": "symmetric"}}
    mod._apply_infer_overrides(pkg, pad_to_min_frames=None, pad_strategy=None)
    assert pkg["infer"]["pad_to_min_frames"] == 20
    assert pkg["infer"]["pad_strategy"] == "symmetric"
