import pytest

from playground.cli import build_parser


def test_parser_accepts_model_name_and_dir():
    args = build_parser().parse_args(["run", "--model", "m", "./seq"])
    assert args.command == "run"
    assert args.model == "m"
    assert args.inputs == ["./seq"]
    assert args.json is False


def test_parser_accepts_package_and_multiple_paths():
    args = build_parser().parse_args(
        ["run", "--model-package", "x.zip", "a.jpg", "b.jpg", "--json"]
    )
    assert args.model_package == "x.zip"
    assert args.inputs == ["a.jpg", "b.jpg"]
    assert args.json is True


def test_parser_requires_inputs():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--model", "m"])
