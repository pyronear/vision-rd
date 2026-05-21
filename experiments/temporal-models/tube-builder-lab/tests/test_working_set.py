from pathlib import Path

from tube_builder_lab.working_set import WorkingItem, load_working_set


def test_load_working_set(tmp_path: Path):
    p = tmp_path / "ws.yaml"
    p.write_text(
        "targets:\n"
        "  - { key: platform_1, note: 'three into one' }\n"
        "  - { key: platform_2 }\n"
        "control:\n"
        "  - { key: platform_9 }\n"
    )
    ws = load_working_set(p)
    assert ws.targets == [
        WorkingItem(key="platform_1", note="three into one"),
        WorkingItem(key="platform_2", note=None),
    ]
    assert ws.control == [WorkingItem(key="platform_9", note=None)]
    assert [i.key for i in ws.all()] == ["platform_1", "platform_2", "platform_9"]


def test_load_working_set_empty_control(tmp_path: Path):
    p = tmp_path / "ws.yaml"
    p.write_text("targets:\n  - { key: platform_1 }\ncontrol: []\n")
    ws = load_working_set(p)
    assert ws.control == []
