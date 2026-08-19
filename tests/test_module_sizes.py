"""Guards on module size, with the frozen research modules grandfathered.

The v0.2 through v0.4 experiment modules are frozen evidence. ``AGENTS.md``
states plainly that their environments, reports, checkpoints, launchers, and
declared results must not be modified, and that constraint is the point of the
repository: the value of a preserved negative result depends on the code that
produced it not being quietly rewritten afterwards.

So this guard does not demand they be split. It records their size at the moment
the guard was written and fails if any of them *grows*, while holding every
other module - anything new, and anything that is ordinary infrastructure - to a
real cap.

That is the honest position. The large files are a genuine readability cost, and
the architecture note says so. They are also immutable, and refactoring them
would trade a documented weakness for an undocumented integrity problem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "dungeon_apprentice"

# Modules that may not be split, recorded at their current size. These are
# allowed to shrink and never to grow.
FROZEN_MODULE_LINES: dict[str, int] = {
    "u2r_anchor.py": 1754,
    "v02_lessons.py": 1129,
    "v02_sentinel.py": 1082,
    "v02_u1.py": 1155,
    "v02_u1_confirm.py": 1602,
    "v02_u1_confirm_v2.py": 1182,
    "v02_u2.py": 3048,
    "v02_u2_confirm.py": 2764,
    "v02_u2_dashboard.py": 826,
    "v02_u2_lessons.py": 2085,
    "v02_u2_qualify.py": 1788,
    "v02_u2_smoke.py": 602,
    "v02_u2r.py": 5888,
    "v02_u2s.py": 4296,
    "v02_u2s_dashboard.py": 1364,
    "v02_u2s_qualify.py": 2311,
    "v02_u2s_smoke.py": 976,
    "v03_action_effect_dashboard.py": 1217,
    "v03_action_effect_qualify.py": 3277,
    "v03_action_effect_smoke.py": 658,
    "v03_action_effect_train.py": 3130,
    "v04_ineffective_trace.py": 751,
    "v04_ineffective_trace_dashboard.py": 1222,
    "v04_ineffective_trace_qualify.py": 2715,
    "v04_ineffective_trace_smoke.py": 883,
    "v04_ineffective_trace_train.py": 3253,
}

# Everything else - new modules and live infrastructure - is held to this.
MAX_MODULE_LINES = 700


def module_paths() -> list[Path]:
    return sorted(SOURCE.rglob("*.py"))


def line_count(path: Path) -> int:
    return len(path.read_text().splitlines())


@pytest.mark.parametrize("path", module_paths(), ids=lambda p: p.name)
def test_module_size(path: Path) -> None:
    count = line_count(path)
    recorded = FROZEN_MODULE_LINES.get(path.name)

    if recorded is not None:
        assert count <= recorded, (
            f"{path.name} grew from {recorded} to {count} lines. It is frozen "
            "research evidence; it may shrink but must not grow."
        )
        return

    assert count <= MAX_MODULE_LINES, (
        f"{path.name} is {count} lines, over the {MAX_MODULE_LINES} cap. "
        "Split it rather than raising the cap or adding it to the frozen list."
    )


def test_frozen_list_only_names_modules_that_exist() -> None:
    """A stale entry would silently create an uncapped module name."""
    present = {path.name for path in module_paths()}
    missing = sorted(set(FROZEN_MODULE_LINES) - present)
    assert missing == [], f"frozen list names modules that no longer exist: {missing}"


def test_frozen_list_does_not_grow() -> None:
    """New oversized modules must be split, not grandfathered."""
    assert len(FROZEN_MODULE_LINES) <= 26, (
        "The frozen list is an inventory of existing evidence, not an escape "
        "hatch. A new module over the cap should be split."
    )


def test_training_package_stays_split() -> None:
    """train.py was 1,115 lines; it is now a package and should stay one."""
    package = SOURCE / "training"
    assert package.is_dir()
    modules = sorted(package.glob("*.py"))
    assert len(modules) >= 4
    for path in modules:
        assert line_count(path) <= MAX_MODULE_LINES, path.name


def test_train_shim_still_exposes_the_original_surface() -> None:
    from dungeon_apprentice import train
    from dungeon_apprentice.training import __all__ as training_all

    missing = [name for name in training_all if not hasattr(train, name)]
    assert missing == [], f"the shim stopped re-exporting: {missing}"


def test_train_entry_point_is_callable() -> None:
    from dungeon_apprentice import train

    assert callable(train.main)
    assert callable(train.build_parser)
