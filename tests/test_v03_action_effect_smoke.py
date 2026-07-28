from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from dungeon_apprentice import v03_action_effect as v03
from dungeon_apprentice import v03_action_effect_smoke as smoke
from dungeon_apprentice.action_effect import ActionEffectMode


def _arm_result(mode: ActionEffectMode) -> dict:
    return {
        "arm": mode.value,
        "before": {"policy": "identical", "optimizer": "identical"},
        "after": {"policy": mode.value},
        "effect_projection_nonzero_parameters": (
            0 if mode is ActionEffectMode.SHAM else 4_608
        ),
        "transplant": {"zero_context_equivalence": True},
        "workers": [],
        "trajectory_identity": [{"trajectory_sha256": "a" * 64}],
        "pre_action_rng_identity": {"aggregate_sha256": "b" * 64},
        "policy_output_sha256": "c" * 64,
        "post_rollout_rng_identity": {"aggregate_sha256": "d" * 64},
        "episode_ledger": {
            "records": 10,
            "normalized_sha256": "e" * 64,
        },
        "seed_evidence": {"training_range_only": True},
        "updated_archive_reloaded_exactly": True,
        "development_checkpoint_reuse_authorized": False,
    }


def test_smoke_contract_is_one_real_matched_rollout_per_arm() -> None:
    assert smoke.TOTAL_TRANSITIONS == 2_048
    assert smoke.TOTAL_TRANSITIONS == v03.WORKERS * v03.ROLLOUT_STEPS
    assert v03.ARM_ORDER == (
        ActionEffectMode.SHAM,
        ActionEffectMode.ACTION_EFFECT,
    )
    assert vars(smoke.build_parser().parse_args([])) == {
        "repository": Path.cwd(),
        "allow_dirty": False,
    }


def test_recursive_space_rng_identity_covers_dict_children() -> None:
    space = gym.spaces.Dict(
        {
            "image": gym.spaces.Box(
                0,
                255,
                shape=(3, 4, 4),
                dtype=np.uint8,
            ),
            "action_effect": gym.spaces.Box(
                0.0,
                1.0,
                shape=(9,),
                dtype=np.float32,
            ),
        }
    )
    space.seed(123)
    identity = smoke._space_rng_identity(space)

    assert identity["type"] == "Dict"
    assert set(identity["children"]) == {"image", "action_effect"}
    assert all(
        len(child["self_sha256"]) == 64
        for child in identity["children"].values()
    )


def test_run_smoke_requires_complete_matched_equality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent.zip"
    parent.write_bytes(b"confirmed U1")
    from dungeon_apprentice.artifacts import file_sha256

    monkeypatch.setattr(v03, "PARENT_CHECKPOINT", parent)
    monkeypatch.setattr(
        v03,
        "PARENT_CHECKPOINT_SHA256",
        file_sha256(parent),
    )
    monkeypatch.setattr(
        smoke,
        "git_snapshot",
        lambda _repository: {"commit": "a" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        smoke.frozen_smoke,
        "_load_protocol_inputs",
        lambda _repository: SimpleNamespace(
            forbidden_layout_hashes={
                lesson: frozenset() for lesson in smoke.lessons.LessonId
            },
            seed_access=object(),
        ),
    )
    monkeypatch.setattr(
        smoke.frozen_u2r,
        "preflight_u2r_training_layout_sampler",
        lambda *_args, **_kwargs: ({"stream": 20260757},),
    )
    monkeypatch.setattr(
        smoke.frozen_smoke,
        "_canonical_root_snapshots",
        lambda: [{"unchanged": True}],
    )
    monkeypatch.setattr(
        smoke,
        "_run_arm",
        lambda mode, **_kwargs: _arm_result(mode),
    )

    result = smoke.run_smoke(repository=tmp_path)
    assert result["verdict"] == "passed"
    assert result["pre_action_rng_identical"]
    assert result["first_rollout_policy_outputs_identical"]
    assert result["post_rollout_pre_optimizer_rng_identical"]
    assert result["sampler_preflight"] == [{"stream": 20260757}]
    assert len(result["sampler_preflight_sha256"]) == 64
    assert len(result["guard_mapping_sha256"]) == 64


def test_run_smoke_rejects_policy_output_divergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent.zip"
    parent.write_bytes(b"confirmed U1")
    from dungeon_apprentice.artifacts import file_sha256

    monkeypatch.setattr(v03, "PARENT_CHECKPOINT", parent)
    monkeypatch.setattr(
        v03,
        "PARENT_CHECKPOINT_SHA256",
        file_sha256(parent),
    )
    monkeypatch.setattr(
        smoke,
        "git_snapshot",
        lambda _repository: {"commit": "a" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        smoke.frozen_smoke,
        "_load_protocol_inputs",
        lambda _repository: SimpleNamespace(
            forbidden_layout_hashes={
                lesson: frozenset() for lesson in smoke.lessons.LessonId
            },
            seed_access=object(),
        ),
    )
    monkeypatch.setattr(
        smoke.frozen_u2r,
        "preflight_u2r_training_layout_sampler",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        smoke.frozen_smoke,
        "_canonical_root_snapshots",
        lambda: [],
    )

    def divergent(mode: ActionEffectMode, **_kwargs: object) -> dict:
        result = _arm_result(mode)
        if mode is ActionEffectMode.ACTION_EFFECT:
            result["policy_output_sha256"] = "f" * 64
        return result

    monkeypatch.setattr(smoke, "_run_arm", divergent)
    with pytest.raises(
        smoke.ActionEffectSmokeError,
        match="diverged before",
    ):
        smoke.run_smoke(repository=tmp_path)


def test_run_smoke_uses_explicit_canonical_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent.zip"
    parent.write_bytes(b"confirmed U1")
    from dungeon_apprentice.artifacts import file_sha256

    mapping = {
        lesson: frozenset({f"{index:064x}"})
        for index, lesson in enumerate(smoke.lessons.LessonId, start=1)
    }
    access = object()
    observed: list[tuple[object, object]] = []
    monkeypatch.setattr(v03, "PARENT_CHECKPOINT", parent)
    monkeypatch.setattr(
        v03,
        "PARENT_CHECKPOINT_SHA256",
        file_sha256(parent),
    )
    monkeypatch.setattr(
        smoke,
        "git_snapshot",
        lambda _repository: {"commit": "a" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        smoke.frozen_smoke,
        "_load_protocol_inputs",
        lambda _repository: pytest.fail("legacy guard loader was used"),
    )

    def preflight(
        supplied: object,
        *,
        seed_access: object,
        **_kwargs: object,
    ) -> tuple[dict[str, int], ...]:
        observed.append((supplied, seed_access))
        return ({"stream": 20260757},)

    monkeypatch.setattr(
        smoke.frozen_u2r,
        "preflight_u2r_training_layout_sampler",
        preflight,
    )
    monkeypatch.setattr(
        smoke.frozen_smoke,
        "_canonical_root_snapshots",
        lambda: [],
    )
    monkeypatch.setattr(
        smoke,
        "_run_arm",
        lambda mode, **_kwargs: _arm_result(mode),
    )

    result = smoke.run_smoke(
        repository=tmp_path,
        seed_access=access,
        forbidden_layout_hashes=mapping,
    )

    assert observed == [(mapping, access)]
    assert result["guard_mapping_sha256"] == smoke._guard_mapping_sha256(
        mapping
    )
