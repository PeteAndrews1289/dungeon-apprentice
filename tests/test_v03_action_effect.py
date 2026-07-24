from __future__ import annotations

from copy import deepcopy

from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v03_action_effect as v03
from dungeon_apprentice.action_effect import ActionEffectMode


def _stable_exam(index: int) -> dict:
    return {
        "child_trained_actions": index * v03.EVALUATION_INTERVAL,
        "allocation_valid": True,
        "practice_profile": "normal",
        "lessons": {
            lesson.value: {
                "successes": 80,
                "panel_successes": [40, 40],
                "mean_ineffective_interactions": 0.0,
            }
            for lesson in lessons.LessonId
        },
        "case_diagnostics": {
            "max_case_ineffective_interactions": 0,
            "cases_with_ineffective_at_least_10": 0,
            "max_repeated_identical_interaction_run": 0,
        },
    }


def _stable_arm() -> list[dict]:
    return [
        _stable_exam(index)
        for index in range(1, v03.EXAM_COUNT + 1)
    ]


def test_stage_a_is_a_matched_architecture_only_study() -> None:
    sham = v03.ARM_SPECS[ActionEffectMode.SHAM]
    candidate = v03.ARM_SPECS[ActionEffectMode.ACTION_EFFECT]

    assert v03.ARM_ORDER == (
        ActionEffectMode.SHAM,
        ActionEffectMode.ACTION_EFFECT,
    )
    assert not sham.context_enabled
    assert not sham.selectable
    assert candidate.context_enabled
    assert candidate.selectable
    assert v03.WORKER_STREAMS == (20260757, 20260758, 20260759, 20260760)
    assert v03.CHILD_ACTION_BUDGET == 1_048_576
    assert v03.EXAM_COUNT == 32
    assert v03.FINAL_STABILITY_EXAMS == 3

    sham_config = v03.effective_config(ActionEffectMode.SHAM)
    candidate_config = v03.effective_config(ActionEffectMode.ACTION_EFFECT)
    assert sham_config["optimization"] == candidate_config["optimization"]
    assert sham_config["reward"] == candidate_config["reward"]
    assert (
        sham_config["observation"]["action_effect"]["sham_always_zero"]
        is True
    )
    assert (
        candidate_config["observation"]["action_effect"]["sham_always_zero"]
        is False
    )
    assert not candidate_config["selection"]["development_checkpoint_reuse_authorized"]


def test_policy_contract_preserves_the_confirmed_parent_topology() -> None:
    import pytest

    pytest.importorskip("sb3_contrib")
    contract = v03.public_policy_contract()
    kwargs = v03.policy_kwargs()

    assert contract["visual_features"] == 512
    assert contract["actor_lstm"] == {"hidden_size": 256, "layers": 1}
    assert contract["critic_lstm"] == {"hidden_size": 256, "layers": 1}
    assert contract["post_lstm_mlp"] == []
    assert kwargs["features_extractor_kwargs"] == {"features_dim": 512}
    assert kwargs["net_arch"] == []
    assert kwargs["ortho_init"] is False


def test_candidate_must_pass_all_three_terminal_exams() -> None:
    records = _stable_arm()
    grade = v03.grade_terminal(ActionEffectMode.ACTION_EFFECT, records)
    assert grade.eligible
    assert grade.final_exam_actions == (
        983_040,
        1_015_808,
        1_048_576,
    )

    records[-2]["case_diagnostics"][
        "cases_with_ineffective_at_least_10"
    ] = 1
    records[-2]["case_diagnostics"][
        "max_case_ineffective_interactions"
    ] = 10
    failed = v03.grade_terminal(
        ActionEffectMode.ACTION_EFFECT,
        records,
    )
    assert not failed.eligible
    assert any("terminal_2:cases" in reason for reason in failed.reasons)


def test_sham_can_never_authorize_replication() -> None:
    stable = _stable_arm()
    failed_candidate = deepcopy(stable)
    for exam in failed_candidate[-3:]:
        exam["lessons"][
            lessons.LessonId.SEPARATED_UNLOCK.value
        ]["successes"] = 71
        exam["lessons"][
            lessons.LessonId.SEPARATED_UNLOCK.value
        ]["panel_successes"] = [36, 35]

    selection = v03.select_architecture(
        {
            ActionEffectMode.SHAM: stable,
            ActionEffectMode.ACTION_EFFECT: failed_candidate,
        }
    )
    assert selection["verdict"] == "architecture_failed"
    assert selection["selected_architecture"] is None
    assert not selection["replication_protocol_authorized"]
    assert not selection["development_checkpoint_reuse_authorized"]
    assert not selection["u3_authorized"]


def test_candidate_pass_selects_definition_not_checkpoint() -> None:
    selection = v03.select_architecture(
        {
            ActionEffectMode.SHAM: _stable_arm(),
            ActionEffectMode.ACTION_EFFECT: _stable_arm(),
        }
    )
    assert selection["verdict"] == "architecture_selected"
    assert selection["selected_architecture"]["name"] == "action-effect"
    assert selection["replication_protocol_authorized"]
    assert not selection["development_checkpoint_reuse_authorized"]
    assert "checkpoint" not in selection["selected_architecture"]
    assert not selection["u3_authorized"]
