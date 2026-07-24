from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2s_smoke as frozen_smoke
from dungeon_apprentice import v04_ineffective_trace as v04
from dungeon_apprentice.action_streak import IneffectiveTraceMode


def _stable_exam(index: int) -> dict:
    return {
        "child_trained_actions": index * v04.EVALUATION_INTERVAL,
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
    return [_stable_exam(index) for index in range(1, v04.EXAM_COUNT + 1)]


def test_v04_is_one_scalar_matched_architecture_study_from_confirmed_u1() -> None:
    control = v04.ARM_SPECS[IneffectiveTraceMode.ZERO_TRACE]
    candidate = v04.ARM_SPECS[IneffectiveTraceMode.STREAK_TRACE]

    assert v04.PROTOCOL == ("dungeon-apprentice-v0.4-ineffective-trace-architecture")
    assert v04.ARCHITECTURE_VERSION == "bounded-ineffective-trace-v1"
    assert v04.ARM_ORDER == (
        IneffectiveTraceMode.ZERO_TRACE,
        IneffectiveTraceMode.STREAK_TRACE,
    )
    assert control.mode.value == "trace-sham"
    assert not control.trace_enabled
    assert not control.selectable
    assert candidate.mode.value == "ineffective-trace"
    assert candidate.trace_enabled
    assert candidate.selectable

    assert v04.PARENT_CHILD_SEED == 20260745
    assert v04.PARENT_U1_SEED == 20260733
    assert v04.PARENT_CHECKPOINT_SHA256 == (
        "3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104"
    )
    assert v04.PARENT_POLICY_TENSOR_SHA256 == (
        "e555d3f7e2364f74e3b43371938c2f25e2ddbf62558a810038509a70868e6888"
    )
    assert v04.PARENT_OPTIMIZER_STATE_SHA256 == (
        "cc07791b374620d680e5cbb3602d59ab83eb55f195d26808abc0adf5f0e3bc5f"
    )
    assert v04.PARENT_LIFETIME_ACTIONS == 786_432
    assert v04.PARENT_OPTIMIZER_UPDATES == 1_536

    assert v04.ARCHITECTURE_INITIALIZATION_SEED == 20260761
    assert v04.ALGORITHM_SEED == 20260762
    assert v04.WORKER_STREAMS == (
        20260762,
        20260763,
        20260764,
        20260765,
    )
    assert v04.CHILD_ACTION_BUDGET == 1_048_576
    assert v04.EXAM_COUNT == 32
    assert v04.FINAL_STABILITY_EXAMS == 3


def test_effective_configs_differ_only_in_trace_exposure_and_arm_identity() -> None:
    control = v04.effective_config(IneffectiveTraceMode.ZERO_TRACE)
    candidate = v04.effective_config(IneffectiveTraceMode.STREAK_TRACE)

    assert control["parent"] == candidate["parent"]
    assert control["paired_randomness"] == candidate["paired_randomness"]
    assert control["budget"] == candidate["budget"]
    assert control["policy"] == candidate["policy"]
    assert control["optimization"] == candidate["optimization"]
    assert control["reward"] == candidate["reward"]
    assert control["selection"] == candidate["selection"]

    control_trace = control["observation"]["ineffective_trace"]
    candidate_trace = candidate["observation"]["ineffective_trace"]
    assert control_trace["shape"] == candidate_trace["shape"] == [1]
    assert control_trace["definition"] == (
        "min(consecutive_same_action_byte_identical_rgb_transitions,9)/9"
    )
    assert control_trace["clip_streak_at"] == 9
    assert control_trace["normalization_divisor"] == 9.0
    assert control_trace["range"] == [0.0, 1.0]
    assert control_trace["transition_rule"] == {
        "episode_start": 0,
        "visible_change": 0,
        "first_unchanged_transition": 1,
        "action_switch_with_unchanged_rgb": 1,
        "same_action_unchanged_rgb": "min(previous_count+1,9)",
    }
    assert control_trace["trace_sham_always_zero"] is True
    assert candidate_trace["trace_sham_always_zero"] is False
    assert candidate["observation"]["retains_v03_nine_dimensional_action_effect"] is False

    parent_text = json.dumps(candidate["parent"], sort_keys=True)
    assert candidate["parent"]["source_protocol"] == ("dungeon-apprentice-v0.2-u1")
    assert "/v03-" not in parent_text
    assert "action-effect" not in parent_text
    assert candidate["reward"]["changed_from_original_u2_control"] is False
    assert candidate["optimization"]["changed_from_original_u2_control"] is False


def test_policy_contract_preserves_u1_topology_with_one_zero_residual() -> None:
    pytest.importorskip("sb3_contrib")
    contract = v04.public_policy_contract()
    kwargs = v04.policy_kwargs()

    assert contract["visual_features"] == 512
    assert contract["ineffective_trace_features"] == 1
    assert contract["trace_encoder_parameter"] == (
        "features_extractor.ineffective_trace_encoder.weight"
    )
    assert contract["feature_fusion"] == ("zero_initialized_additive_residual")
    assert contract["actor_lstm"] == {"hidden_size": 256, "layers": 1}
    assert contract["critic_lstm"] == {"hidden_size": 256, "layers": 1}
    assert contract["post_lstm_mlp"] == []
    assert kwargs["features_extractor_kwargs"] == {"features_dim": 512}
    assert kwargs["net_arch"] == []
    assert kwargs["ortho_init"] is False


def test_vector_observation_contract_rejects_the_failed_nine_dimensional_input() -> None:
    valid = SimpleNamespace(
        observation_space=gym.spaces.Dict(
            {
                v04.IMAGE_KEY: gym.spaces.Box(
                    0,
                    255,
                    shape=(3, 56, 56),
                    dtype=np.uint8,
                ),
                v04.INEFFECTIVE_TRACE_KEY: gym.spaces.Box(
                    0.0,
                    1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )
    )
    v04._validate_dict_environment(valid)

    invalid = SimpleNamespace(
        observation_space=gym.spaces.Dict(
            {
                v04.IMAGE_KEY: gym.spaces.Box(
                    0,
                    255,
                    shape=(3, 56, 56),
                    dtype=np.uint8,
                ),
                v04.INEFFECTIVE_TRACE_KEY: gym.spaces.Box(
                    0.0,
                    1.0,
                    shape=(9,),
                    dtype=np.float32,
                ),
            }
        )
    )
    with pytest.raises(
        v04.IneffectiveTraceProtocolError,
        match="schema changed",
    ):
        v04._validate_dict_environment(invalid)


def test_named_transplant_copies_u1_adam_state_and_only_adds_scalar_encoder() -> None:
    torch = pytest.importorskip("torch")

    class LegacyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2, bias=False)
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=2.5e-4,
            )

    class TraceFeatures(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ineffective_trace_encoder = torch.nn.Linear(
                1,
                512,
                bias=False,
            )

    class TracePolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2, bias=False)
            self.features_extractor = TraceFeatures()
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=9.9e-4,
            )

    legacy_policy = LegacyPolicy()
    legacy_policy.backbone.weight.sum().backward()
    legacy_policy.optimizer.step()
    legacy = SimpleNamespace(
        policy=legacy_policy,
        num_timesteps=786_432,
        _n_updates=1_536,
    )
    trace_policy = TracePolicy()
    trace = SimpleNamespace(
        policy=trace_policy,
        num_timesteps=0,
        _n_updates=0,
    )

    evidence = v04.transplant_legacy_model_state(legacy, trace)

    assert evidence.new_parameter_names == ("features_extractor.ineffective_trace_encoder.weight",)
    assert not evidence.missing_optimizer_state_names
    assert evidence.inherited_timesteps == 786_432
    assert evidence.inherited_optimizer_updates == 1_536
    assert torch.equal(
        trace_policy.backbone.weight,
        legacy_policy.backbone.weight,
    )
    encoder = trace_policy.features_extractor.ineffective_trace_encoder.weight
    assert torch.count_nonzero(encoder).item() == 0
    assert encoder not in trace_policy.optimizer.state
    assert trace_policy.optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4)

    old_state = legacy_policy.optimizer.state[legacy_policy.backbone.weight]
    new_state = trace_policy.optimizer.state[trace_policy.backbone.weight]
    assert set(old_state) == set(new_state)
    assert all(
        torch.equal(old_state[key], new_state[key])
        if torch.is_tensor(old_state[key])
        else old_state[key] == new_state[key]
        for key in old_state
    )


def test_candidate_must_pass_the_unchanged_three_terminal_exam_gate() -> None:
    records = _stable_arm()
    grade = v04.grade_terminal(
        IneffectiveTraceMode.STREAK_TRACE,
        records,
    )
    assert grade.eligible
    assert grade.final_exam_actions == (
        983_040,
        1_015_808,
        1_048_576,
    )

    records[-2]["case_diagnostics"]["cases_with_ineffective_at_least_10"] = 1
    records[-2]["case_diagnostics"]["max_case_ineffective_interactions"] = 10
    failed = v04.grade_terminal(
        IneffectiveTraceMode.STREAK_TRACE,
        records,
    )
    assert not failed.eligible
    assert any("terminal_2:cases" in reason for reason in failed.reasons)


def test_only_candidate_definition_can_authorize_versioned_replication() -> None:
    stable = _stable_arm()
    failed_candidate = deepcopy(stable)
    for exam in failed_candidate[-3:]:
        exam["lessons"][lessons.LessonId.SEPARATED_UNLOCK.value]["successes"] = 71
        exam["lessons"][lessons.LessonId.SEPARATED_UNLOCK.value]["panel_successes"] = [36, 35]

    failed = v04.select_architecture(
        {
            IneffectiveTraceMode.ZERO_TRACE: stable,
            IneffectiveTraceMode.STREAK_TRACE: failed_candidate,
        }
    )
    assert failed["verdict"] == "architecture_failed"
    assert failed["selected_architecture"] is None
    assert not failed["replication_protocol_authorized"]
    assert not failed["separately_versioned_replication_required"]
    assert not failed["development_checkpoint_reuse_authorized"]
    assert not failed["u3_authorized"]

    failed_control = deepcopy(stable)
    failed_control[-1]["case_diagnostics"]["max_case_ineffective_interactions"] = 10
    failed_control[-1]["case_diagnostics"]["cases_with_ineffective_at_least_10"] = 1
    selected = v04.select_architecture(
        {
            IneffectiveTraceMode.ZERO_TRACE: failed_control,
            IneffectiveTraceMode.STREAK_TRACE: stable,
        }
    )
    assert selected["verdict"] == "architecture_selected"
    assert selected["selected_architecture"]["name"] == ("ineffective-trace")
    assert selected["replication_protocol_authorized"]
    assert selected["separately_versioned_replication_required"]
    assert not selected["development_checkpoint_reuse_authorized"]
    assert "checkpoint" not in selected["selected_architecture"]
    assert not selected["u3_authorized"]


def test_first_rollout_digest_retains_the_qualified_lf_profile() -> None:
    value = {"unicode": "Pokémon", "nested": [4, {"matched": True}]}
    assert v04.FIRST_ROLLOUT_DIGEST_PROFILE == ("u2s-canonical-json-v1-lf")
    assert v04.first_rollout_identity_sha256(value) == frozen_smoke._canonical_sha256(value)
    with pytest.raises(ValueError):
        v04.first_rollout_identity_sha256({"invalid": float("nan")})
