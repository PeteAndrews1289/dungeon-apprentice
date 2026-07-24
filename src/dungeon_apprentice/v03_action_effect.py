"""v0.3 matched action-effect architecture study.

This is a new architecture protocol, not a continuation of U2-S.  Two fresh
development twins begin at the same authenticated, confirmed U1 parent:

``sham``
    Receives the new Dict observation and the new zero-initialized network
    pathway, but its context vector is always zero.
``action-effect``
    Receives its own preceding primitive action plus a visible
    changed/unchanged outcome derived from consecutive RGB observations.

Reward, PPO, curriculum, lesson mechanics, action budget, and evaluation gates
are held at the original U2 control settings.  No U2, U2r, or U2-S checkpoint
is loaded.  The study may select only an architecture definition; its
development checkpoints are never promotable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import gymnasium as gym
import numpy as np

from dungeon_apprentice import v02_u2 as frozen_u2
from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2s as frozen_u2s
from dungeon_apprentice.action_effect import (
    ACTION_EFFECT_DIM,
    ACTION_EFFECT_KEY,
    IMAGE_KEY,
    ActionEffectMode,
    ActionEffectNatureCNN,
    ActionEffectObservation,
    assert_zero_context_equivalence,
    transplant_legacy_model_state,
)

PROTOCOL = "dungeon-apprentice-v0.3-action-effect-architecture"
SCHEMA_VERSION = 1
ARCHITECTURE_VERSION = "retrospective-action-effect-context-v1"

PARENT_CHILD_SEED = 20260745
PARENT_U1_SEED = 20260733
PARENT_CHECKPOINT = frozen_u2.FROZEN_PARENTS[PARENT_CHILD_SEED].archive
PARENT_CHECKPOINT_SHA256 = "3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104"
PARENT_POLICY_TENSOR_SHA256 = "e555d3f7e2364f74e3b43371938c2f25e2ddbf62558a810038509a70868e6888"
PARENT_OPTIMIZER_STATE_SHA256 = "cc07791b374620d680e5cbb3602d59ab83eb55f195d26808abc0adf5f0e3bc5f"
PARENT_LIFETIME_ACTIONS = 786_432
PARENT_OPTIMIZER_UPDATES = 1_536

ALGORITHM_SEED = 20260757
WORKER_STREAMS = (20260757, 20260758, 20260759, 20260760)
ARCHITECTURE_INITIALIZATION_SEED = 20260756
WORKERS = 4
ROLLOUT_STEPS = 512
ROLLOUT_TRANSITIONS = WORKERS * ROLLOUT_STEPS
BATCH_SIZE = 256
PPO_EPOCHS = 4
LEARNING_RATE = 2.5e-4
CLIP_RANGE = 0.2
ENTROPY_COEFFICIENT = 0.01
GAMMA = 0.995
GAE_LAMBDA = 0.98

CHILD_ACTION_BUDGET = 1_048_576
EVALUATION_INTERVAL = 32_768
EVALUATION_SEED_COUNT = 80
EXAM_COUNT = CHILD_ACTION_BUDGET // EVALUATION_INTERVAL
FINAL_STABILITY_EXAMS = 3
MINIMUM_FREE_GIB = 25.0
LINEAGE_CAP_BYTES = 2 * 1024**3
COHORT_SCIENTIFIC_CAP_BYTES = 4 * 1024**3
MEDIA_CAP_BYTES = 6 * 1024**3
COMBINED_PLANNED_CAP_BYTES = 10 * 1024**3
PROTOCOL_DOCUMENT = "docs/protocol-v0.3-action-effect-architecture-r2.md"

ARM_ORDER = (
    ActionEffectMode.SHAM,
    ActionEffectMode.ACTION_EFFECT,
)


class ActionEffectProtocolError(RuntimeError):
    """Raised when the v0.3 study changes or loses its frozen boundary."""


@dataclass(frozen=True)
class ArmSpec:
    mode: ActionEffectMode
    context_enabled: bool
    selectable: bool

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.mode.value,
            "architecture_version": ARCHITECTURE_VERSION,
            "context_enabled": self.context_enabled,
            "selectable": self.selectable,
            "reward_shaping": False,
            "ppo_changed": False,
        }


ARM_SPECS: Mapping[ActionEffectMode, ArmSpec] = MappingProxyType(
    {
        ActionEffectMode.SHAM: ArmSpec(
            mode=ActionEffectMode.SHAM,
            context_enabled=False,
            selectable=False,
        ),
        ActionEffectMode.ACTION_EFFECT: ArmSpec(
            mode=ActionEffectMode.ACTION_EFFECT,
            context_enabled=True,
            selectable=True,
        ),
    }
)


def policy_kwargs() -> dict[str, Any]:
    """Return the only policy topology allowed by this protocol."""

    if ActionEffectNatureCNN is None:
        raise ActionEffectProtocolError(
            'install training dependencies with: pip install -e ".[train]"'
        )
    return {
        "features_extractor_class": ActionEffectNatureCNN,
        "features_extractor_kwargs": {"features_dim": 512},
        "net_arch": [],
        # NatureCNN and every inherited tensor are overwritten by an exact
        # transplant.  Disabling the policy-level orthogonal pass also
        # prevents it from replacing the extractor's required zero residual.
        "ortho_init": False,
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,
    }


def public_policy_contract() -> dict[str, Any]:
    return {
        "policy_class": "RecurrentMultiInputActorCriticPolicy",
        "feature_extractor": ("dungeon_apprentice.action_effect.ActionEffectNatureCNN"),
        "visual_features": 512,
        "action_effect_features": ACTION_EFFECT_DIM,
        "feature_fusion": "zero_initialized_additive_residual",
        "post_lstm_mlp": [],
        "actor_lstm": {"hidden_size": 256, "layers": 1},
        "critic_lstm": {"hidden_size": 256, "layers": 1},
        "actions": 7,
    }


def make_training_environment(
    *,
    scheduler: lessons.TransitionDeficitScheduler,
    seed: int,
    seed_access: Any,
    forbidden_layout_hashes: Mapping[lessons.LessonId, frozenset[str]],
    mode: ActionEffectMode | str,
) -> gym.Env:
    """Create the frozen U2 stack with action-effect context outermost."""

    environment = lessons.make_training_env(
        scheduler=scheduler,
        seed=seed,
        size=9,
        seed_access=seed_access,
        forbidden_layout_hashes=forbidden_layout_hashes,
        max_layout_resample_attempts=frozen_u2s.LAYOUT_RESAMPLE_ATTEMPTS,
    )
    return ActionEffectObservation(environment, mode=mode)


def _validate_dict_environment(vector_environment: Any) -> None:
    observation_space = vector_environment.observation_space
    if not isinstance(observation_space, gym.spaces.Dict):
        raise ActionEffectProtocolError("v0.3 requires a Dict vector observation")
    if set(observation_space.spaces) != {IMAGE_KEY, ACTION_EFFECT_KEY}:
        raise ActionEffectProtocolError("v0.3 Dict observation keys changed")
    image = observation_space.spaces[IMAGE_KEY]
    context = observation_space.spaces[ACTION_EFFECT_KEY]
    if (
        not isinstance(image, gym.spaces.Box)
        or tuple(image.shape) != (3, 56, 56)
        or image.dtype != np.uint8
        or not isinstance(context, gym.spaces.Box)
        or context.shape != (ACTION_EFFECT_DIM,)
        or context.dtype != np.float32
    ):
        raise ActionEffectProtocolError("v0.3 vector observation schema changed")


def _validate_transplanted_model(model: Any) -> None:
    policy = model.policy
    extractor = policy.features_extractor
    parameter_names = set(dict(policy.named_parameters()))
    expected_new = "features_extractor.action_effect_encoder.weight"
    if (
        int(model.action_space.n) != 7
        or int(policy.lstm_actor.input_size) != 512
        or int(policy.lstm_actor.hidden_size) != 256
        or int(policy.lstm_actor.num_layers) != 1
        or policy.lstm_critic is None
        or int(policy.lstm_critic.input_size) != 512
        or int(policy.lstm_critic.hidden_size) != 256
        or tuple(policy.action_net.weight.shape) != (7, 256)
        or tuple(policy.value_net.weight.shape) != (1, 256)
        or len(policy.mlp_extractor.policy_net) != 0
        or len(policy.mlp_extractor.value_net) != 0
        or expected_new not in parameter_names
        or tuple(extractor.action_effect_encoder.weight.shape) != (512, ACTION_EFFECT_DIM)
    ):
        raise ActionEffectProtocolError("v0.3 policy topology changed")


def build_transplanted_model(
    vector_environment: Any,
    *,
    parent_checkpoint: Path = PARENT_CHECKPOINT,
    device: str = "cpu",
    equivalence_images: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build one exact zero-context child from the confirmed U1 parent."""

    _validate_dict_environment(vector_environment)
    try:
        from sb3_contrib import RecurrentPPO
        from sb3_contrib.common.recurrent.policies import (
            RecurrentMultiInputActorCriticPolicy,
        )
    except ImportError as error:  # pragma: no cover - dependency message
        raise ActionEffectProtocolError(
            'install training dependencies with: pip install -e ".[train]"'
        ) from error

    legacy = RecurrentPPO.load(parent_checkpoint, device=device)
    model = RecurrentPPO(
        RecurrentMultiInputActorCriticPolicy,
        vector_environment,
        learning_rate=LEARNING_RATE,
        n_steps=ROLLOUT_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=PPO_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENTROPY_COEFFICIENT,
        policy_kwargs=policy_kwargs(),
        seed=ARCHITECTURE_INITIALIZATION_SEED,
        device=device,
        verbose=0,
    )
    transplant = transplant_legacy_model_state(legacy, model)
    _validate_transplanted_model(model)
    if (
        transplant.inherited_timesteps != PARENT_LIFETIME_ACTIONS
        or transplant.inherited_optimizer_updates != PARENT_OPTIMIZER_UPDATES
        or transplant.missing_optimizer_state_names
    ):
        raise ActionEffectProtocolError("v0.3 parent counters or Adam moments changed")
    if equivalence_images is None:
        generator = np.random.default_rng(ARCHITECTURE_INITIALIZATION_SEED)
        equivalence_images = generator.integers(
            0,
            256,
            size=(4, 3, 56, 56),
            dtype=np.uint8,
        )
    equivalence = assert_zero_context_equivalence(
        legacy,
        model,
        equivalence_images,
    )
    model.set_random_seed(ALGORITHM_SEED)
    return model, {
        "architecture_version": ARCHITECTURE_VERSION,
        "transplant": transplant.public_dict(),
        "zero_context_equivalence": equivalence,
        "policy_contract": public_policy_contract(),
    }


@dataclass(frozen=True)
class ArchitectureGrade:
    arm: ActionEffectMode
    eligible: bool
    checks: Mapping[str, bool]
    reasons: tuple[str, ...]
    final_exam_actions: tuple[int, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "eligible": self.eligible,
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
            "final_exam_actions": list(self.final_exam_actions),
        }


def grade_terminal(
    arm: ActionEffectMode | str,
    exam_records: Sequence[Mapping[str, Any]],
) -> ArchitectureGrade:
    """Apply the complete frozen U2-S capability-and-stability gate."""

    selected = ActionEffectMode(arm)
    checks: dict[str, bool] = {
        "exact_exam_count": len(exam_records) == EXAM_COUNT,
        "terminal_boundary": bool(exam_records)
        and int(exam_records[-1].get("child_trained_actions", -1)) == CHILD_ACTION_BUDGET,
        "three_terminal_exams_available": (len(exam_records) >= FINAL_STABILITY_EXAMS),
    }
    boundaries = [int(record.get("child_trained_actions", -1)) for record in exam_records]
    checks["contiguous_exam_boundaries"] = boundaries == [
        index * EVALUATION_INTERVAL for index in range(1, len(exam_records) + 1)
    ]
    terminal = tuple(exam_records[-FINAL_STABILITY_EXAMS:])
    for index, exam in enumerate(terminal, start=1):
        for label, passed in frozen_u2s.exam_stability_checks(exam).items():
            checks[f"terminal_{index}:{label}"] = bool(passed)
    reasons = tuple(label for label, passed in checks.items() if not passed)
    return ArchitectureGrade(
        arm=selected,
        eligible=not reasons,
        checks=MappingProxyType(checks),
        reasons=reasons,
        final_exam_actions=tuple(
            int(record.get("child_trained_actions", -1)) for record in terminal
        ),
    )


def select_architecture(
    arm_exam_records: Mapping[
        ActionEffectMode | str,
        Sequence[Mapping[str, Any]],
    ],
) -> dict[str, Any]:
    """Select only the action-effect definition when its own gate passes."""

    normalized = {
        ActionEffectMode(name): tuple(records) for name, records in arm_exam_records.items()
    }
    if set(normalized) != set(ARM_ORDER):
        raise ActionEffectProtocolError("architecture decision requires both matched arms")
    grades = {arm: grade_terminal(arm, normalized[arm]) for arm in ARM_ORDER}
    candidate_passed = grades[ActionEffectMode.ACTION_EFFECT].eligible
    return {
        "protocol": PROTOCOL,
        "selection_kind": "architecture_definition_only",
        "selected_architecture": (
            ARM_SPECS[ActionEffectMode.ACTION_EFFECT].public_dict() if candidate_passed else None
        ),
        "sham_is_calibration_only": True,
        "development_checkpoint_reuse_authorized": False,
        "replication_protocol_authorized": candidate_passed,
        "u3_authorized": False,
        "grades": {arm.value: grades[arm].public_dict() for arm in ARM_ORDER},
        "verdict": ("architecture_selected" if candidate_passed else "architecture_failed"),
    }


def effective_config(mode: ActionEffectMode | str) -> dict[str, Any]:
    selected = ActionEffectMode(mode)
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "development_only": True,
        "matched_architecture_study": True,
        "arm": ARM_SPECS[selected].public_dict(),
        "parent": {
            "source_protocol": "dungeon-apprentice-v0.2-u1",
            "u1_child_seed": PARENT_U1_SEED,
            "checkpoint": str(PARENT_CHECKPOINT),
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "policy_tensor_sha256": PARENT_POLICY_TENSOR_SHA256,
            "optimizer_state_sha256": PARENT_OPTIMIZER_STATE_SHA256,
            "optimizer_inherited_by_parameter_name": True,
            "lifetime_actions": PARENT_LIFETIME_ACTIONS,
            "optimizer_updates": PARENT_OPTIMIZER_UPDATES,
        },
        "paired_randomness": {
            "architecture_initialization_seed": (ARCHITECTURE_INITIALIZATION_SEED),
            "algorithm_seed": ALGORITHM_SEED,
            "worker_streams": list(WORKER_STREAMS),
            "same_across_both_arms": True,
        },
        "budget": {
            "new_child_actions": CHILD_ACTION_BUDGET,
            "rollout_transitions": ROLLOUT_TRANSITIONS,
            "exam_interval": EVALUATION_INTERVAL,
            "exam_count": EXAM_COUNT,
            "terminal_exam_actions": [
                CHILD_ACTION_BUDGET - 2 * EVALUATION_INTERVAL,
                CHILD_ACTION_BUDGET - EVALUATION_INTERVAL,
                CHILD_ACTION_BUDGET,
            ],
        },
        "observation": {
            "image": {
                "kind": "egocentric_partial_rgb",
                "shape_hwc": [56, 56, 3],
            },
            "action_effect": {
                "shape": [ACTION_EFFECT_DIM],
                "previous_action_one_hot": 7,
                "visible_outcome_one_hot": ["changed", "unchanged"],
                "reset_sentinel": "all_zero",
                "sham_always_zero": (selected is ActionEffectMode.SHAM),
                "derived_only_from_policy_visible_pixels": True,
            },
            "recurrent_state_units": 256,
            "coordinates": False,
            "objects": False,
            "inventory_labels": False,
            "oracle_state": False,
            "mission_text": False,
            "trainer_info": False,
        },
        "policy": public_policy_contract(),
        "optimization": {
            "algorithm": "RecurrentPPO",
            "rollout_steps": ROLLOUT_STEPS,
            "batch_size": BATCH_SIZE,
            "n_epochs": PPO_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "clip_range": CLIP_RANGE,
            "entropy_coefficient": ENTROPY_COEFFICIENT,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "changed_from_original_u2_control": False,
        },
        "reward": {
            "changed_from_original_u2_control": False,
            "no_effect_penalty": 0.0,
            "evaluation_shaping": False,
        },
        "selection": {
            "gate": "complete frozen U2-S terminal-three gate",
            "sham_selectable": False,
            "development_checkpoint_reuse_authorized": False,
            "u3_authorized_by_stage_a": False,
        },
    }
