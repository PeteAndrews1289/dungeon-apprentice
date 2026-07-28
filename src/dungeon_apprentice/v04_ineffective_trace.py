"""v0.4 matched ineffective-trace architecture study.

This protocol tests one deliberately narrow change after the failed v0.3
action-effect study.  Two fresh development twins begin at the same
authenticated, confirmed U1 parent:

``trace-sham``
    Receives the new Dict observation and zero-initialized scalar pathway, but
    its trace value is always zero.
``ineffective-trace``
    Receives one bounded scalar derived only from its own action history and
    consecutive policy-visible RGB observations.  The scalar is
    ``min(streak, 9) / 9`` for runs of identical actions whose RGB result is
    byte-identical. The first unchanged transition starts a run at one; an
    unchanged action switch starts a fresh run at one.

The failed v0.3 nine-dimensional action/outcome input is not retained.  Reward,
PPO, curriculum, lesson mechanics, action budget, and evaluation gates remain
at the original U2 control settings.  Neither v0.3 policy state nor any later
checkpoint is a parent.  A passing candidate can authorize only a separately
versioned replication protocol; development checkpoints are never promotable.
"""

from __future__ import annotations

import copy
import hashlib
import json
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
from dungeon_apprentice.action_streak import (
    INEFFECTIVE_TRACE_DIM,
    INEFFECTIVE_TRACE_KEY,
    IneffectiveTraceMode,
    IneffectiveTraceNatureCNN,
    IneffectiveTraceObservation,
    IneffectiveTracePredictAdapter,
)

PROTOCOL = "dungeon-apprentice-v0.4-ineffective-trace-architecture"
SCHEMA_VERSION = 1
ARCHITECTURE_VERSION = "bounded-ineffective-trace-v1"

IMAGE_KEY = "image"
TRACE_CAP = 9
TRACE_NORMALIZATION_DIVISOR = 9.0
TRACE_ENCODER_ATTRIBUTE = "ineffective_trace_encoder"
TRACE_ENCODER_PARAMETER = f"features_extractor.{TRACE_ENCODER_ATTRIBUTE}.weight"

PARENT_CHILD_SEED = 20260745
PARENT_U1_SEED = 20260733
PARENT_CHECKPOINT = frozen_u2.FROZEN_PARENTS[PARENT_CHILD_SEED].archive
PARENT_CHECKPOINT_SHA256 = "3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104"
PARENT_POLICY_TENSOR_SHA256 = "e555d3f7e2364f74e3b43371938c2f25e2ddbf62558a810038509a70868e6888"
PARENT_OPTIMIZER_STATE_SHA256 = "cc07791b374620d680e5cbb3602d59ab83eb55f195d26808abc0adf5f0e3bc5f"
PARENT_LIFETIME_ACTIONS = 786_432
PARENT_OPTIMIZER_UPDATES = 1_536

ARCHITECTURE_INITIALIZATION_SEED = 20260761
ALGORITHM_SEED = 20260762
WORKER_STREAMS = (20260762, 20260763, 20260764, 20260765)
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
PROTOCOL_DOCUMENT = "docs/protocol-v0.4-ineffective-trace-architecture.md"
FIRST_ROLLOUT_DIGEST_PROFILE = "u2s-canonical-json-v1-lf"

ARM_ORDER = (
    IneffectiveTraceMode.ZERO_TRACE,
    IneffectiveTraceMode.STREAK_TRACE,
)


class IneffectiveTraceProtocolError(RuntimeError):
    """Raised when the v0.4 study changes or loses its frozen boundary."""


def first_rollout_identity_sha256(value: Any) -> str:
    """Hash first-rollout identity evidence with the frozen LF profile."""

    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ArmSpec:
    mode: IneffectiveTraceMode
    trace_enabled: bool
    selectable: bool

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.mode.value,
            "architecture_version": ARCHITECTURE_VERSION,
            "ineffective_trace_enabled": self.trace_enabled,
            "selectable": self.selectable,
            "reward_shaping": False,
            "ppo_changed": False,
        }


ARM_SPECS: Mapping[IneffectiveTraceMode, ArmSpec] = MappingProxyType(
    {
        IneffectiveTraceMode.ZERO_TRACE: ArmSpec(
            mode=IneffectiveTraceMode.ZERO_TRACE,
            trace_enabled=False,
            selectable=False,
        ),
        IneffectiveTraceMode.STREAK_TRACE: ArmSpec(
            mode=IneffectiveTraceMode.STREAK_TRACE,
            trace_enabled=True,
            selectable=True,
        ),
    }
)


def policy_kwargs() -> dict[str, Any]:
    """Return the only policy topology allowed by this protocol."""

    if IneffectiveTraceNatureCNN is None:
        raise IneffectiveTraceProtocolError(
            'install training dependencies with: pip install -e ".[train]"'
        )
    return {
        "features_extractor_class": IneffectiveTraceNatureCNN,
        "features_extractor_kwargs": {"features_dim": 512},
        "net_arch": [],
        # Every inherited tensor is overwritten by the exact U1 transplant.
        # Disabling the policy-level orthogonal pass preserves the required
        # exact-zero scalar residual.
        "ortho_init": False,
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,
    }


def public_policy_contract() -> dict[str, Any]:
    return {
        "policy_class": "RecurrentMultiInputActorCriticPolicy",
        "feature_extractor": ("dungeon_apprentice.action_streak.IneffectiveTraceNatureCNN"),
        "visual_features": 512,
        "ineffective_trace_features": INEFFECTIVE_TRACE_DIM,
        "trace_encoder_parameter": TRACE_ENCODER_PARAMETER,
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
    mode: IneffectiveTraceMode | str,
) -> gym.Env:
    """Create the frozen U2 stack with ineffective trace outermost."""

    environment = lessons.make_training_env(
        scheduler=scheduler,
        seed=seed,
        size=9,
        seed_access=seed_access,
        forbidden_layout_hashes=forbidden_layout_hashes,
        max_layout_resample_attempts=frozen_u2s.LAYOUT_RESAMPLE_ATTEMPTS,
    )
    return IneffectiveTraceObservation(environment, mode=mode)


def _validate_dict_environment(vector_environment: Any) -> None:
    observation_space = vector_environment.observation_space
    if not isinstance(observation_space, gym.spaces.Dict):
        raise IneffectiveTraceProtocolError("v0.4 requires a Dict vector observation")
    if set(observation_space.spaces) != {IMAGE_KEY, INEFFECTIVE_TRACE_KEY}:
        raise IneffectiveTraceProtocolError("v0.4 Dict observation keys changed")
    image = observation_space.spaces[IMAGE_KEY]
    trace = observation_space.spaces[INEFFECTIVE_TRACE_KEY]
    if (
        not isinstance(image, gym.spaces.Box)
        or tuple(image.shape) != (3, 56, 56)
        or image.dtype != np.uint8
        or not isinstance(trace, gym.spaces.Box)
        or trace.shape != (INEFFECTIVE_TRACE_DIM,)
        or trace.dtype != np.float32
    ):
        raise IneffectiveTraceProtocolError("v0.4 vector observation schema changed")


@dataclass(frozen=True)
class TransplantEvidence:
    """Audit record for the confirmed-U1-to-v0.4 transplant."""

    inherited_parameter_names: tuple[str, ...]
    new_parameter_names: tuple[str, ...]
    inherited_optimizer_state_names: tuple[str, ...]
    missing_optimizer_state_names: tuple[str, ...]
    inherited_timesteps: int
    inherited_optimizer_updates: int
    trace_encoder_zero: bool

    def public_dict(self) -> dict[str, Any]:
        return {
            "inherited_parameter_names": list(self.inherited_parameter_names),
            "new_parameter_names": list(self.new_parameter_names),
            "inherited_optimizer_state_names": list(self.inherited_optimizer_state_names),
            "missing_optimizer_state_names": list(self.missing_optimizer_state_names),
            "inherited_timesteps": self.inherited_timesteps,
            "inherited_optimizer_updates": self.inherited_optimizer_updates,
            "trace_encoder_zero": self.trace_encoder_zero,
        }


def transplant_legacy_model_state(
    legacy_model: Any,
    ineffective_trace_model: Any,
) -> TransplantEvidence:
    """Copy every confirmed-U1 policy tensor and named Adam moment exactly."""

    try:
        import torch as th
    except ImportError as error:  # pragma: no cover - dependency message
        raise IneffectiveTraceProtocolError(
            'install training dependencies with: pip install -e ".[train]"'
        ) from error

    legacy_named = dict(legacy_model.policy.named_parameters())
    new_named = dict(ineffective_trace_model.policy.named_parameters())
    missing = sorted(set(legacy_named) - set(new_named))
    if missing:
        raise IneffectiveTraceProtocolError(
            f"ineffective-trace policy lost U1 parameters: {missing}"
        )
    new_only = tuple(sorted(set(new_named) - set(legacy_named)))
    expected_new = (TRACE_ENCODER_PARAMETER,)
    if new_only != expected_new:
        raise IneffectiveTraceProtocolError(
            f"ineffective-trace policy has an unexpected parameter delta: {list(new_only)}"
        )

    with th.no_grad():
        for name, source in legacy_named.items():
            destination = new_named[name]
            if source.shape != destination.shape:
                raise IneffectiveTraceProtocolError(
                    f"U1 parameter shape changed for {name}: "
                    f"{tuple(source.shape)} != {tuple(destination.shape)}"
                )
            destination.copy_(
                source.detach().to(
                    device=destination.device,
                    dtype=destination.dtype,
                )
            )
        new_named[TRACE_ENCODER_PARAMETER].zero_()

    legacy_optimizer = legacy_model.policy.optimizer
    new_optimizer = ineffective_trace_model.policy.optimizer
    if len(legacy_optimizer.param_groups) != 1 or len(new_optimizer.param_groups) != 1:
        raise IneffectiveTraceProtocolError(
            "ineffective-trace transplant requires one optimizer group"
        )
    for key, value in legacy_optimizer.param_groups[0].items():
        if key != "params":
            new_optimizer.param_groups[0][key] = copy.deepcopy(value)

    inherited_states: list[str] = []
    missing_states: list[str] = []
    new_optimizer.state.clear()
    for name, source in legacy_named.items():
        source_state = legacy_optimizer.state.get(source)
        if source_state is None:
            missing_states.append(name)
            continue
        destination = new_named[name]
        copied: dict[str, Any] = {}
        for key, value in source_state.items():
            copied[key] = (
                value.detach().clone().to(destination.device)
                if th.is_tensor(value)
                else copy.deepcopy(value)
            )
        new_optimizer.state[destination] = copied
        inherited_states.append(name)

    ineffective_trace_model.num_timesteps = int(legacy_model.num_timesteps)
    ineffective_trace_model._n_updates = int(legacy_model._n_updates)
    trace_zero = bool(th.count_nonzero(new_named[TRACE_ENCODER_PARAMETER].detach()).item() == 0)
    if not trace_zero:
        raise IneffectiveTraceProtocolError(
            "ineffective-trace residual is not zero after transplant"
        )
    return TransplantEvidence(
        inherited_parameter_names=tuple(sorted(legacy_named)),
        new_parameter_names=new_only,
        inherited_optimizer_state_names=tuple(sorted(inherited_states)),
        missing_optimizer_state_names=tuple(sorted(missing_states)),
        inherited_timesteps=int(ineffective_trace_model.num_timesteps),
        inherited_optimizer_updates=int(ineffective_trace_model._n_updates),
        trace_encoder_zero=trace_zero,
    )


def assert_zero_trace_equivalence(
    legacy_model: Any,
    ineffective_trace_model: Any,
    images: np.ndarray,
) -> dict[str, Any]:
    """Prove exact deterministic behavior at the U1 transplant boundary."""

    try:
        import torch as th
        from sb3_contrib.common.recurrent.type_aliases import RNNStates
    except ImportError as error:  # pragma: no cover - dependency message
        raise IneffectiveTraceProtocolError(
            'install training dependencies with: pip install -e ".[train]"'
        ) from error

    values = np.asarray(images)
    if values.ndim != 4 or tuple(values.shape[1:]) != (3, 56, 56):
        raise ValueError("equivalence images must have shape (batch, 3, 56, 56)")
    batch = int(values.shape[0])
    if legacy_model.device != ineffective_trace_model.device:
        raise IneffectiveTraceProtocolError("equivalence models must use the same device")
    image_tensor = th.as_tensor(values, device=legacy_model.device)
    trace_tensor = th.zeros(
        (batch, INEFFECTIVE_TRACE_DIM),
        dtype=th.float32,
        device=ineffective_trace_model.device,
    )
    hidden_size = int(legacy_model.policy.lstm_actor.hidden_size)
    layers = int(legacy_model.policy.lstm_actor.num_layers)

    def state() -> tuple[th.Tensor, th.Tensor]:
        shape = (layers, batch, hidden_size)
        return (
            th.zeros(shape, device=legacy_model.device),
            th.zeros(shape, device=legacy_model.device),
        )

    states = RNNStates(state(), state())
    episode_starts = th.zeros(
        (batch,),
        dtype=th.float32,
        device=legacy_model.device,
    )
    episode_starts[0] = 1.0
    with th.no_grad():
        legacy_features = legacy_model.policy.extract_features(image_tensor)
        trace_features = ineffective_trace_model.policy.extract_features(
            {
                IMAGE_KEY: image_tensor,
                INEFFECTIVE_TRACE_KEY: trace_tensor,
            }
        )
        legacy_output = legacy_model.policy.forward(
            image_tensor,
            states,
            episode_starts,
            deterministic=True,
        )
        trace_output = ineffective_trace_model.policy.forward(
            {
                IMAGE_KEY: image_tensor,
                INEFFECTIVE_TRACE_KEY: trace_tensor,
            },
            states,
            episode_starts,
            deterministic=True,
        )
    tensors = (
        ("features", legacy_features, trace_features),
        ("actions", legacy_output[0], trace_output[0]),
        ("values", legacy_output[1], trace_output[1]),
        ("log_probabilities", legacy_output[2], trace_output[2]),
    )
    mismatched = [label for label, left, right in tensors if not th.equal(left, right)]
    legacy_states = tuple(tensor for branch in legacy_output[3] for tensor in branch)
    trace_states = tuple(tensor for branch in trace_output[3] for tensor in branch)
    if any(
        not th.equal(left, right) for left, right in zip(legacy_states, trace_states, strict=True)
    ):
        mismatched.append("recurrent_states")
    if mismatched:
        raise IneffectiveTraceProtocolError(
            f"zero-context transplant changed U1 behavior: {mismatched}"
        )
    return {
        "batch": batch,
        "features_exact": True,
        "actions_exact": True,
        "values_exact": True,
        "log_probabilities_exact": True,
        "recurrent_states_exact": True,
    }


def _validate_transplanted_model(model: Any) -> None:
    policy = model.policy
    extractor = policy.features_extractor
    parameter_names = set(dict(policy.named_parameters()))
    trace_encoder = getattr(extractor, TRACE_ENCODER_ATTRIBUTE, None)
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
        or TRACE_ENCODER_PARAMETER not in parameter_names
        or trace_encoder is None
        or tuple(trace_encoder.weight.shape) != (512, INEFFECTIVE_TRACE_DIM)
    ):
        raise IneffectiveTraceProtocolError("v0.4 policy topology changed")


def build_transplanted_model(
    vector_environment: Any,
    *,
    parent_checkpoint: Path = PARENT_CHECKPOINT,
    device: str = "cpu",
    equivalence_images: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build one exact trace-sham child from the confirmed U1 parent."""

    _validate_dict_environment(vector_environment)
    try:
        from sb3_contrib import RecurrentPPO
        from sb3_contrib.common.recurrent.policies import (
            RecurrentMultiInputActorCriticPolicy,
        )
    except ImportError as error:  # pragma: no cover - dependency message
        raise IneffectiveTraceProtocolError(
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
        raise IneffectiveTraceProtocolError("v0.4 parent counters or Adam moments changed")
    if equivalence_images is None:
        generator = np.random.default_rng(ARCHITECTURE_INITIALIZATION_SEED)
        equivalence_images = generator.integers(
            0,
            256,
            size=(4, 3, 56, 56),
            dtype=np.uint8,
        )
    equivalence = assert_zero_trace_equivalence(
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
    arm: IneffectiveTraceMode
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
    arm: IneffectiveTraceMode | str,
    exam_records: Sequence[Mapping[str, Any]],
) -> ArchitectureGrade:
    """Apply the complete frozen U2-S capability-and-stability gate."""

    selected = IneffectiveTraceMode(arm)
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
        IneffectiveTraceMode | str,
        Sequence[Mapping[str, Any]],
    ],
) -> dict[str, Any]:
    """Select only the ineffective-trace definition when its gate passes."""

    normalized = {
        IneffectiveTraceMode(name): tuple(records) for name, records in arm_exam_records.items()
    }
    if set(normalized) != set(ARM_ORDER):
        raise IneffectiveTraceProtocolError("architecture decision requires both matched arms")
    grades = {arm: grade_terminal(arm, normalized[arm]) for arm in ARM_ORDER}
    candidate = IneffectiveTraceMode.STREAK_TRACE
    candidate_passed = grades[candidate].eligible
    return {
        "protocol": PROTOCOL,
        "selection_kind": "architecture_definition_only",
        "selected_architecture": (ARM_SPECS[candidate].public_dict() if candidate_passed else None),
        "trace_sham_is_calibration_only": True,
        "development_checkpoint_reuse_authorized": False,
        "replication_protocol_authorized": candidate_passed,
        "separately_versioned_replication_required": candidate_passed,
        "u3_authorized": False,
        "grades": {arm.value: grades[arm].public_dict() for arm in ARM_ORDER},
        "verdict": ("architecture_selected" if candidate_passed else "architecture_failed"),
    }


def effective_config(
    mode: IneffectiveTraceMode | str,
) -> dict[str, Any]:
    """Return the complete learner-facing configuration for one arm."""

    selected = IneffectiveTraceMode(mode)
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "development_only": True,
        "matched_architecture_study": True,
        "arm": ARM_SPECS[selected].public_dict(),
        "parent": {
            "source_protocol": "dungeon-apprentice-v0.2-u1",
            "u1_child_seed": PARENT_U1_SEED,
            "source_child_seed": PARENT_CHILD_SEED,
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
            "ineffective_trace": {
                "shape": [INEFFECTIVE_TRACE_DIM],
                "dtype": "float32",
                "definition": ("min(consecutive_same_action_byte_identical_rgb_transitions,9)/9"),
                "clip_streak_at": TRACE_CAP,
                "normalization_divisor": TRACE_NORMALIZATION_DIVISOR,
                "range": [0.0, 1.0],
                "transition_rule": {
                    "episode_start": 0,
                    "visible_change": 0,
                    "first_unchanged_transition": 1,
                    "action_switch_with_unchanged_rgb": 1,
                    "same_action_unchanged_rgb": "min(previous_count+1,9)",
                },
                "trace_sham_always_zero": (selected is IneffectiveTraceMode.ZERO_TRACE),
                "derived_only_from_policy_visible_pixels_and_actions": True,
            },
            "retains_v03_nine_dimensional_action_effect": False,
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
            "trace_sham_selectable": False,
            "candidate_definition_only": True,
            "separately_versioned_replication_required": True,
            "development_checkpoint_reuse_authorized": False,
            "u3_authorized_by_stage_a": False,
        },
    }


__all__ = [
    "ALGORITHM_SEED",
    "ARCHITECTURE_INITIALIZATION_SEED",
    "ARCHITECTURE_VERSION",
    "ARM_ORDER",
    "ARM_SPECS",
    "CHILD_ACTION_BUDGET",
    "EVALUATION_INTERVAL",
    "EXAM_COUNT",
    "FINAL_STABILITY_EXAMS",
    "FIRST_ROLLOUT_DIGEST_PROFILE",
    "INEFFECTIVE_TRACE_DIM",
    "INEFFECTIVE_TRACE_KEY",
    "PARENT_CHECKPOINT",
    "PARENT_CHECKPOINT_SHA256",
    "PARENT_CHILD_SEED",
    "PARENT_LIFETIME_ACTIONS",
    "PARENT_OPTIMIZER_STATE_SHA256",
    "PARENT_OPTIMIZER_UPDATES",
    "PARENT_POLICY_TENSOR_SHA256",
    "PARENT_U1_SEED",
    "PROTOCOL",
    "TRACE_CAP",
    "TRACE_ENCODER_PARAMETER",
    "TRACE_NORMALIZATION_DIVISOR",
    "WORKER_STREAMS",
    "IneffectiveTraceMode",
    "IneffectiveTraceObservation",
    "IneffectiveTracePredictAdapter",
    "IneffectiveTraceProtocolError",
    "assert_zero_trace_equivalence",
    "build_transplanted_model",
    "effective_config",
    "first_rollout_identity_sha256",
    "grade_terminal",
    "make_training_environment",
    "policy_kwargs",
    "public_policy_contract",
    "select_architecture",
    "transplant_legacy_model_state",
]
