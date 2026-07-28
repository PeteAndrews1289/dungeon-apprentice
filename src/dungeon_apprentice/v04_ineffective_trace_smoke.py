"""Disposable matched first-rollout smoke for v0.4 ineffective-trace.

This command is an engineering gate, not scientific evidence.  It creates two
temporary twins from the exact confirmed U1 archive, sends them through one
real four-worker 2,048-transition rollout, performs one ordinary PPO phase,
reloads both updated archives, and removes every mutable artifact.

The decisive pre-update assertion is that ``trace-sham`` and ``ineffective-trace``
produce the same action/environment trajectory.  Their observations differ,
but the trace residual is exactly zero, so no behavioral divergence is
allowed before learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2r as frozen_u2r
from dungeon_apprentice import v02_u2s as frozen_u2s
from dungeon_apprentice import v02_u2s_smoke as frozen_smoke
from dungeon_apprentice import v04_ineffective_trace as v04
from dungeon_apprentice.action_streak import (
    INEFFECTIVE_TRACE_KEY,
    IneffectiveTraceMode,
    IneffectiveTraceObservation,
)
from dungeon_apprentice.artifacts import file_sha256, git_snapshot, utc_now

SMOKE_PROTOCOL = "dungeon-apprentice-v0.4-ineffective-trace-disposable-smoke"
SCHEMA_VERSION = 1
TOTAL_TRANSITIONS = v04.ROLLOUT_TRANSITIONS


class IneffectiveTraceSmokeError(RuntimeError):
    """Raised when the disposable matched gate fails closed."""


@dataclass(frozen=True)
class IneffectiveTraceSmokeInputs:
    """The exact seed issuer and static history guard used by both twins."""

    seed_access: Any
    forbidden_layout_hashes: Mapping[lessons.LessonId, frozenset[str]]


@dataclass
class TraceExerciseMetrics:
    """Descriptive evidence that the learner-facing scalar was exercised."""

    transitions: int = 0
    visible_changed: int = 0
    visible_unchanged: int = 0
    raw_trace_nonzero_transitions: int = 0
    raw_trace_max: int = 0
    raw_trace_histogram: Counter[int] = field(default_factory=Counter)
    exposed_trace_nonzero_transitions: int = 0
    exposed_trace_max: float = 0.0

    def observe(
        self,
        *,
        infos: tuple[Mapping[str, Any], ...],
        observations: Mapping[str, Any],
        mode: IneffectiveTraceMode,
    ) -> None:
        traces = np.asarray(observations[INEFFECTIVE_TRACE_KEY])
        if traces.shape != (len(infos), 1) or traces.dtype != np.float32:
            raise IneffectiveTraceSmokeError(
                "smoke received a noncanonical trace observation"
            )
        for index, info in enumerate(infos):
            count = int(info.get("ineffective_trace_raw_count", -1))
            changed = bool(info.get("ineffective_trace_visible_changed"))
            enabled = bool(info.get("ineffective_trace_context_active"))
            terminal = info.get("terminal_observation")
            if isinstance(terminal, Mapping):
                exposed_array = np.asarray(
                    terminal.get(INEFFECTIVE_TRACE_KEY)
                )
                if exposed_array.shape != (1,):
                    raise IneffectiveTraceSmokeError(
                        "terminal trace observation changed shape"
                    )
                exposed = float(exposed_array[0])
            else:
                exposed = float(traces[index, 0])
            expected = (
                0.0
                if mode is IneffectiveTraceMode.ZERO_TRACE
                else min(count, v04.TRACE_CAP) / float(v04.TRACE_CAP)
            )
            if (
                count < 0
                or count > v04.TRACE_CAP
                or enabled is not (mode is IneffectiveTraceMode.STREAK_TRACE)
                or not np.isfinite(exposed)
                or not np.isclose(exposed, expected, rtol=0.0, atol=1e-7)
            ):
                raise IneffectiveTraceSmokeError(
                    "trace exercise evidence disagrees with the frozen encoding"
                )
            self.transitions += 1
            self.visible_changed += int(changed)
            self.visible_unchanged += int(not changed)
            self.raw_trace_nonzero_transitions += int(count > 0)
            self.raw_trace_max = max(self.raw_trace_max, count)
            self.raw_trace_histogram[count] += 1
            self.exposed_trace_nonzero_transitions += int(exposed > 0.0)
            self.exposed_trace_max = max(self.exposed_trace_max, exposed)

    def public_dict(self, *, mode: IneffectiveTraceMode) -> dict[str, Any]:
        return {
            "transitions": self.transitions,
            "visible_changed": self.visible_changed,
            "visible_unchanged": self.visible_unchanged,
            "raw_trace_nonzero_transitions": (
                self.raw_trace_nonzero_transitions
            ),
            "raw_trace_max": self.raw_trace_max,
            "raw_trace_histogram": {
                str(count): self.raw_trace_histogram[count]
                for count in sorted(self.raw_trace_histogram)
            },
            "exposed_trace_nonzero_transitions": (
                self.exposed_trace_nonzero_transitions
            ),
            "exposed_trace_max": self.exposed_trace_max,
            "trace_enabled": mode is IneffectiveTraceMode.STREAK_TRACE,
            "descriptive_only": True,
        }


@dataclass
class TraceGradientEvidence:
    """Observe every gradient delivered to the new encoder parameter."""

    observations: int = 0
    nonzero_observations: int = 0
    nonzero_parameters_max: int = 0

    def observe(self, gradient: Any) -> Any:
        nonzero = int(gradient.detach().count_nonzero().cpu().item())
        self.observations += 1
        self.nonzero_observations += int(nonzero > 0)
        self.nonzero_parameters_max = max(
            self.nonzero_parameters_max,
            nonzero,
        )
        return gradient


def _trace_encoder_update_evidence(
    model: Any,
    *,
    gradients: TraceGradientEvidence,
) -> dict[str, Any]:
    """Return exact-zero/nonzero evidence for the new weight and Adam state."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency message
        raise IneffectiveTraceSmokeError(
            'install training dependencies with: pip install -e ".[train]"'
        ) from error
    weight = model.policy.features_extractor.ineffective_trace_encoder.weight
    state = model.policy.optimizer.state.get(weight)
    if not isinstance(state, Mapping):
        raise IneffectiveTraceSmokeError(
            "trace encoder has no Adam state after the first optimizer"
        )
    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    if not torch.is_tensor(exp_avg) or not torch.is_tensor(exp_avg_sq):
        raise IneffectiveTraceSmokeError(
            "trace encoder Adam moments are missing"
        )
    weight_nonzero = int(weight.detach().count_nonzero().cpu().item())
    exp_avg_nonzero = int(exp_avg.detach().count_nonzero().cpu().item())
    exp_avg_sq_nonzero = int(
        exp_avg_sq.detach().count_nonzero().cpu().item()
    )
    return {
        "weight_nonzero_parameters": weight_nonzero,
        "gradient_observations": gradients.observations,
        "gradient_nonzero_observations": gradients.nonzero_observations,
        "gradient_nonzero_parameters_max": (
            gradients.nonzero_parameters_max
        ),
        "adam_state_present": True,
        "adam_exp_avg_nonzero_parameters": exp_avg_nonzero,
        "adam_exp_avg_sq_nonzero_parameters": exp_avg_sq_nonzero,
        "weight_exact_zero": weight_nonzero == 0,
        "adam_moments_exact_zero": (
            exp_avg_nonzero == 0 and exp_avg_sq_nonzero == 0
        ),
    }


def _guard_mapping_sha256(
    mapping: Mapping[lessons.LessonId, frozenset[str]],
) -> str:
    if set(mapping) != set(lessons.LessonId):
        raise IneffectiveTraceSmokeError(
            "v0.4 smoke requires the complete static history guard"
        )
    return frozen_u2s._canonical_sha256(
        {
            lesson.value: sorted(mapping[lesson])
            for lesson in lessons.LessonId
        }
    )


def _trace_weight_nonzero(model: Any) -> int:
    weight = model.policy.features_extractor.ineffective_trace_encoder.weight
    return int(weight.detach().count_nonzero().cpu().item())


def _normalized_worker_evidence(
    wrappers: list[frozen_smoke.TransitionEvidenceWrapper],
) -> list[dict[str, Any]]:
    return [
        wrapper.public_dict()
        for wrapper in sorted(
            wrappers,
            key=lambda value: value.worker_index,
        )
    ]


def _trajectory_identity(
    worker_evidence: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "worker_index": int(value["worker_index"]),
            "transitions": int(value["transitions"]),
            "episodes_started": int(value["episodes_started"]),
            "episode_starts_sha256": str(
                value["episode_starts_sha256"]
            ),
            "action_counts": dict(value["action_counts"]),
            "lesson_transition_counts": dict(
                value["lesson_transition_counts"]
            ),
            "trajectory_sha256": str(value["trajectory_sha256"]),
            "reward_evidence_sha256": str(
                value["reward_evidence_sha256"]
            ),
        }
        for value in worker_evidence
    ]


def _update_array_digest(digest: Any, label: str, value: Any) -> None:
    try:
        array = value.detach().cpu().numpy()
    except AttributeError:
        array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    digest.update(label.encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            list(contiguous.shape),
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(contiguous.tobytes())


def _space_rng_identity(space: gym.Space) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": type(space).__name__,
        "self_sha256": frozen_u2s._space_rng_state_sha256(
            space,
            f"v0.4 {type(space).__name__}",
        ),
    }
    if isinstance(space, gym.spaces.Dict):
        value["children"] = {
            key: _space_rng_identity(child)
            for key, child in space.spaces.items()
        }
    return value


def _rng_identity(
    *,
    model: Any,
    scheduler: lessons.TransitionDeficitScheduler,
    episodes: list[frozen_u2s.EpisodeEvidenceWrapper],
    contexts: list[IneffectiveTraceObservation],
    phase: str,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency message
        raise IneffectiveTraceSmokeError(
            'install training dependencies with: pip install -e ".[train]"'
        ) from error
    ordered = sorted(
        zip(episodes, contexts, strict=True),
        key=lambda value: value[0].worker_index,
    )
    workers: list[dict[str, Any]] = []
    for episode, context in ordered:
        curriculum = frozen_u2s._curriculum_environment(episode)
        workers.append(
            {
                "worker_index": episode.worker_index,
                "worker_stream": episode.worker_stream,
                "curriculum_sha256": frozen_u2s._rng_state_sha256(
                    curriculum._rng.bit_generator.state
                ),
                "base_environment_sha256": (
                    frozen_u2s._rng_state_sha256(
                        curriculum.unwrapped.np_random.bit_generator.state
                    )
                ),
                "action_space": _space_rng_identity(
                    context.action_space
                ),
                "observation_space": _space_rng_identity(
                    context.observation_space
                ),
            }
        )
    components = {
        "python_random_sha256": frozen_u2s._rng_state_sha256(
            random.getstate()
        ),
        "numpy_global_sha256": frozen_u2s._rng_state_sha256(
            np.random.get_state()
        ),
        "torch_cpu_sha256": frozen_u2s._torch_rng_state_sha256(
            torch.random.get_rng_state()
        ),
        "model_action_space": _space_rng_identity(model.action_space),
        "scheduler_sha256": frozen_u2s._rng_state_sha256(
            scheduler._rng.bit_generator.state
        ),
        "workers": workers,
    }
    return {
        "phase": phase,
        "components": components,
        "aggregate_sha256": frozen_u2s._canonical_sha256(components),
    }


def _run_arm(
    mode: IneffectiveTraceMode,
    *,
    arm_directory: Path,
    inputs: IneffectiveTraceSmokeInputs,
) -> dict[str, Any]:
    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            VecTransposeImage,
        )
    except ImportError as error:  # pragma: no cover - dependency message
        raise IneffectiveTraceSmokeError(
            'install training dependencies with: pip install -e ".[train]"'
        ) from error

    arm_directory.mkdir()
    parent_copy = arm_directory / "confirmed-u1-parent.zip"
    shutil.copy2(v04.PARENT_CHECKPOINT, parent_copy)
    if file_sha256(parent_copy) != v04.PARENT_CHECKPOINT_SHA256:
        raise IneffectiveTraceSmokeError("disposable U1 parent copy changed")

    state = lessons.CurriculumState()
    scheduler = lessons.TransitionDeficitScheduler(
        state,
        seed=v04.ALGORITHM_SEED + 90_000,
    )
    episode_wrappers: list[frozen_u2s.EpisodeEvidenceWrapper] = []
    audit_wrappers: list[frozen_smoke.TransitionEvidenceWrapper] = []
    context_wrappers: list[IneffectiveTraceObservation] = []
    factories: list[Callable[[], gym.Env]] = []
    ledger = arm_directory / "episode-starts.jsonl"
    for worker_index, worker_stream in enumerate(v04.WORKER_STREAMS):

        def factory(
            index: int = worker_index,
            stream: int = worker_stream,
        ) -> gym.Env:
            environment = lessons.make_training_env(
                scheduler=scheduler,
                seed=stream,
                size=9,
                seed_access=inputs.seed_access,
                forbidden_layout_hashes=inputs.forbidden_layout_hashes,
                max_layout_resample_attempts=(
                    frozen_u2s.LAYOUT_RESAMPLE_ATTEMPTS
                ),
            )
            episode = frozen_u2s.EpisodeEvidenceWrapper(
                environment,
                worker_index=index,
                worker_stream=stream,
                ledger=ledger,
            )
            audit = frozen_smoke.TransitionEvidenceWrapper(
                episode,
                worker_index=index,
            )
            contextual = IneffectiveTraceObservation(audit, mode=mode)
            episode_wrappers.append(episode)
            audit_wrappers.append(audit)
            context_wrappers.append(contextual)
            return contextual

        factories.append(factory)

    vector_environment: Any | None = None
    try:
        from stable_baselines3.common.callbacks import BaseCallback

        vector_environment = VecTransposeImage(DummyVecEnv(factories))
        model, transplant = v04.build_transplanted_model(
            vector_environment,
            parent_checkpoint=parent_copy,
            device="cpu",
        )
        before = frozen_smoke.model_fingerprint(model)
        if (
            before.trained_timesteps != v04.PARENT_LIFETIME_ACTIONS
            or before.optimizer_updates != v04.PARENT_OPTIMIZER_UPDATES
            or _trace_weight_nonzero(model) != 0
        ):
            raise IneffectiveTraceSmokeError(
                "disposable model did not begin at the transplant boundary"
            )
        model.set_random_seed(v04.ALGORITHM_SEED)
        ordered_episodes = sorted(
            episode_wrappers,
            key=lambda value: value.worker_index,
        )
        ordered_contexts = sorted(
            zip(episode_wrappers, context_wrappers, strict=True),
            key=lambda value: value[0].worker_index,
        )
        if (
            len(ordered_episodes) != v04.WORKERS
            or [worker.worker_index for worker in ordered_episodes]
            != list(range(v04.WORKERS))
            or [worker.worker_stream for worker in ordered_episodes]
            != list(v04.WORKER_STREAMS)
            or any(
                worker.episode_ordinal
                or worker.local_transition_count
                or worker._active is not None
                for worker in ordered_episodes
            )
        ):
            raise IneffectiveTraceSmokeError(
                "v0.4 RNG spaces were not ready before reset"
            )
        for episode, context in ordered_contexts:
            context.action_space.seed(episode.worker_stream)
            context.observation_space.seed(episode.worker_stream)

        class FirstRolloutEvidenceCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)
                self._policy_digest = hashlib.sha256()
                self.trace_exercise = TraceExerciseMetrics()
                self.steps = 0
                self.pre_action_rng_identity: dict[str, Any] | None = None
                self.policy_output_sha256: str | None = None
                self.post_rollout_rng_identity: dict[str, Any] | None = None

            def _on_training_start(self) -> None:
                self.pre_action_rng_identity = _rng_identity(
                    model=self.model,
                    scheduler=scheduler,
                    episodes=episode_wrappers,
                    contexts=context_wrappers,
                    phase="post_reset_pre_action_one",
                )

            def _on_step(self) -> bool:
                self.steps += 1
                infos = tuple(self.locals.get("infos", ()))
                observations = self.locals.get("new_obs")
                if (
                    not all(isinstance(info, Mapping) for info in infos)
                    or not isinstance(observations, Mapping)
                ):
                    raise IneffectiveTraceSmokeError(
                        "smoke lost trace transition evidence"
                    )
                self.trace_exercise.observe(
                    infos=infos,
                    observations=observations,
                    mode=mode,
                )
                _update_array_digest(
                    self._policy_digest,
                    "actions",
                    self.locals["actions"],
                )
                _update_array_digest(
                    self._policy_digest,
                    "values",
                    self.locals["values"],
                )
                _update_array_digest(
                    self._policy_digest,
                    "log_probs",
                    self.locals["log_probs"],
                )
                states = self.locals["lstm_states"]
                for branch_name, branch in (
                    ("pi", states.pi),
                    ("vf", states.vf),
                ):
                    _update_array_digest(
                        self._policy_digest,
                        f"{branch_name}_hidden",
                        branch[0],
                    )
                    _update_array_digest(
                        self._policy_digest,
                        f"{branch_name}_cell",
                        branch[1],
                    )
                return True

            def _on_rollout_end(self) -> None:
                self.policy_output_sha256 = (
                    self._policy_digest.hexdigest()
                )
                self.post_rollout_rng_identity = (
                    _rng_identity(
                        model=self.model,
                        scheduler=scheduler,
                        episodes=episode_wrappers,
                        contexts=context_wrappers,
                        phase="post_rollout_pre_optimizer",
                    )
                )

        evidence_callback = FirstRolloutEvidenceCallback()
        gradients = TraceGradientEvidence()
        trace_weight = (
            model.policy.features_extractor
            .ineffective_trace_encoder.weight
        )
        gradient_hook = trace_weight.register_hook(gradients.observe)
        rollout_count, restore = frozen_smoke._install_single_rollout_limit(
            model
        )
        try:
            model.learn(
                total_timesteps=v04.CHILD_ACTION_BUDGET,
                callback=evidence_callback,
                reset_num_timesteps=False,
                progress_bar=False,
                log_interval=None,
            )
        finally:
            gradient_hook.remove()
            restore()
        if rollout_count() != 1:
            raise IneffectiveTraceSmokeError(
                "smoke did not complete exactly one rollout"
            )
        if (
            evidence_callback.steps != v04.ROLLOUT_STEPS
            or evidence_callback.pre_action_rng_identity is None
            or evidence_callback.policy_output_sha256 is None
            or evidence_callback.post_rollout_rng_identity is None
        ):
            raise IneffectiveTraceSmokeError(
                "smoke did not seal its pre-update rollout boundary"
            )
        after = frozen_smoke.model_fingerprint(model)
        if (
            after.trained_timesteps - before.trained_timesteps
            != TOTAL_TRANSITIONS
            or after.optimizer_updates - before.optimizer_updates
            != v04.PPO_EPOCHS
            or after.policy_tensor_sha256 == before.policy_tensor_sha256
            or after.optimizer_state_sha256
            == before.optimizer_state_sha256
        ):
            raise IneffectiveTraceSmokeError(
                "smoke did not complete one ordinary PPO phase"
            )
        trace_exercise = evidence_callback.trace_exercise.public_dict(
            mode=mode
        )
        if trace_exercise["transitions"] != TOTAL_TRANSITIONS:
            raise IneffectiveTraceSmokeError(
                "trace exercise did not cover the complete first rollout"
            )
        encoder_update = _trace_encoder_update_evidence(
            model,
            gradients=gradients,
        )
        if mode is IneffectiveTraceMode.ZERO_TRACE:
            if (
                trace_exercise["exposed_trace_nonzero_transitions"] != 0
                or not encoder_update["weight_exact_zero"]
                or not encoder_update["adam_moments_exact_zero"]
                or encoder_update["gradient_nonzero_observations"] != 0
            ):
                raise IneffectiveTraceSmokeError(
                    "zero-input sham changed its trace pathway"
                )
        elif (
            trace_exercise["raw_trace_nonzero_transitions"] <= 0
            or trace_exercise["exposed_trace_nonzero_transitions"] <= 0
            or encoder_update["weight_exact_zero"]
            or encoder_update["adam_moments_exact_zero"]
            or encoder_update["gradient_nonzero_observations"] <= 0
        ):
            raise IneffectiveTraceSmokeError(
                "candidate did not exercise and learn from the trace"
            )

        updated_archive = arm_directory / "updated-disposable-policy.zip"
        model.save(updated_archive)
        reloaded = RecurrentPPO.load(updated_archive, device="cpu")
        if frozen_smoke.model_fingerprint(reloaded) != after:
            raise IneffectiveTraceSmokeError(
                "updated ineffective-trace archive did not reload exactly"
            )
        if file_sha256(parent_copy) != v04.PARENT_CHECKPOINT_SHA256:
            raise IneffectiveTraceSmokeError("smoke mutated its parent copy")
        workers = _normalized_worker_evidence(audit_wrappers)
        if (
            len(workers) != v04.WORKERS
            or sum(int(value["transitions"]) for value in workers)
            != TOTAL_TRANSITIONS
        ):
            raise IneffectiveTraceSmokeError(
                "smoke worker transition evidence is incomplete"
            )
        ledger_records = frozen_smoke._read_normalized_episode_ledger(
            ledger
        )
        seed_evidence = frozen_smoke._validate_seed_evidence(
            [
                reset
                for wrapper in audit_wrappers
                for reset in wrapper.resets
            ]
        )
        return {
            "arm": mode.value,
            "before": before.public_dict(),
            "after": after.public_dict(),
            "trace_projection_nonzero_parameters": (
                encoder_update["weight_nonzero_parameters"]
            ),
            "trace_exercise": trace_exercise,
            "encoder_update": encoder_update,
            "transplant": transplant,
            "workers": workers,
            "trajectory_identity": _trajectory_identity(workers),
            "pre_action_rng_identity": (
                evidence_callback.pre_action_rng_identity
            ),
            "policy_output_sha256": (
                evidence_callback.policy_output_sha256
            ),
            "post_rollout_rng_identity": (
                evidence_callback.post_rollout_rng_identity
            ),
            "episode_ledger": {
                "records": len(ledger_records),
                "normalized_sha256": (
                    frozen_smoke._canonical_sha256(ledger_records)
                ),
            },
            "seed_evidence": seed_evidence,
            "updated_archive_reloaded_exactly": True,
            "development_checkpoint_reuse_authorized": False,
        }
    finally:
        if vector_environment is not None:
            vector_environment.close()


def run_smoke(
    *,
    repository: Path,
    require_clean_source: bool = True,
    seed_access: Any | None = None,
    forbidden_layout_hashes: (
        Mapping[lessons.LessonId, frozenset[str]] | None
    ) = None,
) -> dict[str, Any]:
    source = git_snapshot(repository)
    if require_clean_source and source.get("dirty") is not False:
        raise IneffectiveTraceSmokeError(
            "v0.4 smoke requires a clean source tree"
        )
    if (
        not v04.PARENT_CHECKPOINT.is_file()
        or file_sha256(v04.PARENT_CHECKPOINT)
        != v04.PARENT_CHECKPOINT_SHA256
    ):
        raise IneffectiveTraceSmokeError(
            "confirmed U1 parent is absent or changed"
        )
    if (seed_access is None) != (forbidden_layout_hashes is None):
        raise IneffectiveTraceSmokeError(
            "v0.4 smoke seed access and static guard must be supplied together"
        )
    if seed_access is None:
        # Standalone engineering use retains the historical loader. Canonical
        # qualification supplies the extended v0.4 mapping explicitly and
        # binds its digest below.
        legacy_inputs = frozen_smoke._load_protocol_inputs(repository)
        inputs = IneffectiveTraceSmokeInputs(
            seed_access=legacy_inputs.seed_access,
            forbidden_layout_hashes=legacy_inputs.forbidden_layout_hashes,
        )
    else:
        assert forbidden_layout_hashes is not None
        inputs = IneffectiveTraceSmokeInputs(
            seed_access=seed_access,
            forbidden_layout_hashes=forbidden_layout_hashes,
        )
    guard_mapping_sha256 = _guard_mapping_sha256(
        inputs.forbidden_layout_hashes
    )
    sampler_preflight = frozen_u2r.preflight_u2r_training_layout_sampler(
        inputs.forbidden_layout_hashes,
        seed_access=inputs.seed_access,
        worker_streams=v04.WORKER_STREAMS,
        max_attempts=frozen_u2s.LAYOUT_RESAMPLE_ATTEMPTS,
    )
    sampler_preflight_public = [dict(value) for value in sampler_preflight]
    sampler_preflight_sha256 = frozen_u2s._canonical_sha256(
        sampler_preflight_public
    )
    canonical_before = frozen_smoke._canonical_root_snapshots()
    results: dict[str, dict[str, Any]] = {}
    temporary_path: str | None = None
    with tempfile.TemporaryDirectory(
        prefix="dungeon-apprentice-v04-trace-smoke-"
    ) as temporary:
        root = Path(temporary).resolve()
        temporary_path = str(root)
        for mode in v04.ARM_ORDER:
            results[mode.value] = _run_arm(
                mode,
                arm_directory=root / mode.value,
                inputs=inputs,
            )
    if temporary_path is None or Path(temporary_path).exists():
        raise IneffectiveTraceSmokeError(
            "disposable v0.4 smoke root was not removed"
        )
    for result in results.values():
        result["updated_archive_destroyed"] = True
    canonical_after = frozen_smoke._canonical_root_snapshots()
    if canonical_after != canonical_before:
        raise IneffectiveTraceSmokeError(
            "v0.4 smoke changed a predecessor canonical root"
        )
    sham = results[IneffectiveTraceMode.ZERO_TRACE.value]
    candidate = results[IneffectiveTraceMode.STREAK_TRACE.value]
    raw_trace_fields = (
        "transitions",
        "visible_changed",
        "visible_unchanged",
        "raw_trace_nonzero_transitions",
        "raw_trace_max",
        "raw_trace_histogram",
    )
    if (
        sham["before"] != candidate["before"]
        or sham["pre_action_rng_identity"]
        != candidate["pre_action_rng_identity"]
        or sham["trajectory_identity"]
        != candidate["trajectory_identity"]
        or sham["policy_output_sha256"]
        != candidate["policy_output_sha256"]
        or sham["post_rollout_rng_identity"]
        != candidate["post_rollout_rng_identity"]
        or sham["episode_ledger"]["normalized_sha256"]
        != candidate["episode_ledger"]["normalized_sha256"]
        or any(
            sham["trace_exercise"][field]
            != candidate["trace_exercise"][field]
            for field in raw_trace_fields
        )
    ):
        raise IneffectiveTraceSmokeError(
            "matched twins diverged before the first architecture update"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": SMOKE_PROTOCOL,
        "completed_at": utc_now(),
        "verdict": "passed",
        "source": source,
        "parent_checkpoint": str(v04.PARENT_CHECKPOINT),
        "parent_checkpoint_sha256": v04.PARENT_CHECKPOINT_SHA256,
        "worker_streams": list(v04.WORKER_STREAMS),
        "guard_mapping_sha256": guard_mapping_sha256,
        "sampler_preflight": sampler_preflight_public,
        "sampler_preflight_sha256": sampler_preflight_sha256,
        "actions_per_arm": TOTAL_TRANSITIONS,
        "pre_action_rng_identical": True,
        "pre_update_behavior_identical": True,
        "first_rollout_trajectory_identical": True,
        "first_rollout_policy_outputs_identical": True,
        "post_rollout_pre_optimizer_rng_identical": True,
        "first_rollout_episode_ledger_identical": True,
        "learning_divergence_begins_after_first_update": True,
        "raw_trace_evidence_identical": True,
        "arms": results,
        "canonical_predecessor_roots_unchanged": True,
        "temporary_root_removed": True,
        "updated_archives_reloaded_exactly": all(
            result["updated_archive_reloaded_exactly"]
            for result in results.values()
        ),
        "updated_archives_destroyed": all(
            result["updated_archive_destroyed"]
            for result in results.values()
        ),
        "scientific_evidence": False,
        "checkpoint_reuse_authorized": False,
        "r3_checkpoint_or_state_reused": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the disposable v0.4 matched architecture smoke."
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Engineering-only local use; the report records dirty source.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_smoke(
        repository=args.repository.expanduser().resolve(),
        require_clean_source=not args.allow_dirty,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
