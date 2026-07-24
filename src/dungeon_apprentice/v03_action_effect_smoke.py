"""Disposable matched first-rollout smoke for v0.3 action-effect.

This command is an engineering gate, not scientific evidence.  It creates two
temporary twins from the exact confirmed U1 archive, sends them through one
real four-worker 2,048-transition rollout, performs one ordinary PPO phase,
reloads both updated archives, and removes every mutable artifact.

The decisive pre-update assertion is that ``sham`` and ``action-effect``
produce the same action/environment trajectory.  Their observations differ,
but the context residual is exactly zero, so no behavioral divergence is
allowed before learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2r as frozen_u2r
from dungeon_apprentice import v02_u2s as frozen_u2s
from dungeon_apprentice import v02_u2s_smoke as frozen_smoke
from dungeon_apprentice import v03_action_effect as v03
from dungeon_apprentice.action_effect import (
    ActionEffectMode,
    ActionEffectObservation,
)
from dungeon_apprentice.artifacts import file_sha256, git_snapshot, utc_now

SMOKE_PROTOCOL = "dungeon-apprentice-v0.3-action-effect-disposable-smoke"
SCHEMA_VERSION = 1
TOTAL_TRANSITIONS = v03.ROLLOUT_TRANSITIONS


class ActionEffectSmokeError(RuntimeError):
    """Raised when the disposable matched gate fails closed."""


@dataclass(frozen=True)
class ActionEffectSmokeInputs:
    """The exact seed issuer and static history guard used by both twins."""

    seed_access: Any
    forbidden_layout_hashes: Mapping[lessons.LessonId, frozenset[str]]


def _guard_mapping_sha256(
    mapping: Mapping[lessons.LessonId, frozenset[str]],
) -> str:
    if set(mapping) != set(lessons.LessonId):
        raise ActionEffectSmokeError(
            "v0.3 smoke requires the complete static history guard"
        )
    return frozen_u2s._canonical_sha256(
        {
            lesson.value: sorted(mapping[lesson])
            for lesson in lessons.LessonId
        }
    )


def _effect_weight_nonzero(model: Any) -> int:
    weight = model.policy.features_extractor.action_effect_encoder.weight
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
            f"v0.3 {type(space).__name__}",
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
    contexts: list[ActionEffectObservation],
    phase: str,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency message
        raise ActionEffectSmokeError(
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
    mode: ActionEffectMode,
    *,
    arm_directory: Path,
    inputs: ActionEffectSmokeInputs,
) -> dict[str, Any]:
    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            VecTransposeImage,
        )
    except ImportError as error:  # pragma: no cover - dependency message
        raise ActionEffectSmokeError(
            'install training dependencies with: pip install -e ".[train]"'
        ) from error

    arm_directory.mkdir()
    parent_copy = arm_directory / "confirmed-u1-parent.zip"
    shutil.copy2(v03.PARENT_CHECKPOINT, parent_copy)
    if file_sha256(parent_copy) != v03.PARENT_CHECKPOINT_SHA256:
        raise ActionEffectSmokeError("disposable U1 parent copy changed")

    state = lessons.CurriculumState()
    scheduler = lessons.TransitionDeficitScheduler(
        state,
        seed=v03.ALGORITHM_SEED + 90_000,
    )
    episode_wrappers: list[frozen_u2s.EpisodeEvidenceWrapper] = []
    audit_wrappers: list[frozen_smoke.TransitionEvidenceWrapper] = []
    context_wrappers: list[ActionEffectObservation] = []
    factories: list[Callable[[], gym.Env]] = []
    ledger = arm_directory / "episode-starts.jsonl"
    for worker_index, worker_stream in enumerate(v03.WORKER_STREAMS):

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
            contextual = ActionEffectObservation(audit, mode=mode)
            episode_wrappers.append(episode)
            audit_wrappers.append(audit)
            context_wrappers.append(contextual)
            return contextual

        factories.append(factory)

    vector_environment: Any | None = None
    try:
        from stable_baselines3.common.callbacks import BaseCallback

        vector_environment = VecTransposeImage(DummyVecEnv(factories))
        model, transplant = v03.build_transplanted_model(
            vector_environment,
            parent_checkpoint=parent_copy,
            device="cpu",
        )
        before = frozen_smoke.model_fingerprint(model)
        if (
            before.trained_timesteps != v03.PARENT_LIFETIME_ACTIONS
            or before.optimizer_updates != v03.PARENT_OPTIMIZER_UPDATES
            or _effect_weight_nonzero(model) != 0
        ):
            raise ActionEffectSmokeError(
                "disposable model did not begin at the transplant boundary"
            )
        model.set_random_seed(v03.ALGORITHM_SEED)
        ordered_episodes = sorted(
            episode_wrappers,
            key=lambda value: value.worker_index,
        )
        ordered_contexts = sorted(
            zip(episode_wrappers, context_wrappers, strict=True),
            key=lambda value: value[0].worker_index,
        )
        if (
            len(ordered_episodes) != v03.WORKERS
            or [worker.worker_index for worker in ordered_episodes]
            != list(range(v03.WORKERS))
            or [worker.worker_stream for worker in ordered_episodes]
            != list(v03.WORKER_STREAMS)
            or any(
                worker.episode_ordinal
                or worker.local_transition_count
                or worker._active is not None
                for worker in ordered_episodes
            )
        ):
            raise ActionEffectSmokeError(
                "v0.3 RNG spaces were not ready before reset"
            )
        for episode, context in ordered_contexts:
            context.action_space.seed(episode.worker_stream)
            context.observation_space.seed(episode.worker_stream)

        class FirstRolloutEvidenceCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)
                self._policy_digest = hashlib.sha256()
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
        rollout_count, restore = frozen_smoke._install_single_rollout_limit(
            model
        )
        try:
            model.learn(
                total_timesteps=v03.CHILD_ACTION_BUDGET,
                callback=evidence_callback,
                reset_num_timesteps=False,
                progress_bar=False,
                log_interval=None,
            )
        finally:
            restore()
        if rollout_count() != 1:
            raise ActionEffectSmokeError(
                "smoke did not complete exactly one rollout"
            )
        if (
            evidence_callback.steps != v03.ROLLOUT_STEPS
            or evidence_callback.pre_action_rng_identity is None
            or evidence_callback.policy_output_sha256 is None
            or evidence_callback.post_rollout_rng_identity is None
        ):
            raise ActionEffectSmokeError(
                "smoke did not seal its pre-update rollout boundary"
            )
        after = frozen_smoke.model_fingerprint(model)
        if (
            after.trained_timesteps - before.trained_timesteps
            != TOTAL_TRANSITIONS
            or after.optimizer_updates - before.optimizer_updates
            != v03.PPO_EPOCHS
            or after.policy_tensor_sha256 == before.policy_tensor_sha256
            or after.optimizer_state_sha256
            == before.optimizer_state_sha256
        ):
            raise ActionEffectSmokeError(
                "smoke did not complete one ordinary PPO phase"
            )
        effect_nonzero = _effect_weight_nonzero(model)
        if mode is ActionEffectMode.SHAM and effect_nonzero:
            raise ActionEffectSmokeError(
                "zero-input sham changed its context projection"
            )
        if (
            mode is ActionEffectMode.ACTION_EFFECT
            and not effect_nonzero
        ):
            raise ActionEffectSmokeError(
                "action-effect projection received no first-rollout update"
            )

        updated_archive = arm_directory / "updated-disposable-policy.zip"
        model.save(updated_archive)
        reloaded = RecurrentPPO.load(updated_archive, device="cpu")
        if frozen_smoke.model_fingerprint(reloaded) != after:
            raise ActionEffectSmokeError(
                "updated action-effect archive did not reload exactly"
            )
        if file_sha256(parent_copy) != v03.PARENT_CHECKPOINT_SHA256:
            raise ActionEffectSmokeError("smoke mutated its parent copy")
        workers = _normalized_worker_evidence(audit_wrappers)
        if (
            len(workers) != v03.WORKERS
            or sum(int(value["transitions"]) for value in workers)
            != TOTAL_TRANSITIONS
        ):
            raise ActionEffectSmokeError(
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
            "effect_projection_nonzero_parameters": effect_nonzero,
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
        raise ActionEffectSmokeError(
            "v0.3 smoke requires a clean source tree"
        )
    if (
        not v03.PARENT_CHECKPOINT.is_file()
        or file_sha256(v03.PARENT_CHECKPOINT)
        != v03.PARENT_CHECKPOINT_SHA256
    ):
        raise ActionEffectSmokeError(
            "confirmed U1 parent is absent or changed"
        )
    if (seed_access is None) != (forbidden_layout_hashes is None):
        raise ActionEffectSmokeError(
            "v0.3 smoke seed access and static guard must be supplied together"
        )
    if seed_access is None:
        # Standalone engineering use retains the historical loader. Canonical
        # qualification supplies the extended v0.3 mapping explicitly and
        # binds its digest below.
        legacy_inputs = frozen_smoke._load_protocol_inputs(repository)
        inputs = ActionEffectSmokeInputs(
            seed_access=legacy_inputs.seed_access,
            forbidden_layout_hashes=legacy_inputs.forbidden_layout_hashes,
        )
    else:
        assert forbidden_layout_hashes is not None
        inputs = ActionEffectSmokeInputs(
            seed_access=seed_access,
            forbidden_layout_hashes=forbidden_layout_hashes,
        )
    guard_mapping_sha256 = _guard_mapping_sha256(
        inputs.forbidden_layout_hashes
    )
    sampler_preflight = frozen_u2r.preflight_u2r_training_layout_sampler(
        inputs.forbidden_layout_hashes,
        seed_access=inputs.seed_access,
        worker_streams=v03.WORKER_STREAMS,
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
        prefix="dungeon-apprentice-v03-effect-smoke-"
    ) as temporary:
        root = Path(temporary).resolve()
        temporary_path = str(root)
        for mode in v03.ARM_ORDER:
            results[mode.value] = _run_arm(
                mode,
                arm_directory=root / mode.value,
                inputs=inputs,
            )
    if temporary_path is None or Path(temporary_path).exists():
        raise ActionEffectSmokeError(
            "disposable v0.3 smoke root was not removed"
        )
    canonical_after = frozen_smoke._canonical_root_snapshots()
    if canonical_after != canonical_before:
        raise ActionEffectSmokeError(
            "v0.3 smoke changed a predecessor canonical root"
        )
    sham = results[ActionEffectMode.SHAM.value]
    candidate = results[ActionEffectMode.ACTION_EFFECT.value]
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
    ):
        raise ActionEffectSmokeError(
            "matched twins diverged before the first architecture update"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": SMOKE_PROTOCOL,
        "completed_at": utc_now(),
        "verdict": "passed",
        "source": source,
        "parent_checkpoint": str(v03.PARENT_CHECKPOINT),
        "parent_checkpoint_sha256": v03.PARENT_CHECKPOINT_SHA256,
        "worker_streams": list(v03.WORKER_STREAMS),
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
        "arms": results,
        "canonical_predecessor_roots_unchanged": True,
        "temporary_root_removed": True,
        "scientific_evidence": False,
        "checkpoint_reuse_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the disposable v0.3 matched architecture smoke."
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
