"""Disposable one-rollout integration smoke for the U2-S four-arm ablation.

This is an engineering release gate, not a scientific run.  It authenticates
the exact confirmed U1 parent and the prospective U2-S layout exclusions, then
loads a separate copy of that parent for each of the four frozen arms.  Every
copy collects exactly one production-size 2,048-transition recurrent-PPO
rollout through the real four-worker curriculum and performs its configured
optimizer phase.

All mutable files live under an operating-system temporary directory.  The
command emits canonical JSON only after that directory has been removed and
after proving that the claim-bearing parent and all canonical U2-S roots are
unchanged.  Its updated policies are deliberately non-promotable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from dungeon_apprentice import v02_u1_confirm as state_digests
from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice import v02_u2r as frozen_u2r
from dungeon_apprentice import v02_u2s
from dungeon_apprentice.artifacts import file_sha256, git_snapshot
from dungeon_apprentice.u2_seed_guard import SEED_PARTITIONS, U2SeedRole

SMOKE_PROTOCOL = "dungeon-apprentice-v0.2-u2s-disposable-integration-smoke"
SCHEMA_VERSION = 1
TOTAL_TRANSITIONS = v02_u2s.ROLLOUT_TRANSITIONS

CANONICAL_QUALIFICATION_ROOT = Path(
    "/Volumes/T7 Developer/DungeonApprentice/qualifications/v0.2-u2s-20260723"
)
CANONICAL_ROOTS = (
    CANONICAL_QUALIFICATION_ROOT,
    v02_u2s.CANONICAL_COHORT_ROOT,
    v02_u2s.CANONICAL_MEDIA_ROOT,
)

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_PROTECTED_ROLES = frozenset(
    partition.role
    for partition in SEED_PARTITIONS
    if "confirmation" in partition.role.value
    or partition.role is U2SeedRole.FINAL_TEST
)


class U2SSmokeError(RuntimeError):
    """Raised when disposable U2-S integration evidence fails closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in _SHA256_CHARACTERS for character in digest
    ):
        raise U2SSmokeError(f"{label} is not a lowercase SHA-256 digest")
    return digest


def _float_hex(value: Any, label: str) -> str:
    measured = float(value)
    if not math.isfinite(measured):
        raise U2SSmokeError(f"{label} is not finite")
    return measured.hex()


def _observation_sha256(observation: Any) -> str:
    array = np.asarray(observation)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _path_snapshot(path: Path) -> dict[str, Any]:
    """Return a content-only snapshot without following symlinks."""

    target = path.expanduser().absolute()
    if target.is_symlink():
        raise U2SSmokeError(f"canonical smoke boundary is a symlink: {target}")
    if not target.exists():
        return {"path": str(target), "exists": False}
    metadata = target.lstat()
    if stat.S_ISREG(metadata.st_mode):
        return {
            "path": str(target),
            "exists": True,
            "kind": "file",
            "bytes": int(metadata.st_size),
            "sha256": file_sha256(target),
        }
    if not stat.S_ISDIR(metadata.st_mode):
        raise U2SSmokeError(f"canonical smoke boundary has unsafe type: {target}")
    entries: list[dict[str, Any]] = []
    for child in sorted(target.rglob("*"), key=lambda item: item.as_posix()):
        relative = child.relative_to(target).as_posix()
        child_metadata = child.lstat()
        if stat.S_ISLNK(child_metadata.st_mode):
            raise U2SSmokeError(
                f"canonical smoke boundary contains a symlink: {child}"
            )
        if stat.S_ISDIR(child_metadata.st_mode):
            entries.append({"path": relative, "kind": "directory"})
        elif stat.S_ISREG(child_metadata.st_mode):
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "bytes": int(child_metadata.st_size),
                    "sha256": file_sha256(child),
                }
            )
        else:
            raise U2SSmokeError(
                f"canonical smoke boundary contains an unsafe entry: {child}"
            )
    return {
        "path": str(target),
        "exists": True,
        "kind": "directory",
        "entries": len(entries),
        "tree_sha256": _canonical_sha256(entries),
    }


def _canonical_root_snapshots() -> list[dict[str, Any]]:
    return [_path_snapshot(path) for path in CANONICAL_ROOTS]


def _assert_disposable_root(path: Path) -> None:
    root = path.expanduser().resolve()
    for canonical in CANONICAL_ROOTS:
        protected = canonical.expanduser().absolute()
        if root == protected or root in protected.parents or protected in root.parents:
            raise U2SSmokeError(
                "disposable smoke root overlaps a canonical U2-S root"
            )


@dataclass(frozen=True)
class ModelFingerprint:
    trained_timesteps: int
    optimizer_updates: int
    policy_tensor_sha256: str
    optimizer_state_sha256: str
    optimizer_has_state: bool

    def public_dict(self) -> dict[str, Any]:
        return {
            "trained_timesteps": self.trained_timesteps,
            "optimizer_updates": self.optimizer_updates,
            "policy_tensor_sha256": self.policy_tensor_sha256,
            "optimizer_state_sha256": self.optimizer_state_sha256,
            "optimizer_has_state": self.optimizer_has_state,
        }


def model_fingerprint(model: Any) -> ModelFingerprint:
    optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
    return ModelFingerprint(
        trained_timesteps=int(model.num_timesteps),
        optimizer_updates=int(model._n_updates),
        policy_tensor_sha256=_require_sha256(
            state_digests.policy_tensor_sha256(model),
            "smoke policy tensor state",
        ),
        optimizer_state_sha256=_require_sha256(
            state_digests.optimizer_state_sha256(model),
            "smoke optimizer state",
        ),
        optimizer_has_state=bool(optimizer is not None and optimizer.state),
    )


def _verify_parent_fingerprint(value: ModelFingerprint) -> None:
    if (
        value.trained_timesteps != v02_u2s.PARENT_LIFETIME_ACTIONS
        or value.optimizer_updates != v02_u2s.PARENT_OPTIMIZER_UPDATES
        or value.policy_tensor_sha256
        != v02_u2s.PARENT_POLICY_TENSOR_SHA256
        or value.optimizer_state_sha256
        != v02_u2s.PARENT_OPTIMIZER_STATE_SHA256
        or not value.optimizer_has_state
    ):
        raise U2SSmokeError("disposable arm did not load the exact U1 model state")


class TransitionEvidenceWrapper(gym.Wrapper):
    """Collect deterministic transition and reward summaries around a smoke worker."""

    def __init__(self, env: gym.Env, *, worker_index: int) -> None:
        super().__init__(env)
        self.worker_index = int(worker_index)
        self.transitions = 0
        self.resets: list[dict[str, Any]] = []
        self.action_counts: Counter[int] = Counter()
        self.lesson_transition_counts: Counter[str] = Counter()
        self.reward_total = 0.0
        self.extrinsic_total = 0.0
        self.curiosity_total = 0.0
        self.penalty_total = 0.0
        self.penalty_events = 0
        self.penalty_eligible_events = 0
        self.maximum_episode_penalty_count = 0
        self._previous_observation: np.ndarray | None = None
        self._trajectory_records: list[dict[str, Any]] = []
        self._reward_records: list[dict[str, Any]] = []

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        seed = int(info["seed"])
        lesson_id = str(info["lesson_id"])
        role = info.get("u2_seed_role")
        self.resets.append(
            {
                "worker_index": self.worker_index,
                "episode_ordinal": len(self.resets) + 1,
                "seed": seed,
                "seed_role": str(role) if role is not None else None,
                "lesson_id": lesson_id,
                "layout_sha256": _require_sha256(
                    info.get("layout_sha256"),
                    "smoke reset layout",
                ),
                "geometry_sha256": _require_sha256(
                    info.get("geometry_sha256"),
                    "smoke reset geometry",
                ),
                "worker_transition_at_start": self.transitions,
            }
        )
        self._previous_observation = np.array(observation, copy=True)
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        selected = int(np.asarray(action).item())
        observation, reward, terminated, truncated, info = self.env.step(selected)
        if self._previous_observation is None:
            raise U2SSmokeError("smoke worker stepped before reset")
        extrinsic = float(info.get("extrinsic_reward", reward))
        curiosity = float(info.get("curiosity_reward", 0.0))
        penalty = float(info.get("u2s_no_effect_penalty", 0.0))
        for measured, label in (
            (reward, "reward"),
            (extrinsic, "extrinsic reward"),
            (curiosity, "curiosity reward"),
            (penalty, "no-effect penalty"),
        ):
            if not math.isfinite(float(measured)):
                raise U2SSmokeError(f"smoke {label} is not finite")
        if not math.isclose(
            float(reward),
            extrinsic + curiosity + penalty,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise U2SSmokeError("smoke reward components do not reconcile")
        self.transitions += 1
        self.action_counts[selected] += 1
        lesson_id = str(info["lesson_id"])
        self.lesson_transition_counts[lesson_id] += 1
        self.reward_total += float(reward)
        self.extrinsic_total += extrinsic
        self.curiosity_total += curiosity
        self.penalty_total += penalty
        self.penalty_events += int(penalty < 0.0)
        self.penalty_eligible_events += int(
            info.get("u2s_no_effect_eligible_event_count", 0) > 0
            and int(info.get("u2s_identical_interaction_run", 0)) >= 2
        )
        self.maximum_episode_penalty_count = max(
            self.maximum_episode_penalty_count,
            int(info.get("u2s_no_effect_applied_penalty_count", 0)),
        )
        self._trajectory_records.append(
            {
                "worker_index": self.worker_index,
                "worker_transition": self.transitions,
                "action": selected,
                "lesson_id": lesson_id,
                "before_sha256": _observation_sha256(
                    self._previous_observation
                ),
                "after_sha256": _observation_sha256(observation),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        self._reward_records.append(
            {
                "reward": _float_hex(reward, "smoke reward"),
                "extrinsic": _float_hex(extrinsic, "smoke extrinsic reward"),
                "curiosity": _float_hex(curiosity, "smoke curiosity reward"),
                "penalty": _float_hex(penalty, "smoke no-effect penalty"),
            }
        )
        self._previous_observation = np.array(observation, copy=True)
        return observation, reward, terminated, truncated, info

    def public_dict(self) -> dict[str, Any]:
        return {
            "worker_index": self.worker_index,
            "transitions": self.transitions,
            "episodes_started": len(self.resets),
            "episode_starts_sha256": _canonical_sha256(self.resets),
            "action_counts": {
                str(action): int(self.action_counts.get(action, 0))
                for action in range(7)
            },
            "lesson_transition_counts": {
                lesson.value: int(
                    self.lesson_transition_counts.get(lesson.value, 0)
                )
                for lesson in lessons.LessonId
            },
            "reward_total_hex": self.reward_total.hex(),
            "extrinsic_total_hex": self.extrinsic_total.hex(),
            "curiosity_total_hex": self.curiosity_total.hex(),
            "penalty_total_hex": self.penalty_total.hex(),
            "penalty_events": self.penalty_events,
            "penalty_eligible_events": self.penalty_eligible_events,
            "maximum_episode_penalty_count": (
                self.maximum_episode_penalty_count
            ),
            "trajectory_sha256": _canonical_sha256(
                self._trajectory_records
            ),
            "reward_evidence_sha256": _canonical_sha256(
                self._reward_records
            ),
        }


def _read_normalized_episode_ledger(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise U2SSmokeError("smoke episode ledger record is invalid")
                normalized = dict(value)
                normalized.pop("timestamp", None)
                values.append(normalized)
    except (OSError, json.JSONDecodeError) as error:
        raise U2SSmokeError("cannot authenticate smoke episode ledger") from error
    return values


def _validate_seed_evidence(
    resets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not resets:
        raise U2SSmokeError("smoke produced no episode-start evidence")
    protected_hits: list[dict[str, Any]] = []
    separated_roles: set[str] = set()
    seeds: list[int] = []
    for value in resets:
        seed = int(value["seed"])
        seeds.append(seed)
        if not 0 <= seed < 1_000_000:
            raise U2SSmokeError("smoke opened a non-training layout seed")
        for partition in SEED_PARTITIONS:
            if partition.role in _PROTECTED_ROLES and partition.contains(seed):
                protected_hits.append(
                    {"seed": seed, "role": partition.role.value}
                )
        if value.get("lesson_id") == lessons.LessonId.SEPARATED_UNLOCK.value:
            separated_roles.add(str(value.get("seed_role")))
    if protected_hits:
        raise U2SSmokeError("smoke opened a protected seed partition")
    if separated_roles - {U2SeedRole.TRAINING.value}:
        raise U2SSmokeError("smoke U2 episodes were not training-role episodes")
    return {
        "episode_starts": len(resets),
        "minimum_seed": min(seeds),
        "maximum_seed": max(seeds),
        "training_range_only": True,
        "separated_unlock_roles": sorted(separated_roles),
        "protected_roles": sorted(role.value for role in _PROTECTED_ROLES),
        "protected_seed_hits": [],
        "confirmation_or_final_seed_generated": False,
    }


def _install_single_rollout_limit(
    model: Any,
) -> tuple[Callable[[], int], Callable[[], None]]:
    """Make ``learn`` stop before collecting a second rollout.

    Passing the full scientific child budget to ``learn`` is important: SB3
    derives ``progress_remaining`` from that horizon, and the conservative
    learning-rate schedule must therefore see the same progress value as the
    real first rollout.  Returning ``False`` on the *next* collection attempt
    lets the first rollout train normally without adding another transition.
    """

    original = model.collect_rollouts
    had_instance_value = "collect_rollouts" in vars(model)
    previous_instance_value = vars(model).get("collect_rollouts")
    completed = 0

    def limited_collect_rollouts(*args: Any, **kwargs: Any) -> bool:
        nonlocal completed
        if completed:
            return False
        result = bool(original(*args, **kwargs))
        if not result:
            raise U2SSmokeError("smoke could not complete its first rollout")
        completed += 1
        return True

    model.collect_rollouts = limited_collect_rollouts

    def count() -> int:
        return completed

    def restore() -> None:
        if had_instance_value:
            model.collect_rollouts = previous_instance_value
        else:
            delattr(model, "collect_rollouts")

    return count, restore


@dataclass(frozen=True)
class _ProtocolInputs:
    parent: Any
    seed_access: Any
    forbidden_layout_hashes: Mapping[lessons.LessonId, frozenset[str]]
    guard_mapping_sha256: str
    sampler_preflight_sha256: str


def _load_protocol_inputs(repository: Path) -> _ProtocolInputs:
    """Reauthenticate the inputs needed before U2-S qualification exists."""

    from dungeon_apprentice import v02_u2s_qualify as qualification

    (
        parent,
        base_qualification,
        failed_u2_parent,
        terminal,
        _failed_r0,
    ) = qualification._authenticate_predecessors(repository)
    parent_public = qualification._verify_parent_model(parent)
    if (
        parent_public["checkpoint_sha256"]
        != v02_u2s.PARENT_CHECKPOINT_SHA256
        or parent_public["policy_tensor_sha256"]
        != v02_u2s.PARENT_POLICY_TENSOR_SHA256
        or parent_public["optimizer_state_sha256"]
        != v02_u2s.PARENT_OPTIMIZER_STATE_SHA256
    ):
        raise U2SSmokeError("smoke parent authentication changed")
    access = base_qualification.seed_access()
    forbidden, guards = qualification.build_u2s_forbidden_layout_hashes(
        base_qualification.verified_report(),
        failed_u2_parent.confirmation_snapshot(),
        access=access,
        terminal_report=terminal,
    )
    guard_digest = str(guards["applied_mapping_sha256"])
    if guard_digest != v02_u2s.QUALIFIED_GUARD_MAPPING_SHA256:
        raise U2SSmokeError("smoke guard mapping changed")
    sampler = frozen_u2r.preflight_u2r_training_layout_sampler(
        forbidden,
        seed_access=access,
        worker_streams=v02_u2s.WORKER_STREAMS,
        max_attempts=v02_u2s.LAYOUT_RESAMPLE_ATTEMPTS,
    )
    return _ProtocolInputs(
        parent=parent,
        seed_access=access,
        forbidden_layout_hashes=forbidden,
        guard_mapping_sha256=guard_digest,
        sampler_preflight_sha256=_canonical_sha256(
            [dict(item) for item in sampler]
        ),
    )


def _run_arm(
    arm: v02_u2s.ArmName,
    *,
    arm_directory: Path,
    inputs: _ProtocolInputs,
) -> dict[str, Any]:
    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            VecTransposeImage,
        )
    except ImportError as error:
        raise U2SSmokeError(
            'install training dependencies with: pip install -e ".[train]"'
        ) from error

    arm_directory.mkdir()
    parent_copy = arm_directory / "confirmed-u1-parent.zip"
    shutil.copy2(v02_u2s.PARENT_CHECKPOINT, parent_copy)
    if file_sha256(parent_copy) != v02_u2s.PARENT_CHECKPOINT_SHA256:
        raise U2SSmokeError("disposable parent copy differs from U1")

    state = lessons.CurriculumState()
    scheduler = lessons.TransitionDeficitScheduler(
        state,
        seed=v02_u2s.ALGORITHM_SEED + 90_000,
    )
    episode_wrappers: list[v02_u2s.EpisodeEvidenceWrapper] = []
    audit_wrappers: list[TransitionEvidenceWrapper] = []
    factories: list[Callable[[], gym.Env]] = []
    ledger = arm_directory / "episode-starts.jsonl"
    for worker_index, worker_stream in enumerate(v02_u2s.WORKER_STREAMS):

        def factory(
            index: int = worker_index,
            stream: int = worker_stream,
        ) -> gym.Env:
            environment = v02_u2s._make_training_environment(
                scheduler=scheduler,
                seed=stream,
                seed_access=inputs.seed_access,
                forbidden_layout_hashes=inputs.forbidden_layout_hashes,
                penalize_no_effect=(
                    v02_u2s.ARM_SPECS[arm].no_effect_penalty
                ),
            )
            episode = v02_u2s.EpisodeEvidenceWrapper(
                environment,
                worker_index=index,
                worker_stream=stream,
                ledger=ledger,
            )
            audit = TransitionEvidenceWrapper(episode, worker_index=index)
            episode_wrappers.append(episode)
            audit_wrappers.append(audit)
            return audit

        factories.append(factory)

    vector_environment: Any | None = None
    try:
        vector_environment = VecTransposeImage(DummyVecEnv(factories))
        model = RecurrentPPO.load(
            parent_copy,
            env=vector_environment,
            device="cpu",
        )
        v02_u2s._validate_parent_model_before_arm(model, inputs.parent)
        before = model_fingerprint(model)
        _verify_parent_fingerprint(before)
        v02_u2s._apply_arm_optimization(model, arm)
        v02_u2s._verify_arm_model(
            model,
            arm,
            expected_actions=v02_u2s.PARENT_LIFETIME_ACTIONS,
            expected_updates=v02_u2s.PARENT_OPTIMIZER_UPDATES,
            expected_model_state=None,
        )

        class InitialRngCaptureCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)
                self.identity: dict[str, Any] | None = None

            def _on_training_start(self) -> None:
                self.identity = v02_u2s.capture_initial_rng_identity(
                    model=self.model,
                    scheduler=scheduler,
                    active_workers=episode_wrappers,
                )

            def _on_step(self) -> bool:
                return True

        rng_callback = InitialRngCaptureCallback()
        model.set_random_seed(v02_u2s.ALGORITHM_SEED)
        v02_u2s.seed_initial_rng_spaces(episode_wrappers)
        rollout_count, restore_rollout_collection = (
            _install_single_rollout_limit(model)
        )
        try:
            model.learn(
                total_timesteps=v02_u2s.CHILD_ACTION_BUDGET,
                callback=rng_callback,
                reset_num_timesteps=False,
                progress_bar=False,
                log_interval=None,
            )
        finally:
            restore_rollout_collection()
        if rollout_count() != 1:
            raise U2SSmokeError("smoke did not execute exactly one rollout")
        if rng_callback.identity is None:
            raise U2SSmokeError(
                "smoke did not capture RNG identity before action one"
            )
        initial_rng_identity = rng_callback.identity
        after = model_fingerprint(model)
        child_actions = after.trained_timesteps - before.trained_timesteps
        completed_epochs = after.optimizer_updates - before.optimizer_updates
        if child_actions != TOTAL_TRANSITIONS:
            raise U2SSmokeError("smoke action count differs from one rollout")
        minimum, maximum = v02_u2s.optimizer_update_bounds(
            arm,
            child_actions,
        )
        if not minimum <= completed_epochs <= maximum:
            raise U2SSmokeError("smoke optimizer epoch count is invalid")
        if (
            after.policy_tensor_sha256 == before.policy_tensor_sha256
            or after.optimizer_state_sha256
            == before.optimizer_state_sha256
        ):
            raise U2SSmokeError(
                "smoke optimizer phase did not update its disposable model"
            )
        approx_kl = float(
            model.logger.name_to_value.get("train/approx_kl", math.nan)
        )
        if not math.isfinite(approx_kl):
            raise U2SSmokeError("smoke did not record finite PPO KL evidence")
        epoch_accounting = v02_u2s.optimizer_epoch_accounting(
            arm,
            child_actions,
            completed_epochs,
            approx_kl=approx_kl,
        )
        learning_rates = {
            _float_hex(group["lr"], "smoke optimizer learning rate")
            for group in model.policy.optimizer.param_groups
        }
        if len(learning_rates) != 1:
            raise U2SSmokeError(
                "smoke optimizer groups have inconsistent learning rates"
            )
        measured_learning_rate = next(iter(learning_rates))
        spec = v02_u2s.ARM_SPECS[arm]
        expected_learning_rate = (
            v02_u2s.ChildActionLinearSchedule().for_child_actions(
                child_actions
            )
            if spec.conservative_ppo
            else v02_u2s.ORIGINAL_LEARNING_RATE
        )
        if not math.isclose(
            float.fromhex(measured_learning_rate),
            expected_learning_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise U2SSmokeError(
                "smoke conservative learning-rate progress is incorrect"
            )

        updated_archive = arm_directory / "updated-disposable-policy.zip"
        model.save(updated_archive)
        reloaded_model = RecurrentPPO.load(updated_archive, device="cpu")
        reloaded = model_fingerprint(reloaded_model)
        if reloaded != after:
            raise U2SSmokeError(
                "smoke disposable policy does not reload byte-identically"
            )
        if file_sha256(parent_copy) != v02_u2s.PARENT_CHECKPOINT_SHA256:
            raise U2SSmokeError("smoke mutated its read-only parent copy")

        worker_evidence = [
            wrapper.public_dict()
            for wrapper in sorted(
                audit_wrappers,
                key=lambda value: value.worker_index,
            )
        ]
        if (
            len(worker_evidence) != v02_u2s.WORKERS
            or sum(int(item["transitions"]) for item in worker_evidence)
            != TOTAL_TRANSITIONS
        ):
            raise U2SSmokeError("smoke worker transition evidence is incomplete")
        resets = [
            reset
            for wrapper in sorted(
                audit_wrappers,
                key=lambda value: value.worker_index,
            )
            for reset in wrapper.resets
        ]
        seed_evidence = _validate_seed_evidence(resets)
        normalized_ledger = _read_normalized_episode_ledger(ledger)
        if len(normalized_ledger) != len(resets):
            raise U2SSmokeError(
                "smoke episode ledger and reset evidence differ"
            )
        active_workers = [
            wrapper.evidence_state()
            for wrapper in sorted(
                episode_wrappers,
                key=lambda value: value.worker_index,
            )
        ]
        return {
            "arm": arm.value,
            "configuration": spec.public_dict(),
            "parent_copy_sha256": file_sha256(parent_copy),
            "before": before.public_dict(),
            "initial_rng_identity": initial_rng_identity,
            "after": after.public_dict(),
            "reloaded_matches_after": reloaded == after,
            "child_actions": child_actions,
            "rollouts": 1,
            "optimizer": {
                "epochs_planned": int(epoch_accounting["epochs_planned"]),
                "epochs_completed": int(
                    epoch_accounting["epochs_completed"]
                ),
                "epochs_skipped": int(epoch_accounting["epochs_skipped"]),
                "target_kl": (
                    _float_hex(spec.target_kl, "smoke target KL")
                    if spec.target_kl is not None
                    else None
                ),
                "kl_stop_triggered": bool(
                    epoch_accounting["kl_stop_triggered"]
                ),
                "approx_kl_hex": approx_kl.hex(),
                "learning_rate_hex": measured_learning_rate,
                "expected_learning_rate_hex": expected_learning_rate.hex(),
            },
            "workers": worker_evidence,
            "worker_evidence_sha256": _canonical_sha256(worker_evidence),
            "trajectory_sha256": _canonical_sha256(
                [item["trajectory_sha256"] for item in worker_evidence]
            ),
            "episode_ledger": {
                "records": len(normalized_ledger),
                "normalized_sha256": _canonical_sha256(
                    normalized_ledger
                ),
                "active_workers": len(active_workers),
                "active_workers_sha256": _canonical_sha256(active_workers),
            },
            "seed_evidence": seed_evidence,
            "reward_evidence": {
                "reward_total_hex": sum(
                    float.fromhex(str(item["reward_total_hex"]))
                    for item in worker_evidence
                ).hex(),
                "extrinsic_total_hex": sum(
                    float.fromhex(str(item["extrinsic_total_hex"]))
                    for item in worker_evidence
                ).hex(),
                "curiosity_total_hex": sum(
                    float.fromhex(str(item["curiosity_total_hex"]))
                    for item in worker_evidence
                ).hex(),
                "penalty_total_hex": sum(
                    float.fromhex(str(item["penalty_total_hex"]))
                    for item in worker_evidence
                ).hex(),
                "penalty_events": sum(
                    int(item["penalty_events"])
                    for item in worker_evidence
                ),
                "maximum_episode_penalty_count": max(
                    int(item["maximum_episode_penalty_count"])
                    for item in worker_evidence
                ),
                "penalty_enabled": spec.no_effect_penalty,
                "reward_components_reconciled": True,
            },
            "updated_only_in_disposable_copy": True,
            "promotable": False,
            "capability_claim": False,
        }
    finally:
        if vector_environment is not None:
            vector_environment.close()
        gc.collect()


def _require_clean_source(repository: Path) -> dict[str, Any]:
    source = git_snapshot(repository)
    if (
        source.get("dirty") is not False
        or not isinstance(source.get("commit"), str)
        or not source["commit"]
    ):
        raise U2SSmokeError(
            "U2-S integration smoke requires one clean identified source commit"
        )
    return {"commit": str(source["commit"]), "dirty": False}


def run_smoke(*, repository: Path | None = None) -> dict[str, Any]:
    """Run all four disposable arms and return canonicalizable evidence."""

    root = (
        Path(__file__).resolve().parents[2]
        if repository is None
        else repository.expanduser().resolve()
    )
    source = _require_clean_source(root)
    canonical_before = _canonical_root_snapshots()
    parent_before = file_sha256(v02_u2s.PARENT_CHECKPOINT)
    if parent_before != v02_u2s.PARENT_CHECKPOINT_SHA256:
        raise U2SSmokeError("claim-bearing U1 parent archive changed")
    inputs = _load_protocol_inputs(root)

    report: dict[str, Any]
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(
        prefix="dungeon-apprentice-u2s-smoke-"
    ) as temporary:
        temporary_path = Path(temporary).resolve()
        _assert_disposable_root(temporary_path)
        arm_reports = [
            _run_arm(
                arm,
                arm_directory=temporary_path / arm.value,
                inputs=inputs,
            )
            for arm in v02_u2s.ARM_PRIORITY
        ]
        trajectories = {
            str(value["trajectory_sha256"]) for value in arm_reports
        }
        episode_ledgers = {
            str(value["episode_ledger"]["normalized_sha256"])
            for value in arm_reports
        }
        if len(trajectories) != 1 or len(episode_ledgers) != 1:
            raise U2SSmokeError(
                "four smoke arms did not receive matched trajectories"
            )
        initial_fingerprints = {
            _canonical_sha256(value["before"]) for value in arm_reports
        }
        if len(initial_fingerprints) != 1:
            raise U2SSmokeError(
                "four smoke arms did not start from identical model state"
            )
        initial_rng_identities = {
            _canonical_sha256(value["initial_rng_identity"])
            for value in arm_reports
        }
        if len(initial_rng_identities) != 1:
            raise U2SSmokeError(
                "four smoke arms did not start from identical RNG state"
            )
        initial_rng_identity = v02_u2s.verify_initial_rng_identity(
            arm_reports[0]["initial_rng_identity"]
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "protocol": SMOKE_PROTOCOL,
            "kind": "disposable_engineering_integration_only",
            "result": "passed",
            "source": source,
            "parent": {
                "checkpoint_sha256": parent_before,
                "policy_tensor_sha256": (
                    v02_u2s.PARENT_POLICY_TENSOR_SHA256
                ),
                "optimizer_state_sha256": (
                    v02_u2s.PARENT_OPTIMIZER_STATE_SHA256
                ),
                "trained_timesteps": v02_u2s.PARENT_LIFETIME_ACTIONS,
                "optimizer_updates": v02_u2s.PARENT_OPTIMIZER_UPDATES,
            },
            "integration": {
                "workers": v02_u2s.WORKERS,
                "rollout_steps_per_worker": v02_u2s.ROLLOUT_STEPS,
                "transitions_per_arm": TOTAL_TRANSITIONS,
                "arms": [
                    arm.value for arm in v02_u2s.ARM_PRIORITY
                ],
                "algorithm_seed": v02_u2s.ALGORITHM_SEED,
                "worker_streams": list(v02_u2s.WORKER_STREAMS),
                "guard_mapping_sha256": (
                    inputs.guard_mapping_sha256
                ),
                "sampler_preflight_sha256": (
                    inputs.sampler_preflight_sha256
                ),
                "matched_initial_model_state": True,
                "matched_initial_rng_identity": True,
                "initial_rng_identity": initial_rng_identity,
                "matched_rollout_trajectory": True,
                "trajectory_sha256": next(iter(trajectories)),
                "episode_ledger_sha256": next(iter(episode_ledgers)),
            },
            "arms": arm_reports,
            "canonical_boundaries_before": canonical_before,
            "canonical_or_claim_policy_updates": False,
            "engineering_smoke_copy_updated": True,
            "canonical_roots_untouched": True,
            "disposable_artifacts_removed": False,
            "evaluation_performed": False,
            "promotion_decision_performed": False,
            "promotable": False,
            "capability_claim": False,
        }

    if temporary_path is None or temporary_path.exists():
        raise U2SSmokeError("disposable smoke artifacts were not removed")
    canonical_after = _canonical_root_snapshots()
    if canonical_after != canonical_before:
        raise U2SSmokeError("smoke changed a canonical U2-S root")
    if file_sha256(v02_u2s.PARENT_CHECKPOINT) != parent_before:
        raise U2SSmokeError("smoke changed the claim-bearing U1 parent")
    if _require_clean_source(root) != source:
        raise U2SSmokeError("smoke changed the tagged source")
    report["canonical_boundaries_after"] = canonical_after
    report["disposable_artifacts_removed"] = True
    return report


def build_parser() -> argparse.ArgumentParser:
    """The release smoke intentionally accepts no scientific overrides."""

    return argparse.ArgumentParser(description=__doc__)


def main() -> None:
    build_parser().parse_args()
    try:
        report = run_smoke()
    except (OSError, ValueError, U2SSmokeError) as error:
        raise SystemExit(f"U2-S disposable integration smoke failed: {error}") from error
    print(_canonical_json_bytes(report).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
