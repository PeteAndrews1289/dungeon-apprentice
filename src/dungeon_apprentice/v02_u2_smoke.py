"""One-rollout engineering smoke test for U2 Separated Unlock.

This command is intentionally incapable of evaluating or promoting a policy.  It
loads the exact lead U1 archive, collects one 64-transition recurrent-PPO rollout
from four forced Separated Unlock environments, performs one optimizer phase,
and proves that the resulting archive can be reloaded unchanged.

Every generated layout comes from the permanently non-claim engineering range.
There are no command-line controls for seeds, lesson, parent, worker count,
rollout length, device, or storage volume.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

from dungeon_apprentice import v02_u1_confirm, v02_u2
from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice.artifacts import (
    atomic_model_save,
    atomic_write_json,
    atomic_write_text,
    create_run_directory,
    ensure_disk_space,
    file_sha256,
    git_snapshot,
    runtime_snapshot,
    utc_now,
)
from dungeon_apprentice.contracts import (
    CURIOSITY_BUDGET,
    DEFAULT_CURIOSITY_SCALE,
    PIXEL_TILE_SIZE,
)
from dungeon_apprentice.env import EpisodicPixelCuriosity
from dungeon_apprentice.u2_seed_guard import (
    U2SeedAccess,
    U2SeedAccessError,
    U2SeedRole,
    authorize_u2_seed,
    engineering_seed_access,
)
from dungeon_apprentice.u2_storage import validate_directory_path

SMOKE_PROTOCOL = "dungeon-apprentice-v0.2-u2-engineering-smoke"
SCHEMA_VERSION = 1
LEAD_CHILD_SEED = 20260737
WORKERS = 4
ROLLOUT_STEPS = 16
TOTAL_TRANSITIONS = WORKERS * ROLLOUT_STEPS
BATCH_SIZE = TOTAL_TRANSITIONS
N_EPOCHS = 1
ALGORITHM_SEED = 20260802
MINIMUM_FREE_GIB = 25.0
MINIMUM_FREE_BYTES = int(MINIMUM_FREE_GIB * 1024**3)
STORAGE_ROOT = Path("/Volumes/T7 Developer")
RUN_ROOT = STORAGE_ROOT / "DungeonApprentice" / "u2-engineering-smokes"
CONFIRMATION_REPORT = v02_u2.CANONICAL_U1_CONFIRMATION


class U2EngineeringSmokeError(RuntimeError):
    """Raised when an engineering-only plumbing assertion fails."""


@dataclass(frozen=True)
class ModelFingerprint:
    """Minimal state proving that an optimizer phase and reload really occurred."""

    trained_timesteps: int
    optimizer_updates: int
    policy_tensor_sha256: str
    optimizer_state_sha256: str
    optimizer_has_state: bool

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


class ForcedSeparatedEngineeringEnv(gym.Wrapper):
    """Reset only onto this worker's fixed slice of the engineering partition."""

    def __init__(
        self,
        env: gym.Env,
        *,
        worker_index: int,
        access: U2SeedAccess,
        seed_audit: list[dict[str, Any]],
    ) -> None:
        if not 0 <= int(worker_index) < WORKERS:
            raise ValueError(f"worker index must be in [0, {WORKERS})")
        super().__init__(env)
        self._worker_index = int(worker_index)
        self._access = access
        self._seed_audit = seed_audit
        self._episode_index = 0

    def _next_seed(self) -> int:
        offset = self._worker_index + self._episode_index * WORKERS
        if offset >= lessons.ENGINEERING_SEED_COUNT:
            raise U2EngineeringSmokeError(
                "engineering seed slice exhausted before smoke completion"
            )
        seed = lessons.ENGINEERING_SEED_BASE + offset
        authorize_u2_seed(
            seed,
            expected_role=U2SeedRole.ENGINEERING,
            access=self._access,
        )
        return seed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        # SB3 may pass an algorithm seed through VecEnv.reset.  It is deliberately
        # ignored: callers can influence neither the layout seed nor its role.
        requested_seed_ignored = int(seed) if seed is not None else None
        if options:
            raise U2SeedAccessError("engineering smoke reset options are forbidden")
        selected_seed = self._next_seed()
        observation, info = super().reset(
            seed=selected_seed,
            options={
                "u2_seed_role": U2SeedRole.ENGINEERING.value,
                "u2_seed_access": self._access,
            },
        )
        if (
            info.get("u2_seed_role") != U2SeedRole.ENGINEERING.value
            or info.get("lesson_id") != lessons.U2LessonId.SEPARATED_UNLOCK.value
        ):
            raise U2EngineeringSmokeError(
                "environment did not attest engineering Separated Unlock"
            )
        layout_sha256 = str(info.get("layout_sha256", ""))
        geometry_sha256 = str(info.get("geometry_sha256", ""))
        if len(layout_sha256) != 64 or len(geometry_sha256) != 64:
            raise U2EngineeringSmokeError("environment omitted layout digests")
        self._seed_audit.append(
            {
                "worker_index": self._worker_index,
                "episode_index": self._episode_index,
                "seed": selected_seed,
                "requested_seed_ignored": requested_seed_ignored,
                "seed_role": U2SeedRole.ENGINEERING.value,
                "lesson_id": lessons.U2LessonId.SEPARATED_UNLOCK.value,
                "layout_sha256": layout_sha256,
                "geometry_sha256": geometry_sha256,
            }
        )
        self._episode_index += 1
        return observation, info


def make_engineering_env(
    *,
    worker_index: int,
    access: U2SeedAccess,
    seed_audit: list[dict[str, Any]],
) -> gym.Env:
    """Build the actual U2 pixel/curiosity stack around a fixed engineering reset."""

    base = lessons.U2LessonEnv(
        lesson=lessons.U2LessonId.SEPARATED_UNLOCK,
        size=9,
        render_mode=None,
    )
    env: gym.Env = ForcedSeparatedEngineeringEnv(
        base,
        worker_index=worker_index,
        access=access,
        seed_audit=seed_audit,
    )
    env = RGBImgPartialObsWrapper(env, tile_size=PIXEL_TILE_SIZE)
    env = ImgObsWrapper(env)
    return EpisodicPixelCuriosity(
        env,
        scale=DEFAULT_CURIOSITY_SCALE,
        budget=CURIOSITY_BUDGET,
    )


def model_fingerprint(model: Any) -> ModelFingerprint:
    """Fingerprint trained counters, optimizer restoration, and all policy tensors."""

    optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
    return ModelFingerprint(
        trained_timesteps=int(model.num_timesteps),
        optimizer_updates=int(model._n_updates),
        policy_tensor_sha256=v02_u1_confirm.policy_tensor_sha256(model),
        optimizer_state_sha256=(
            v02_u1_confirm.optimizer_state_sha256(model)
            if optimizer is not None
            else ""
        ),
        optimizer_has_state=bool(optimizer is not None and optimizer.state),
    )


def verify_parent_architecture(model: Any, *, parent: Any) -> ModelFingerprint:
    """Check the loaded archive still has the recurrent pixel policy U2 expects."""

    fingerprint = model_fingerprint(model)
    problems: list[str] = []
    if fingerprint.trained_timesteps != int(parent.trained_timesteps):
        problems.append("trained timestep counter")
    if fingerprint.optimizer_updates != int(parent.n_updates):
        problems.append("optimizer update counter")
    if not fingerprint.optimizer_has_state:
        problems.append("optimizer state")
    if int(getattr(model.action_space, "n", -1)) != 7:
        problems.append("seven-action space")
    if tuple(getattr(model.observation_space, "shape", ())) != (3, 56, 56):
        problems.append("3x56x56 pixel observation")
    actor = getattr(getattr(model, "policy", None), "lstm_actor", None)
    if (
        actor is None
        or int(getattr(actor, "hidden_size", -1)) != 256
        or int(getattr(actor, "num_layers", -1)) != 1
    ):
        problems.append("single-layer 256-unit recurrent state")
    if problems:
        raise U2EngineeringSmokeError(
            "lead parent architecture/state mismatch: " + ", ".join(problems)
        )
    return fingerprint


def verify_optimizer_transition(
    before: ModelFingerprint,
    after: ModelFingerprint,
    reloaded: ModelFingerprint,
) -> None:
    """Require exactly one complete tiny optimizer phase and byte-stable reload state."""

    if after.trained_timesteps - before.trained_timesteps != TOTAL_TRANSITIONS:
        raise U2EngineeringSmokeError(
            "smoke did not train exactly one 64-transition vector rollout"
        )
    if after.optimizer_updates - before.optimizer_updates != N_EPOCHS:
        raise U2EngineeringSmokeError(
            "optimizer update counter did not advance by the single frozen epoch"
        )
    if after.policy_tensor_sha256 == before.policy_tensor_sha256:
        raise U2EngineeringSmokeError("optimizer phase did not change policy parameters")
    if after.optimizer_state_sha256 == before.optimizer_state_sha256:
        raise U2EngineeringSmokeError("optimizer phase did not change optimizer state")
    if not after.optimizer_has_state:
        raise U2EngineeringSmokeError("updated policy lost optimizer state")
    if reloaded != after:
        raise U2EngineeringSmokeError("reloaded archive differs from trained in-memory state")


def validate_seed_audit(seed_audit: list[dict[str, Any]], *, access: U2SeedAccess) -> None:
    """Prove every reset stayed inside the only partition this harness may open."""

    seen_workers = {int(item.get("worker_index", -1)) for item in seed_audit}
    if seen_workers != set(range(WORKERS)):
        raise U2EngineeringSmokeError("not every engineering worker reset successfully")
    for item in seed_audit:
        seed = int(item.get("seed", -1))
        authorize_u2_seed(
            seed,
            expected_role=U2SeedRole.ENGINEERING,
            access=access,
        )
        if (
            item.get("seed_role") != U2SeedRole.ENGINEERING.value
            or item.get("lesson_id")
            != lessons.U2LessonId.SEPARATED_UNLOCK.value
        ):
            raise U2EngineeringSmokeError("seed audit contains a non-engineering episode")


def _require_clean_source(
    repository: Path,
    *,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    source = git_snapshot(repository)
    commit = source.get("commit")
    if source.get("dirty") is not False or not isinstance(commit, str) or not commit:
        raise U2EngineeringSmokeError(
            "engineering smoke requires one clean identified source commit"
        )
    if expected_commit is not None and commit != expected_commit:
        raise U2EngineeringSmokeError("source commit changed during engineering smoke")
    return source


def _validate_run_name(run_name: str) -> str:
    candidate = Path(run_name)
    if (
        not run_name
        or candidate.is_absolute()
        or candidate.parent != Path(".")
        or candidate.name in {"", ".", ".."}
    ):
        raise U2EngineeringSmokeError(
            "run name must be one new directory name beneath the fixed T7 smoke root"
        )
    return candidate.name


def _prepare_output(run_name: str, *, protected_paths: tuple[Path, ...]) -> Path:
    if not STORAGE_ROOT.is_dir():
        raise U2EngineeringSmokeError(f"T7 storage root is not mounted: {STORAGE_ROOT}")
    target = RUN_ROOT / _validate_run_name(run_name)
    validate_directory_path(
        target,
        root=STORAGE_ROOT,
        protected_paths=protected_paths,
        allow_missing=True,
    )
    if target.exists():
        raise FileExistsError(f"engineering smoke output already exists: {target}")
    ensure_disk_space(RUN_ROOT, MINIMUM_FREE_BYTES)
    return create_run_directory(RUN_ROOT, target.name)


def _write_checksum(path: Path) -> str:
    digest = file_sha256(path)
    atomic_write_text(
        path.with_name(f"{path.name}.sha256"),
        f"{digest}  {path.name}\n",
    )
    return digest


def _publish_checksums(run_directory: Path, names: tuple[str, ...]) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for name in names:
        path = run_directory / name
        if path.is_file():
            checksums[name] = _write_checksum(path)
    atomic_write_text(
        run_directory / "SHA256SUMS",
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
    )
    return checksums


def _success_report(
    *,
    started_at: str,
    source: dict[str, Any],
    parent: Any,
    before: ModelFingerprint,
    after: ModelFingerprint,
    reloaded: ModelFingerprint,
    archive: Path,
    seed_audit: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": SMOKE_PROTOCOL,
        "kind": "engineering_plumbing_only",
        "result": "passed",
        "started_at": started_at,
        "completed_at": utc_now(),
        "source": source,
        "parent": parent.public_dict(),
        "configuration": {
            "lesson": lessons.U2LessonId.SEPARATED_UNLOCK.value,
            "seed_role": U2SeedRole.ENGINEERING.value,
            "workers": WORKERS,
            "rollout_steps_per_worker": ROLLOUT_STEPS,
            "total_transitions": TOTAL_TRANSITIONS,
            "batch_size": BATCH_SIZE,
            "optimizer_epochs": N_EPOCHS,
            "device": "cpu",
            "algorithm_seed": ALGORITHM_SEED,
        },
        "before": before.public_dict(),
        "after": after.public_dict(),
        "reloaded": reloaded.public_dict(),
        "deltas": {
            "trained_timesteps": after.trained_timesteps - before.trained_timesteps,
            "optimizer_updates": after.optimizer_updates - before.optimizer_updates,
            "policy_parameters_changed": (
                before.policy_tensor_sha256 != after.policy_tensor_sha256
            ),
        },
        "archive": {
            "path": str(archive),
            "sha256": file_sha256(archive),
            "reload_matches_trained_state": reloaded == after,
        },
        "seed_audit": seed_audit,
        "seed_audit_count": len(seed_audit),
        "evaluation_performed": False,
        "promotion_decision_performed": False,
        "capability_claim": False,
        "interpretation": (
            "This passes only the recurrent-PPO U2 engineering plumbing check. "
            "It is not learning, validation, promotion, or capability evidence."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        required=True,
        help=f"new directory name beneath {RUN_ROOT}",
    )
    return parser


def run_smoke(*, run_name: str) -> Path:
    """Execute the fixed smoke protocol and return its exclusive report path."""

    repository = Path(__file__).resolve().parents[2]
    source = _require_clean_source(repository)
    selection = v02_u2.FROZEN_PARENTS[LEAD_CHILD_SEED]
    if file_sha256(selection.archive) != selection.archive_sha256:
        raise U2EngineeringSmokeError("exact lead U1 archive checksum changed")
    parent = v02_u2.verify_parent(
        selection.archive,
        CONFIRMATION_REPORT,
        child_seed=LEAD_CHILD_SEED,
    )

    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    except ImportError as error:
        raise U2EngineeringSmokeError(
            'install training dependencies with: pip install -e ".[train]"'
        ) from error

    protected_paths = (
        selection.archive,
        Path(parent.sidecar),
        Path(parent.manifest),
        Path(parent.confirmation_report),
        Path(parent.attempt_ledger),
        Path(parent.checksum_file),
    )
    run_directory = _prepare_output(run_name, protected_paths=protected_paths)
    started_at = utc_now()
    manifest_path = run_directory / "manifest.json"
    report_path = run_directory / "report.json"
    archive_path = run_directory / "updated-policy.zip"
    seed_audit: list[dict[str, Any]] = []
    access = engineering_seed_access()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": SMOKE_PROTOCOL,
        "kind": "engineering_plumbing_only",
        "started_at": started_at,
        "source": source,
        "runtime": runtime_snapshot(),
        "output_directory": str(run_directory),
        "parent": parent.public_dict(),
        "configuration": {
            "lesson": lessons.U2LessonId.SEPARATED_UNLOCK.value,
            "seed_partition": {
                "role": U2SeedRole.ENGINEERING.value,
                "start": lessons.ENGINEERING_SEED_BASE,
                "end": (
                    lessons.ENGINEERING_SEED_BASE
                    + lessons.ENGINEERING_SEED_COUNT
                    - 1
                ),
            },
            "workers": WORKERS,
            "rollout_steps_per_worker": ROLLOUT_STEPS,
            "total_transitions": TOTAL_TRANSITIONS,
            "batch_size": BATCH_SIZE,
            "optimizer_epochs": N_EPOCHS,
            "device": "cpu",
            "algorithm_seed": ALGORITHM_SEED,
            "minimum_free_gib": MINIMUM_FREE_GIB,
        },
        "information_boundary": (
            "56x56x3 partial RGB pixels plus private 256-unit recurrent state"
        ),
        "online_model_calls": False,
        "demonstrations": False,
        "oracle_actions_used_for_training": False,
        "evaluation_performed": False,
        "promotion_decision_performed": False,
        "capability_claim": False,
    }
    atomic_write_json(manifest_path, manifest)
    _write_checksum(manifest_path)

    vector_environment: Any | None = None
    try:
        factories = [
            (
                lambda worker_index=worker_index: make_engineering_env(
                    worker_index=worker_index,
                    access=access,
                    seed_audit=seed_audit,
                )
            )
            for worker_index in range(WORKERS)
        ]
        vector_environment = VecTransposeImage(DummyVecEnv(factories))
        model = RecurrentPPO.load(
            selection.archive,
            env=vector_environment,
            device="cpu",
            n_steps=ROLLOUT_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
        )
        before = verify_parent_architecture(model, parent=parent)
        model.set_random_seed(ALGORITHM_SEED)
        model.learn(
            total_timesteps=TOTAL_TRANSITIONS,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        after = model_fingerprint(model)
        _require_clean_source(repository, expected_commit=str(source["commit"]))
        validate_seed_audit(seed_audit, access=access)
        if (
            int(getattr(model, "n_steps", -1)) != ROLLOUT_STEPS
            or int(getattr(model, "batch_size", -1)) != BATCH_SIZE
            or int(getattr(model, "n_epochs", -1)) != N_EPOCHS
        ):
            raise U2EngineeringSmokeError("tiny optimizer configuration drifted")

        atomic_model_save(model, archive_path)
        reloaded_model = RecurrentPPO.load(archive_path, device="cpu")
        reloaded = model_fingerprint(reloaded_model)
        verify_optimizer_transition(before, after, reloaded)
        _require_clean_source(repository, expected_commit=str(source["commit"]))

        report = _success_report(
            started_at=started_at,
            source=source,
            parent=parent,
            before=before,
            after=after,
            reloaded=reloaded,
            archive=archive_path,
            seed_audit=seed_audit,
        )
        atomic_write_json(report_path, report)
        _publish_checksums(
            run_directory,
            ("manifest.json", "report.json", "updated-policy.zip"),
        )
        return report_path
    except BaseException as error:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "protocol": SMOKE_PROTOCOL,
            "kind": "engineering_plumbing_only",
            "result": "failed",
            "started_at": started_at,
            "completed_at": utc_now(),
            "source": source,
            "error_type": type(error).__name__,
            "error": str(error),
            "seed_audit": seed_audit,
            "evaluation_performed": False,
            "promotion_decision_performed": False,
            "capability_claim": False,
        }
        atomic_write_json(report_path, failure)
        _publish_checksums(
            run_directory,
            tuple(
                name
                for name in ("manifest.json", "report.json", "updated-policy.zip")
                if (run_directory / name).is_file()
            ),
        )
        raise
    finally:
        if vector_environment is not None:
            vector_environment.close()


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_smoke(run_name=args.run_name)
    except (OSError, ValueError, U2EngineeringSmokeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"report": str(report), "result": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
