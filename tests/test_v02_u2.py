import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from dungeon_apprentice import v02_u2 as u2
from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice.artifacts import atomic_write_json, file_sha256


def _qualification(tmp_path: Path) -> u2.QualificationProvenance:
    tmp_path.mkdir(parents=True, exist_ok=True)
    report = tmp_path / "qualification.json"
    report.write_text("qualified", encoding="utf-8")
    return u2.QualificationProvenance(
        report=str(report),
        report_sha256=file_sha256(report),
        source_commit="d" * 40,
        evidence={
            "report": str(report),
            "report_sha256": file_sha256(report),
            "source_commit": "d" * 40,
        },
    )


def _parent(tmp_path: Path) -> u2.ParentProvenance:
    return u2.ParentProvenance(
        checkpoint=str(tmp_path / "parent.zip"),
        checkpoint_sha256="a" * 64,
        sidecar=str(tmp_path / "parent.json"),
        manifest=str(tmp_path / "manifest.json"),
        source_commit="b" * 40,
        u1_child_seed=20260725,
        u1_parent_seed=20260725,
        trained_timesteps=884_736,
        n_updates=1_728,
        u2_child_seed=20260737,
        worker_streams=(20260737, 20260738, 20260739, 20260740),
        confirmation_report=str(tmp_path / "u1-report.json"),
        confirmation_sha256="c" * 64,
        confirmation_protocol=u2.u1_confirmation_v2.CONFIRMATION_PROTOCOL,
        confirmation_verdict="confirmed",
        confirmation_completed_at="2026-07-23T04:42:41+00:00",
        confirmation_source_commit=u2.EXPECTED_U1_CONFIRMATION_SOURCE,
        attempt_ledger=str(tmp_path / "attempt.json"),
        attempt_ledger_sha256="e" * 64,
        checksum_file=str(tmp_path / "report.json.sha256"),
    )


def _args() -> object:
    args = u2.build_parser().parse_args(["--qualification-report", "qualification.json"])
    u2._validate_args(args)
    return args


def test_frozen_lineages_and_controller_configuration_are_exact(tmp_path: Path) -> None:
    assert u2.CHECKPOINT_SCHEMA_VERSION == 4
    assert u2.CHILD_ACTION_BUDGET == 1_048_576
    assert u2.COHORT_ACTION_BUDGET == 3_145_728
    assert tuple(u2.FROZEN_PARENTS) == (20260737, 20260741, 20260745)
    assert [item.worker_streams for item in u2.FROZEN_PARENTS.values()] == [
        (20260737, 20260738, 20260739, 20260740),
        (20260741, 20260742, 20260743, 20260744),
        (20260745, 20260746, 20260747, 20260748),
    ]

    config = u2.effective_config(
        _args(),
        parent=_parent(tmp_path),
        qualification=_qualification(tmp_path),
    )
    assert config["schema_version"] == 4
    assert config["lessons"] == [lesson.value for lesson in lessons.LessonId]
    assert config["optimization"] == {
        "rollout_steps": 512,
        "rollout_transitions": 2_048,
        "batch_size": 256,
        "n_epochs": 4,
        "learning_rate": 0.00025,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "ent_coef": 0.01,
        "policy_kwargs": {"lstm_hidden_size": 256, "n_lstm_layers": 1},
    }
    assert len(config["transition_target_profiles"]) == 8
    assert config["evaluation"]["interval"] == 32_768
    assert config["evaluation"]["thresholds"]["navigate/full"] == {
        "overall": 68,
        "panel": 34,
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--seed", "99"], "child seed"),
        (["--child-budget", "524288"], "configuration changed"),
        (["--workers", "2"], "configuration changed"),
        (["--gamma", "0.99"], "reward dominance"),
        (["--minimum-free-gib", "24.9"], "at least 25"),
        (["--keep-rolling-exams", "6"], "at most five"),
    ],
)
def test_frozen_argument_changes_fail_before_training(
    arguments: list[str], message: str
) -> None:
    args = u2.build_parser().parse_args(
        ["--qualification-report", "qualification.json", *arguments]
    )
    with pytest.raises(SystemExit, match=message):
        u2._validate_args(args)


def test_parent_requires_exact_archive_report_ledger_and_no_update_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "mastered-local-unlock.zip"
    with ZipFile(checkpoint, "w") as zipped:
        zipped.writestr("policy.optimizer.pth", b"optimizer")
    report = tmp_path / "report.json"
    ledger = tmp_path / "attempt.json"
    checksum = tmp_path / "report.json.sha256"
    selection = u2.FrozenParent(
        u1_child_seed=20260725,
        archive=checkpoint,
        archive_sha256=file_sha256(checkpoint),
        u2_child_seed=20260737,
        worker_streams=(20260737, 20260738, 20260739, 20260740),
        inherited_baseline={
            "navigate/full": 74,
            "unlock/u0-visible": 80,
            "unlock/u1-local": 72,
        },
    )
    source_commit = "f" * 40
    entry = {
        "checkpoint": {
            "child_algorithm_seed": 20260725,
            "checkpoint_sha256": selection.archive_sha256,
            "checkpoint": str(checkpoint),
        },
        "integrity": {"passed": True},
        "passed": True,
        "policy_updates": False,
        "checkpoint_sha256_before_load": selection.archive_sha256,
        "checkpoint_sha256_after_evaluation": selection.archive_sha256,
        "policy_tensor_sha256_before": "1" * 64,
        "policy_tensor_sha256_after": "1" * 64,
        "optimizer_state_sha256_before": "2" * 64,
        "optimizer_state_sha256_after": "2" * 64,
        "model_num_timesteps_before": 884_736,
        "model_num_timesteps_after": 884_736,
        "model_updates_before": 1_728,
        "model_updates_after": 1_728,
    }
    report_value = {
        "schema_version": 2,
        "protocol": u2.u1_confirmation_v2.CONFIRMATION_PROTOCOL,
        "verdict": "confirmed",
        "policy_updates": False,
        "checkpoint_scoring_performed": True,
        "source": {"dirty": False, "commit": source_commit},
        "selected_child_seeds": list(u2.u1_confirmation.EXPECTED_CHILD_SEEDS),
        "checkpoints": [entry],
        "completed_at": "2026-07-23T04:42:41+00:00",
    }
    atomic_write_json(report, report_value)
    report_digest = file_sha256(report)
    atomic_write_json(
        ledger,
        {
            "schema_version": 1,
            "protocol": u2.EXPECTED_U1_ATTEMPT_PROTOCOL,
            "status": "completed_with_report",
            "source_dirty": False,
            "source_commit": source_commit,
            "launcher_exit_status": 0,
            "evaluator_exit_status": 0,
            "report_present": True,
            "report_verdict": "confirmed",
            "report_sha256": report_digest,
            "output": str(report),
            "checkpoints": [str(checkpoint)],
        },
    )
    checksum.write_text(f"{report_digest}  {report}\n", encoding="utf-8")
    verified = SimpleNamespace(
        checkpoint=str(checkpoint),
        checkpoint_sha256=selection.archive_sha256,
        digest_verified=True,
        lineage_verified=True,
        configuration_verified=True,
        source_verified=True,
        child_algorithm_seed=20260725,
        parent_training_seed=20260725,
        trained_timesteps=884_736,
        n_updates=1_728,
        sidecar=str(checkpoint.with_suffix(".json")),
        manifest=str(tmp_path / "manifest.json"),
        source_commit="e" * 40,
    )
    monkeypatch.setattr(u2, "FROZEN_PARENTS", {20260737: selection})
    monkeypatch.setattr(u2, "CANONICAL_U1_CONFIRMATION", report)
    monkeypatch.setattr(u2, "EXPECTED_U1_CONFIRMATION_SHA256", report_digest)
    monkeypatch.setattr(u2, "EXPECTED_U1_CONFIRMATION_SOURCE", source_commit)
    monkeypatch.setattr(
        u2.u1_confirmation,
        "verify_u1_checkpoint",
        lambda _path: verified,
    )

    parent = u2.verify_parent(checkpoint, report, child_seed=20260737)
    assert parent.checkpoint_sha256 == selection.archive_sha256
    assert parent.confirmation_sha256 == report_digest
    assert parent.worker_streams == selection.worker_streams

    report_value["checkpoints"][0]["policy_updates"] = True
    atomic_write_json(report, report_value)
    monkeypatch.setattr(u2, "EXPECTED_U1_CONFIRMATION_SHA256", file_sha256(report))
    with pytest.raises(u2.U2ProtocolError, match="individually pass"):
        u2.verify_parent(checkpoint, report, child_seed=20260737)


def test_qualification_wrapper_binds_verifier_digest_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report.json"
    report.write_bytes(b"sealed result")
    digest = file_sha256(report)
    snapshot = {"result": "passed"}
    evidence = SimpleNamespace(
        public_dict=lambda: {
            "report": str(report),
            "report_sha256": digest,
            "report_byte_length": len(b"sealed result"),
            "source_commit": "d" * 40,
            "generator_profile": lessons.GENERATOR_PROFILE,
            "generator_profile_version": lessons.GENERATOR_PROFILE_VERSION,
            "attempt_id": "u2-preflight-v0.2-u2-20260723-attempt-1",
            "attempt_sha256": "a" * 64,
            "claim_id": "c" * 64,
            "claim_sha256": "b" * 64,
        },
        verified_report=lambda: dict(snapshot),
        qualified_seed_access=lambda: object(),
    )
    anchor = SimpleNamespace(
        public_dict=lambda: {
            "report": str(report),
            "report_sha256": digest,
            "report_byte_length": len(b"sealed result"),
            "source_commit": "d" * 40,
            "attempt_id": "u2-preflight-v0.2-u2-20260723-attempt-1",
            "attempt_sha256": "a" * 64,
            "claim_id": "c" * 64,
            "claim_sha256": "b" * 64,
            "generator_profile": lessons.GENERATOR_PROFILE,
            "generator_profile_version": lessons.GENERATOR_PROFILE_VERSION,
        },
    )
    from dungeon_apprentice import v02_u2_qualify

    monkeypatch.setattr(
        v02_u2_qualify,
        "verify_u2_qualification_report",
        lambda _path, **_anchors: evidence,
    )
    qualified = u2.verify_qualification(
        report,
        expected_source_commit="d" * 40,
        anchor=anchor,
    )
    assert qualified.report_sha256 == digest
    assert qualified.source_commit == "d" * 40
    assert qualified.verified_report() == snapshot
    assert qualified.seed_access() is not None

    report.write_bytes(b"tampered")
    assert qualified.verified_report() == snapshot


def _evaluation(lesson: lessons.LessonId, successes: int) -> lessons.LessonEvaluation:
    panels = (successes // 2, successes - successes // 2)
    return lessons.LessonEvaluation(
        protocol=u2.PROTOCOL,
        timestamp="2026-07-23T00:00:00+00:00",
        lesson_id=lesson.value,
        lesson_label=lessons.LESSON_SPECS[lesson].label,
        episodes=80,
        successes=successes,
        success_rate=successes / 80,
        panel_successes=panels,
        panel_success_rates=tuple(value / 40 for value in panels),
        mean_steps=20.0,
    )


class _FakeBaseCallback:
    def __init__(self, *, verbose: int) -> None:
        self.verbose = verbose
        self.locals = {}

    @property
    def num_timesteps(self) -> int:
        return int(self.model.num_timesteps)


class _FakeModel:
    def __init__(self, *, actions: int, updates: int) -> None:
        self.num_timesteps = actions
        self._n_updates = updates
        self.logger = SimpleNamespace(name_to_value={})

    def save(self, destination: Path) -> None:
        Path(destination).write_bytes(
            f"actions={self.num_timesteps},updates={self._n_updates}".encode()
        )


def _callback(
    tmp_path: Path,
    *,
    state: lessons.CurriculumState | None = None,
    child_actions: int = 0,
    loader=None,
):
    parent = _parent(tmp_path)
    state = state or lessons.CurriculumState()
    scheduler = lessons.TransitionDeficitScheduler(state, seed=71)
    callback_type = u2._CallbackFactory.create(_FakeBaseCallback)
    callback = callback_type(
        run_directory=tmp_path,
        state=state,
        controller=u2.ControllerState(),
        scheduler=scheduler,
        parent=parent,
        qualification=_qualification(tmp_path),
        source={"commit": "d" * 40, "dirty": False},
        segment={
            "id": "segment",
            "index": 0,
            "started_at": "2026-07-23T00:00:00+00:00",
            "lineage_algorithm_seed": parent.u2_child_seed,
            "initial_worker_streams": list(parent.worker_streams),
            "segment_algorithm_seed": 20260737,
            "segment_worker_streams": list(parent.worker_streams),
            "start_lifetime_trained_actions": parent.trained_timesteps,
            "start_child_trained_actions": 0,
            "remaining_child_actions": u2.CHILD_ACTION_BUDGET,
            "resume_checkpoint": None,
        },
        effective_config={"frozen": True},
        exam_model_loader=loader or (lambda _path: None),
        validation_access=object(),
        evaluation_interval=u2.EVALUATION_INTERVAL,
        frame_interval=2_048,
        minimum_free_bytes=0,
        started_at="2026-07-23T00:00:00+00:00",
        initial_trained_actions=parent.trained_timesteps + child_actions,
        initial_updates=(
            parent.n_updates
            + child_actions // u2.ROLLOUT_TRANSITIONS * u2.PPO_EPOCHS
        ),
        child_start_actions=parent.trained_timesteps,
        storage_guard=None,
    )
    callback.model = _FakeModel(
        actions=parent.trained_timesteps + child_actions,
        updates=(
            parent.n_updates
            + child_actions // u2.ROLLOUT_TRANSITIONS * u2.PPO_EPOCHS
        ),
    )
    return callback


def test_mastery_requires_two_adjacent_normal_windows_and_all_prerequisites(
    tmp_path: Path,
) -> None:
    callback = _callback(tmp_path)
    passed = {lesson: _evaluation(lesson, 80) for lesson in lessons.LessonId}

    first = callback._apply_decision(
        passed,
        allocation_ok=True,
        child_actions=32_768,
    )
    callback.controller.first_pass_artifact = {
        "path": "checkpoints/first-pass-separated-unlock.zip",
        "checkpoint_sha256": "f" * 64,
        "child_trained_actions": 32_768,
    }
    second = callback._apply_decision(
        passed,
        allocation_ok=True,
        child_actions=65_536,
    )

    assert first.first_pass is True
    assert first.mastered is False
    assert second.mastered is True
    assert callback.state.mastered is True
    assert callback.state.consecutive_passes == 2

    weak_state = lessons.CurriculumState(consecutive_passes=1)
    weak_callback = _callback(tmp_path / "weak", state=weak_state)
    weak = dict(passed)
    weak[lessons.LessonId.LOCAL_UNLOCK] = _evaluation(
        lessons.LessonId.LOCAL_UNLOCK, 60
    )
    outcome = weak_callback._apply_decision(
        weak,
        allocation_ok=True,
        child_actions=32_768,
    )
    assert "Recovery started" in outcome.message
    assert weak_state.weak_prerequisites == (lessons.LessonId.LOCAL_UNLOCK,)
    assert weak_state.consecutive_passes == 0


def test_recovery_requires_two_clean_allocation_valid_boundaries(
    tmp_path: Path,
) -> None:
    state = lessons.CurriculumState(
        weak_prerequisites=(lessons.LessonId.NAVIGATE,)
    )
    callback = _callback(tmp_path, state=state)
    passed = {lesson: _evaluation(lesson, 80) for lesson in lessons.LessonId}

    held = callback._apply_decision(
        passed,
        allocation_ok=False,
        child_actions=32_768,
    )
    first = callback._apply_decision(
        passed,
        allocation_ok=True,
        child_actions=65_536,
    )
    second = callback._apply_decision(
        passed,
        allocation_ok=True,
        child_actions=98_304,
    )

    assert "outside tolerance" in held.message
    assert "confirmation required" in first.message
    assert "normal U2 practice resumes" in second.message
    assert state.weak_prerequisites == ()
    assert state.recovery_passes == 0
    assert state.consecutive_passes == 0


def test_broken_streak_clears_and_replaces_first_pass_candidate(
    tmp_path: Path,
) -> None:
    callback = _callback(tmp_path)
    passed = {lesson: _evaluation(lesson, 80) for lesson in lessons.LessonId}

    first = callback._apply_decision(
        passed,
        allocation_ok=True,
        child_actions=32_768,
    )
    assert first.first_pass is True
    callback.controller.first_pass_artifact = {
        "path": "checkpoints/first-pass-separated-unlock.zip",
        "checkpoint_sha256": "1" * 64,
        "child_trained_actions": 32_768,
    }

    failed = dict(passed)
    failed[lessons.LessonId.SEPARATED_UNLOCK] = _evaluation(
        lessons.LessonId.SEPARATED_UNLOCK,
        0,
    )
    callback._apply_decision(
        failed,
        allocation_ok=True,
        child_actions=65_536,
    )

    assert callback.state.consecutive_passes == 0
    assert callback.controller.last_normal_pass_child_actions is None
    assert callback.controller.first_pass_artifact is None

    replacement = callback._apply_decision(
        passed,
        allocation_ok=True,
        child_actions=98_304,
    )
    assert replacement.first_pass is True
    callback.controller.first_pass_artifact = {
        "path": "checkpoints/first-pass-separated-unlock.zip",
        "checkpoint_sha256": "2" * 64,
        "child_trained_actions": 98_304,
    }
    mastered = callback._apply_decision(
        passed,
        allocation_ok=True,
        child_actions=131_072,
    )

    assert mastered.mastered is True
    assert (
        callback.controller.first_pass_artifact["child_trained_actions"]
        == 98_304
    )
    assert callback.controller.last_normal_pass_child_actions == 131_072


def test_mastery_rejects_a_nonadjacent_first_pass_candidate(tmp_path: Path) -> None:
    callback = _callback(tmp_path)
    passed = {lesson: _evaluation(lesson, 80) for lesson in lessons.LessonId}
    callback._apply_decision(
        passed,
        allocation_ok=True,
        child_actions=32_768,
    )
    callback.controller.first_pass_artifact = {
        "path": "checkpoints/first-pass-separated-unlock.zip",
        "checkpoint_sha256": "1" * 64,
        "child_trained_actions": 16_384,
    }

    with pytest.raises(u2.U2ProtocolError, match="immediately preceding"):
        callback._apply_decision(
            passed,
            allocation_ok=True,
            child_actions=65_536,
        )


def test_exam_is_reloaded_by_digest_and_decision_uses_separate_resume_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_actions = 32_768
    parent = _parent(tmp_path)
    actions = parent.trained_timesteps + child_actions
    updates = parent.n_updates + child_actions // 2_048 * 4
    graded_model = SimpleNamespace(
        num_timesteps=actions,
        _n_updates=updates,
        marker="reloaded",
    )
    loaded_paths: list[Path] = []

    def load_exam(path: Path):
        loaded_paths.append(path)
        return graded_model

    callback = _callback(
        tmp_path,
        child_actions=child_actions,
        loader=load_exam,
    )
    callback.scheduler.window_transitions.update(
        {
            lessons.LessonId.NAVIGATE: 16_384,
            lessons.LessonId.VISIBLE_UNLOCK: 2_458,
            lessons.LessonId.LOCAL_UNLOCK: 2_458,
            lessons.LessonId.SEPARATED_UNLOCK: 11_468,
        }
    )
    callback.scheduler.lifetime_transitions.update(
        callback.scheduler.window_transitions
    )
    monkeypatch.setattr(
        lessons,
        "validation_seeds",
        lambda _lesson, *, access: tuple(range(80)),
    )

    def evaluate(model, lesson, _seeds, **_kwargs):
        assert model is graded_model
        return _evaluation(lesson, 80)

    monkeypatch.setattr(lessons, "evaluate_lesson", evaluate)
    exam = callback._publish_live_archive(
        "rolling/exam-0032768",
        kind="exam",
        resume_eligible=False,
        replace=False,
    )
    exam_sidecar_before = exam.with_suffix(".json").read_bytes()
    callback._evaluate_archive(exam, trigger="scheduled", decide=True)

    assert loaded_paths == [exam]
    assert exam.with_suffix(".json").read_bytes() == exam_sidecar_before
    exam_sidecar = json.loads(exam_sidecar_before)
    assert exam_sidecar["curriculum"]["consecutive_passes"] == 0
    assert exam_sidecar["resume_eligible"] is False

    candidate = tmp_path / "checkpoints" / "first-pass-separated-unlock.zip"
    resume = tmp_path / "checkpoints" / "resume.zip"
    assert candidate.is_file()
    assert resume.is_file()
    assert file_sha256(candidate) == file_sha256(exam) == file_sha256(resume)
    resume_sidecar = json.loads(resume.with_suffix(".json").read_text())
    assert resume_sidecar["kind"] == "resume"
    assert resume_sidecar["resume_eligible"] is True
    assert resume_sidecar["curriculum"]["consecutive_passes"] == 1
    assert resume_sidecar["controller"]["first_pass_artifact"]["path"].endswith(
        "first-pass-separated-unlock.zip"
    )
    assert resume_sidecar["exam_source"]["checkpoint_sha256"] == file_sha256(exam)
    evaluation = json.loads(
        (tmp_path / "evaluations.jsonl").read_text().splitlines()[0]
    )
    assert evaluation["reloaded_for_grading"] is True
    assert evaluation["checkpoint_sha256"] == file_sha256(exam)
    assert evaluation["sidecar_sha256"] == file_sha256(exam.with_suffix(".json"))


def test_schema4_resume_restores_state_and_rejects_sidecar_tampering(
    tmp_path: Path,
) -> None:
    parent = _parent(tmp_path)
    qualification = _qualification(tmp_path)
    args = _args()
    config = u2.effective_config(
        args,
        parent=parent,
        qualification=qualification,
    )
    child_actions = 32_768
    state = lessons.CurriculumState(consecutive_passes=1, revision=1)
    scheduler = lessons.TransitionDeficitScheduler(state, seed=99)
    scheduler.window_transitions.update(
        {
            lessons.LessonId.NAVIGATE: 16_384,
            lessons.LessonId.VISIBLE_UNLOCK: 2_458,
            lessons.LessonId.LOCAL_UNLOCK: 2_458,
            lessons.LessonId.SEPARATED_UNLOCK: 11_468,
        }
    )
    scheduler.lifetime_transitions.update(scheduler.window_transitions)
    controller = u2.ControllerState(
        last_decision_child_actions=child_actions,
        last_normal_pass_child_actions=child_actions,
    )
    candidate = tmp_path / "checkpoints" / "first-pass-separated-unlock.zip"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"first pass")
    atomic_write_json(candidate.with_suffix(".json"), {"kind": "first_pass"})
    u2._write_integrity(candidate, candidate.with_suffix(".json"))
    controller.first_pass_artifact = {
        "path": "checkpoints/first-pass-separated-unlock.zip",
        "checkpoint_sha256": file_sha256(candidate),
        "child_trained_actions": child_actions,
    }
    checkpoint = tmp_path / "checkpoints" / "resume.zip"
    checkpoint.write_bytes(b"model")
    sidecar = {
        "schema_version": 4,
        "protocol": u2.PROTOCOL,
        "kind": "resume",
        "resume_eligible": True,
        "checkpoint_sha256": file_sha256(checkpoint),
        "source": {"commit": "d" * 40, "dirty": False},
        "effective_config": config,
        "parent": parent.public_dict(),
        "qualification": qualification.public_dict(),
        "curriculum": state.public_dict(),
        "controller": controller.public_dict(),
        "scheduler": scheduler.state_dict(),
        "last_completed_allocation": scheduler.snapshot(),
        "last_completed_allocation_valid": True,
        "progress": {
            "collected_actions": parent.trained_timesteps + child_actions,
            "trained_actions": parent.trained_timesteps + child_actions,
            "lifetime_trained_actions": parent.trained_timesteps + child_actions,
            "inherited_trained_actions": parent.trained_timesteps,
            "child_trained_actions": child_actions,
            "remaining_child_actions": u2.CHILD_ACTION_BUDGET - child_actions,
            "segment_trained_actions": child_actions,
            "optimizer_updates": (
                parent.n_updates
                + child_actions // u2.ROLLOUT_TRANSITIONS * u2.PPO_EPOCHS
            ),
        },
        "segment": {
            "id": "segment",
            "index": 0,
            "started_at": "2026-07-23T00:00:00+00:00",
            "lineage_algorithm_seed": parent.u2_child_seed,
            "initial_worker_streams": list(parent.worker_streams),
            "segment_algorithm_seed": parent.u2_child_seed,
            "segment_worker_streams": list(parent.worker_streams),
            "start_lifetime_trained_actions": parent.trained_timesteps,
            "start_child_trained_actions": 0,
            "remaining_child_actions": u2.CHILD_ACTION_BUDGET,
            "resume_checkpoint": None,
        },
        "exam_source": None,
    }
    atomic_write_json(checkpoint.with_suffix(".json"), sidecar)
    u2._write_integrity(checkpoint, checkpoint.with_suffix(".json"))

    loaded = u2.load_resume(
        checkpoint,
        expected_config=config,
        parent=parent,
        qualification=qualification,
        expected_source_commit="d" * 40,
    )
    assert loaded.child_trained == child_actions
    assert loaded.segment_index == 1
    assert loaded.curriculum.consecutive_passes == 1
    assert loaded.controller.last_normal_pass_child_actions == child_actions

    sidecar["controller"]["first_pass_artifact"]["child_trained_actions"] = 65_536
    atomic_write_json(checkpoint.with_suffix(".json"), sidecar)
    u2._write_integrity(checkpoint, checkpoint.with_suffix(".json"))
    with pytest.raises(u2.U2ProtocolError, match="bound first-pass"):
        u2.load_resume(
            checkpoint,
            expected_config=config,
            parent=parent,
            qualification=qualification,
            expected_source_commit="d" * 40,
        )

    sidecar["controller"]["first_pass_artifact"][
        "child_trained_actions"
    ] = child_actions
    sidecar["controller"]["last_normal_pass_child_actions"] = 65_536
    atomic_write_json(checkpoint.with_suffix(".json"), sidecar)
    with pytest.raises(u2.U2ProtocolError, match="integrity record"):
        u2.load_resume(
            checkpoint,
            expected_config=config,
            parent=parent,
            qualification=qualification,
            expected_source_commit="d" * 40,
        )


def test_source_guard_runs_before_rollout_storage_checks(tmp_path: Path) -> None:
    calls: list[str] = []
    parent = _parent(tmp_path)
    state = lessons.CurriculumState()
    scheduler = lessons.TransitionDeficitScheduler(state, seed=1)
    callback_type = u2._CallbackFactory.create(_FakeBaseCallback)
    callback = callback_type(
        run_directory=tmp_path,
        state=state,
        controller=u2.ControllerState(),
        scheduler=scheduler,
        parent=parent,
        qualification=_qualification(tmp_path),
        source={"commit": "d" * 40, "dirty": False},
        segment={"index": 0},
        effective_config={},
        exam_model_loader=lambda _path: None,
        validation_access=object(),
        evaluation_interval=32_768,
        frame_interval=2_048,
        minimum_free_bytes=0,
        started_at="2026-07-23T00:00:00+00:00",
        initial_trained_actions=parent.trained_timesteps,
        initial_updates=parent.n_updates,
        child_start_actions=parent.trained_timesteps,
        storage_guard=lambda _anticipated: calls.append("storage"),
        source_guard=lambda: calls.append("source"),
    )
    callback.model = _FakeModel(
        actions=parent.trained_timesteps,
        updates=parent.n_updates,
    )

    callback._on_rollout_start()

    assert calls == ["source", "storage"]


def test_checkpoint_bundle_uses_known_growth_then_post_publish_audit(
    tmp_path: Path,
) -> None:
    callback = _callback(tmp_path)
    checks: list[int] = []
    callback.storage_guard = lambda anticipated: checks.append(anticipated) or {}

    checkpoint = callback._publish_live_archive(
        "initial",
        kind="initial",
        resume_eligible=True,
        replace=False,
    )

    bundle_bytes = sum(path.stat().st_size for path in u2._bundle_paths(checkpoint))
    assert checks == [bundle_bytes, 0]
    u2._verify_integrity(checkpoint, checkpoint.with_suffix(".json"))


def test_checkpoint_admission_failure_leaves_no_partial_bundle(
    tmp_path: Path,
) -> None:
    callback = _callback(tmp_path)

    def reject(anticipated: int) -> dict:
        if anticipated:
            raise OSError("frozen cap would be crossed")
        return {}

    callback.storage_guard = reject
    destination = tmp_path / "checkpoints" / "initial.zip"

    with pytest.raises(OSError, match="frozen cap"):
        callback._publish_live_archive(
            "initial",
            kind="initial",
            resume_eligible=True,
            replace=False,
        )

    assert not any(path.exists() for path in u2._bundle_paths(destination))


def test_post_publish_audit_failure_rolls_back_replaced_bundle(
    tmp_path: Path,
) -> None:
    callback = _callback(tmp_path)
    destination = callback._publish_live_archive(
        "latest-safe",
        kind="latest",
        resume_eligible=True,
        replace=True,
    )
    original = {
        path.name: path.read_bytes() for path in u2._bundle_paths(destination)
    }
    checks = 0

    def fail_post_publish(_anticipated: int) -> dict:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise OSError("post-publication cap race")
        return {}

    callback.storage_guard = fail_post_publish
    callback.model.num_timesteps += u2.ROLLOUT_TRANSITIONS
    callback.trained_actions += u2.ROLLOUT_TRANSITIONS
    callback.model._n_updates += u2.PPO_EPOCHS

    with pytest.raises(OSError, match="post-publication cap race"):
        callback._publish_live_archive(
            "latest-safe",
            kind="latest",
            resume_eligible=True,
            replace=True,
        )

    assert {
        path.name: path.read_bytes() for path in u2._bundle_paths(destination)
    } == original
    u2._verify_integrity(destination, destination.with_suffix(".json"))


def test_incomplete_rollback_preserves_recovery_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback = _callback(tmp_path)
    callback._publish_live_archive(
        "latest-safe",
        kind="latest",
        resume_eligible=True,
        replace=True,
    )
    checks = 0

    def fail_post_publish(_anticipated: int) -> dict:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise OSError("post-publication cap race")
        return {}

    callback.storage_guard = fail_post_publish
    callback.model.num_timesteps += u2.ROLLOUT_TRANSITIONS
    callback.trained_actions += u2.ROLLOUT_TRANSITIONS
    callback.model._n_updates += u2.PPO_EPOCHS
    real_replace = u2.os.replace
    rollback_failed = False

    def fail_one_rollback(source, destination):
        nonlocal rollback_failed
        if Path(source).name.startswith("previous-") and not rollback_failed:
            rollback_failed = True
            raise OSError("simulated rollback write failure")
        return real_replace(source, destination)

    monkeypatch.setattr(u2.os, "replace", fail_one_rollback)
    with pytest.raises(
        u2.U2BundleRollbackError,
        match="preserved recovery backups",
    ):
        callback._publish_live_archive(
            "latest-safe",
            kind="latest",
            resume_eligible=True,
            replace=True,
        )

    recovery_files = list(callback.staging_directory.rglob("previous-*"))
    assert rollback_failed
    assert recovery_files
    assert all(path.is_file() and path.stat().st_size > 0 for path in recovery_files)


def test_latest_safe_bundle_is_replaced_after_every_complete_rollout(
    tmp_path: Path,
) -> None:
    callback = _callback(tmp_path)
    parent_actions = callback.trained_actions
    parent_updates = callback.last_updates
    callback.controller.below_twenty_at_half_budget = True

    callback.scheduler.window_transitions[lessons.LessonId.NAVIGATE] = (
        u2.ROLLOUT_TRANSITIONS
    )
    callback.scheduler.lifetime_transitions[lessons.LessonId.NAVIGATE] = (
        u2.ROLLOUT_TRANSITIONS
    )
    callback.model.num_timesteps += u2.ROLLOUT_TRANSITIONS
    callback.model._n_updates += u2.PPO_EPOCHS
    callback._on_rollout_start()

    latest = tmp_path / "checkpoints" / "latest-safe.zip"
    first_digest = file_sha256(latest)
    first_sidecar = json.loads(latest.with_suffix(".json").read_text())
    assert first_sidecar["kind"] == "latest"
    assert first_sidecar["resume_eligible"] is True
    assert first_sidecar["progress"]["trained_actions"] == (
        parent_actions + u2.ROLLOUT_TRANSITIONS
    )
    assert first_sidecar["progress"]["optimizer_updates"] == (
        parent_updates + u2.PPO_EPOCHS
    )

    callback.scheduler.window_transitions[lessons.LessonId.LOCAL_UNLOCK] = (
        u2.ROLLOUT_TRANSITIONS
    )
    callback.scheduler.lifetime_transitions[lessons.LessonId.LOCAL_UNLOCK] = (
        u2.ROLLOUT_TRANSITIONS
    )
    callback.model.num_timesteps += u2.ROLLOUT_TRANSITIONS
    callback.model._n_updates += u2.PPO_EPOCHS
    callback._on_rollout_start()

    second_sidecar = json.loads(latest.with_suffix(".json").read_text())
    assert file_sha256(latest) != first_digest
    assert second_sidecar["progress"]["trained_actions"] == (
        parent_actions + 2 * u2.ROLLOUT_TRANSITIONS
    )
    assert second_sidecar["progress"]["optimizer_updates"] == (
        parent_updates + 2 * u2.PPO_EPOCHS
    )
    assert second_sidecar["controller"]["below_twenty_at_half_budget"] is True
    assert second_sidecar["scheduler"] == callback.scheduler.state_dict()
    assert callback.latest_safe_checkpoint == latest
    assert callback.latest_safe_trained_actions == callback.trained_actions
    assert list((tmp_path / "checkpoints").glob("latest-safe.zip")) == [latest]
    u2._verify_integrity(latest, latest.with_suffix(".json"))
    resumed = u2.load_resume(
        latest,
        expected_config={"frozen": True},
        parent=callback.parent,
        qualification=callback.qualification,
        expected_source_commit="d" * 40,
    )
    assert resumed.lifetime_trained == callback.trained_actions
    assert resumed.n_updates == callback.last_updates
    assert resumed.controller.below_twenty_at_half_budget is True
    assert resumed.scheduler_state == callback.scheduler.state_dict()

    checkpoint_events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [event["kind"] for event in checkpoint_events] == ["latest", "latest"]
    assert {event["path"] for event in checkpoint_events} == {
        "checkpoints/latest-safe.zip"
    }


def test_interruption_recovers_an_immediate_post_train_boundary(
    tmp_path: Path,
) -> None:
    callback = _callback(tmp_path)
    callback.scheduler.window_transitions[lessons.LessonId.SEPARATED_UNLOCK] = (
        u2.ROLLOUT_TRANSITIONS
    )
    callback.scheduler.lifetime_transitions[lessons.LessonId.SEPARATED_UNLOCK] = (
        u2.ROLLOUT_TRANSITIONS
    )
    callback.model.num_timesteps += u2.ROLLOUT_TRANSITIONS
    callback.model._n_updates += u2.PPO_EPOCHS

    callback.finalize_failure("interrupted")

    latest = tmp_path / "checkpoints" / "latest-safe.zip"
    sidecar = json.loads(latest.with_suffix(".json").read_text())
    assert sidecar["progress"]["child_trained_actions"] == u2.ROLLOUT_TRANSITIONS
    assert sidecar["scheduler"] == callback.scheduler.state_dict()
    event = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[-1])
    assert event["type"] == "run_interrupted"
    assert event["pending_optimizer_boundary"] == "complete_recovered"
    assert event["recovered_complete_actions"] == u2.ROLLOUT_TRANSITIONS
    assert event["recovered_complete_optimizer_updates"] == u2.PPO_EPOCHS
    assert event["discarded_partial_actions"] == 0
    assert event["unpublished_complete_actions"] == 0
    assert event["latest_safe_checkpoint"] == "checkpoints/latest-safe.zip"


def test_interruption_discards_only_a_genuine_partial_phase(tmp_path: Path) -> None:
    callback = _callback(tmp_path)
    initial = callback._publish_live_archive(
        "initial",
        kind="initial",
        resume_eligible=True,
        replace=False,
    )
    callback.latest_safe_checkpoint = initial
    callback.latest_safe_trained_actions = callback.trained_actions
    callback.model.num_timesteps += 127

    callback.finalize_failure("interrupted")

    event = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[-1])
    assert event["pending_optimizer_boundary"] == "partial"
    assert event["recovered_complete_actions"] == 0
    assert event["discarded_partial_actions"] == 127
    assert event["discarded_partial_optimizer_updates"] == 0
    assert event["latest_safe_checkpoint"] == "checkpoints/initial.zip"
    assert not (tmp_path / "checkpoints" / "latest-safe.zip").exists()


def test_interruption_records_failure_and_exits_nonzero() -> None:
    phases: list[str] = []
    callback = SimpleNamespace(
        finalize_failure=lambda phase: phases.append(phase)
    )

    with pytest.raises(SystemExit) as stopped:
        u2._exit_after_interruption(callback)

    assert stopped.value.code == 130
    assert phases == ["interrupted"]


def test_pretraining_failure_has_durable_dashboard_status(
    tmp_path: Path,
) -> None:
    parent = _parent(tmp_path)
    qualification = _qualification(tmp_path)
    started = "2026-07-23T00:00:00+00:00"

    u2._write_setup_terminal_status(
        tmp_path,
        phase="crashed",
        started_at=started,
        wall_start=0.0,
        parent=parent,
        qualification=qualification,
        source={"commit": "d" * 40, "dirty": False},
        initial_trained_actions=parent.trained_timesteps,
        child_trained_actions=0,
        expected_updates=parent.n_updates,
        segment={"index": 0},
    )

    status = json.loads((tmp_path / "status.json").read_text())
    event = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[-1])
    assert status["phase"] == "crashed"
    assert status["setup_complete"] is False
    assert status["parent_checkpoint_sha256"] == parent.checkpoint_sha256
    assert status["qualification_sha256"] == qualification.report_sha256
    assert status["child_trained_timesteps"] == 0
    assert status["optimizer_updates"] == parent.n_updates
    assert event["classification"] == "pre_training_setup_failure"


def test_resume_carries_permanent_decision_bundle_into_new_segment(
    tmp_path: Path,
) -> None:
    old_run = tmp_path / "old"
    archive = old_run / "checkpoints" / "resume.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"resume")
    artifact = old_run / "checkpoints" / "first-pass-separated-unlock.zip"
    artifact.write_bytes(b"first-pass")
    atomic_write_json(artifact.with_suffix(".json"), {"kind": "first_pass"})
    u2._write_integrity(artifact, artifact.with_suffix(".json"))
    controller = u2.ControllerState(
        first_pass_artifact={
            "path": "checkpoints/first-pass-separated-unlock.zip",
            "checkpoint_sha256": file_sha256(artifact),
            "child_trained_actions": 32_768,
        }
    )
    resume = u2.ResumeBundle(
        checkpoint=archive,
        sidecar={},
        curriculum=lessons.CurriculumState(),
        controller=controller,
        scheduler_state={},
        lifetime_trained=0,
        child_trained=32_768,
        n_updates=0,
        segment_index=1,
        last_completed_allocation=None,
        last_completed_allocation_valid=None,
    )
    new_run = tmp_path / "new"
    new_run.mkdir()

    carried = u2._carry_forward_decision_artifacts(
        resume,
        run_directory=new_run,
    )

    copied = new_run / "checkpoints" / "first-pass-separated-unlock.zip"
    assert len(carried) == 1
    assert file_sha256(copied) == file_sha256(artifact)
    u2._verify_integrity(copied, copied.with_suffix(".json"))
    chained = u2.ResumeBundle(
        checkpoint=new_run / "checkpoints" / "resume.zip",
        sidecar={},
        curriculum=lessons.CurriculumState(),
        controller=controller,
        scheduler_state={},
        lifetime_trained=0,
        child_trained=32_768,
        n_updates=0,
        segment_index=2,
        last_completed_allocation=None,
        last_completed_allocation_valid=None,
    )
    chained.checkpoint.write_bytes(b"next-resume")
    u2._verify_named_decision_artifact(
        chained.checkpoint,
        controller.first_pass_artifact,
        label="first-pass",
    )
