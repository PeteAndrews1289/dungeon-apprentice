from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import dungeon_apprentice.u2_confirmation_anchor as confirmation_anchor
import dungeon_apprentice.v02_u2_confirm as confirmation
from dungeon_apprentice import v02_u2_lessons as lessons
from dungeon_apprentice.u2_seed_guard import (
    U2AccessPhase,
    U2SeedAccessError,
    U2SeedRole,
    _new_confirmation_launch_claim,
    _post_training_confirmation_seed_access,
    authorize_u2_seed,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _confirmation_access():
    token = "ab" * 32
    claim = _new_confirmation_launch_claim(
        source_commit="a" * 40,
        clean_source=True,
        protocol=confirmation.CONFIRMATION_PROTOCOL,
        plan_sha256="b" * 64,
        checkpoint_set_sha256="c" * 64,
        anchor_tag=confirmation_anchor.ANCHOR_TAG,
        anchor_tag_object="d" * 40,
        anchor_remote_url=confirmation_anchor.EXPECTED_ORIGIN_URL,
        attempt_id=confirmation.ATTEMPT_ID,
        claim_id="e" * 64,
        claim_sha256="f" * 64,
        launcher_token_sha256=hashlib.sha256(token.encode("ascii")).hexdigest(),
    )
    return _post_training_confirmation_seed_access(
        claim=claim,
        launcher_token=token,
    )


def _candidate(
    seed: int,
    *,
    layout: str | None = None,
    geometry: str | None = None,
) -> dict:
    return {
        "seed": seed,
        "layout_sha256": layout or _sha(f"layout-{seed}"),
        "geometry_sha256": geometry or _sha(f"geometry-{seed}"),
        "visibility_stratum": "key_hidden_door_hidden",
    }


def _exclusions(
    *,
    gating: dict[str, frozenset[str]] | None = None,
    diagnostic: dict[str, frozenset[str]] | None = None,
) -> confirmation.LessonReferenceExclusions:
    return confirmation.LessonReferenceExclusions(
        gating=gating or {},
        diagnostic=diagnostic or {},
        sources={},
    )


def _verified_checkpoint() -> confirmation.VerifiedU2Checkpoint:
    frozen = confirmation.FROZEN_CHECKPOINTS[20260737]
    return confirmation.VerifiedU2Checkpoint(
        frozen=frozen,
        sidecar={},
        manifest={},
        artifact_sha256s={"checkpoint": frozen.archive_sha256},
    )


def test_candidate_streams_are_exact_disjoint_and_do_not_open_layouts() -> None:
    expected = {
        lessons.LessonId.SEPARATED_UNLOCK: 15_200_000,
        lessons.LessonId.NAVIGATE: 15_210_000,
        lessons.LessonId.VISIBLE_UNLOCK: 15_220_000,
        lessons.LessonId.LOCAL_UNLOCK: 15_230_000,
    }
    populations = {}
    for lesson, start in expected.items():
        seeds = confirmation.candidate_seeds(lesson)
        assert seeds == range(start, start + 10_000)
        populations[lesson] = set(seeds)
    assert all(
        populations[left].isdisjoint(populations[right])
        for left in lessons.LessonId
        for right in lessons.LessonId
        if left != right
    )
    audit = confirmation.candidate_partition_audit()
    assert audit["passed"] is True
    assert audit["collisions"] == []
    assert audit["untouched_final_start"] == 20_000_000


def test_confirmation_access_requires_bound_remote_tag_and_opens_only_its_streams() -> None:
    access = _confirmation_access()
    assert access.phase is U2AccessPhase.POST_TRAINING_CONFIRMATION
    assert access.confirmation_anchor_tag == confirmation_anchor.ANCHOR_TAG
    for lesson, role in confirmation.CANDIDATE_ROLES.items():
        del lesson
        seed = confirmation.PARTITION_BY_ROLE[role].start
        assert authorize_u2_seed(seed, expected_role=role, access=access).role is role
        with pytest.raises(U2SeedAccessError):
            authorize_u2_seed(seed, expected_role=role)
    with pytest.raises(U2SeedAccessError):
        authorize_u2_seed(
            20_000_000,
            expected_role=U2SeedRole.FINAL_TEST,
            access=access,
        )
    with pytest.raises(U2SeedAccessError, match="anchor remote"):
        _new_confirmation_launch_claim(
            source_commit="a" * 40,
            clean_source=True,
            protocol=confirmation.CONFIRMATION_PROTOCOL,
            plan_sha256="b" * 64,
            checkpoint_set_sha256="c" * 64,
            anchor_tag=confirmation_anchor.ANCHOR_TAG,
            anchor_tag_object="d" * 40,
            anchor_remote_url="https://example.invalid/repository.git",
            attempt_id=confirmation.ATTEMPT_ID,
            claim_id="e" * 64,
            claim_sha256="f" * 64,
            launcher_token_sha256="0" * 64,
        )


def test_frozen_checkpoint_table_names_only_three_first_mastery_archives() -> None:
    assert tuple(confirmation.FROZEN_CHECKPOINTS) == (
        20260737,
        20260741,
        20260745,
    )
    assert [item.child_trained_actions for item in confirmation.FROZEN_CHECKPOINTS.values()] == [
        688_128,
        557_056,
        688_128,
    ]
    assert all(
        item.archive.name == "mastered-separated-unlock.zip"
        and len(item.archive_sha256) == 64
        and item.child_trained_actions - item.first_pass_actions == 32_768
        for item in confirmation.FROZEN_CHECKPOINTS.values()
    )
    digest = confirmation.checkpoint_set_sha256(
        [
            confirmation.VerifiedU2Checkpoint(
                frozen=item,
                sidecar={},
                manifest={},
                artifact_sha256s={},
            )
            for item in reversed(tuple(confirmation.FROZEN_CHECKPOINTS.values()))
        ]
    )
    assert digest == confirmation.checkpoint_set_sha256(
        [
            confirmation.VerifiedU2Checkpoint(
                frozen=item,
                sidecar={},
                manifest={},
                artifact_sha256s={},
            )
            for item in confirmation.FROZEN_CHECKPOINTS.values()
        ]
    )


def test_allocation_verifier_accepts_empty_post_decision_scheduler_window() -> None:
    transitions = {
        "navigate/full": 16_330,
        "unlock/u0-visible": 2_487,
        "unlock/u1-local": 2_503,
        "unlock/u2-separated": 11_448,
    }
    targets = {
        lesson.value: share
        for lesson, share in lessons.NORMAL_TARGETS.items()
    }
    realized = {
        name: count / confirmation.frozen_training.EVALUATION_INTERVAL
        for name, count in transitions.items()
    }
    allocation = {
        "profile": "normal",
        "revision": 0,
        "target_shares": targets,
        "window_transitions": transitions,
        "realized_shares": realized,
        "lifetime_transitions": transitions,
    }
    event = {
        "type": "curriculum_decision",
        "child_trained_actions": confirmation.frozen_training.EVALUATION_INTERVAL,
        "allocation_complete": True,
        "allocation_transitions": confirmation.frozen_training.EVALUATION_INTERVAL,
        "allocation_within_tolerance": True,
        "allocation": allocation,
    }
    scheduler = {
        "profile": "normal",
        "revision": 1,
        "target_shares": targets,
        "window_transitions": {name: 0 for name in transitions},
        "realized_shares": {name: 0.0 for name in transitions},
        "lifetime_transitions": transitions,
        "rng_state": {"bit_generator": "PCG64"},
    }
    evidence = confirmation._verify_allocation_history(
        [event],
        child_actions=confirmation.frozen_training.EVALUATION_INTERVAL,
        final_allocation=allocation,
        scheduler=scheduler,
    )
    assert evidence["windows"] == 1
    with pytest.raises(
        confirmation.U2ConfirmationError,
        match="empty post-decision",
    ):
        confirmation._verify_allocation_history(
            [event],
            child_actions=confirmation.frozen_training.EVALUATION_INTERVAL,
            final_allocation=allocation,
            scheduler=scheduler | {"revision": 0},
        )


def test_selector_rejects_every_gating_collision_and_assigns_panels_by_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = _sha("validation")
    history = _sha("history")
    first = _sha("first")
    population = range(1_000, 1_204)
    monkeypatch.setattr(confirmation, "candidate_seeds", lambda _lesson: population)

    def inspect(_lesson, seed, **_kwargs):
        offset = seed - population.start
        if offset == 0:
            raise confirmation.CandidateRejected("oracle_contract", "fake")
        layout = {
            1: validation,
            2: history,
            3: first,
            4: first,
        }.get(offset)
        return _candidate(seed, layout=layout)

    monkeypatch.setattr(confirmation, "_inspect_candidate", inspect)
    result = confirmation.select_confirmation_block(
        lessons.LessonId.NAVIGATE,
        _exclusions(
            gating={
                "development_validation": frozenset({validation}),
                "u2_child_training_history": frozenset({history}),
            }
        ),
        confirmation_access=_confirmation_access(),
    )
    assert result["result"] == "passed"
    assert result["accepted_count"] == 200
    assert result["candidates_examined"] == 204
    assert result["rejected_count"] == 4
    assert result["cases"][0]["seed"] == 1_003
    assert result["cases"][99]["panel"] == "A"
    assert result["cases"][100]["panel"] == "B"
    assert result["cases"][-1]["seed"] == 1_203
    assert result["rejection_reason_counts"] == {
        "development_validation_exact_overlap": 1,
        "duplicate_accepted_exact_layout": 1,
        "oracle_contract": 1,
        "u2_child_training_history_exact_overlap": 1,
    }


def test_u0_diagnostic_overlap_does_not_select_or_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = _sha("finite-u0-repeat")
    population = range(2_000, 2_200)
    monkeypatch.setattr(confirmation, "candidate_seeds", lambda _lesson: population)
    monkeypatch.setattr(
        confirmation,
        "_inspect_candidate",
        lambda _lesson, seed, **_kwargs: _candidate(
            seed,
            layout=shared if seed == population.start else None,
        ),
    )
    result = confirmation.select_confirmation_block(
        lessons.LessonId.VISIBLE_UNLOCK,
        _exclusions(
            diagnostic={"u2_child_training_history": frozenset({shared})}
        ),
        confirmation_access=_confirmation_access(),
    )
    assert result["result"] == "passed"
    assert result["rejected_count"] == 0
    assert result["diagnostic_reference_overlap"] == {
        "u2_child_training_history": 1
    }


@pytest.mark.parametrize(
    "lesson",
    [lessons.LessonId.LOCAL_UNLOCK, lessons.LessonId.SEPARATED_UNLOCK],
)
def test_u1_and_u2_require_190_geometries_and_95_per_panel(lesson) -> None:
    weak = [
        _candidate(
            3_000 + index,
            geometry=_sha(
                f"a-{min(index, 94)}"
                if index < 100
                else f"b-{min(index - 100, 93)}"
            ),
        )
        for index in range(200)
    ]
    result = confirmation._selection_result(
        lesson,
        weak,
        [],
        examined=200,
        exclusions=_exclusions(),
    )
    assert result["geometry_unique_layouts"] == 189
    assert result["result"] == "failed"
    strong = [
        dict(case)
        | {
            "geometry_sha256": _sha(
                f"a-{min(index, 94)}"
                if index < 100
                else f"b-{min(index - 100, 94)}"
            )
        }
        for index, case in enumerate(weak)
    ]
    result = confirmation._selection_result(
        lesson,
        strong,
        [],
        examined=200,
        exclusions=_exclusions(),
    )
    assert result["geometry_unique_layouts"] == 190
    assert [item["geometry_unique_layouts"] for item in result["panel_qualification"]] == [
        95,
        95,
    ]
    assert result["result"] == "passed"


def test_selected_seed_lists_preserve_acceptance_order_and_reject_tampering() -> None:
    selections = []
    for lesson_index, lesson in enumerate(lessons.LessonId):
        seeds = [100_000 * (lesson_index + 1) + index for index in range(200)]
        selections.append(
            {
                "lesson_id": lesson.value,
                "result": "passed",
                "cases": [_candidate(seed) for seed in seeds],
                "accepted_seed_sha256": confirmation._seed_list_sha256(seeds),
            }
        )
    selected = confirmation.selected_seed_lists(selections)
    assert selected[lessons.LessonId.NAVIGATE][0] == 100_000
    tampered = [dict(item) for item in selections]
    tampered[0] = dict(tampered[0]) | {"accepted_seed_sha256": "0" * 64}
    with pytest.raises(confirmation.U2ConfirmationError, match="changed"):
        confirmation.selected_seed_lists(tampered)


@pytest.mark.parametrize(
    ("lesson", "successes", "panels", "passed"),
    [
        (lessons.LessonId.NAVIGATE, 170, (85, 85), True),
        (lessons.LessonId.NAVIGATE, 169, (85, 84), False),
        (lessons.LessonId.SEPARATED_UNLOCK, 170, (80, 90), True),
        (lessons.LessonId.SEPARATED_UNLOCK, 179, (79, 100), False),
    ],
)
def test_frozen_capability_gates(lesson, successes, panels, passed) -> None:
    result = lessons.U2LessonEvaluation(
        protocol=lessons.PROTOCOL,
        timestamp="2026-07-23T00:00:00Z",
        lesson_id=lesson.value,
        lesson_label=lesson.value,
        episodes=200,
        successes=successes,
        success_rate=successes / 200,
        panel_successes=panels,
        panel_success_rates=(panels[0] / 100, panels[1] / 100),
        mean_steps=10.0,
    )
    assert confirmation.grade_confirmation_evaluation(result)["passed"] is passed


def test_policy_scoring_uses_all_four_exact_panels_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    class FakePolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = torch.nn.Linear(2, 2)
            self.lstm_actor = torch.nn.LSTM(512, 256)
            self.optimizer = torch.optim.Adam(self.parameters())
            for parameter in self.parameters():
                self.optimizer.state[parameter] = {
                    "step": torch.tensor(1.0),
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter),
                }

    frozen = confirmation.FROZEN_CHECKPOINTS[20260737]

    class FakeModel:
        def __init__(self) -> None:
            self.policy = FakePolicy()
            self.observation_space = SimpleNamespace(shape=(3, 56, 56))
            self.action_space = SimpleNamespace(n=7)
            self.num_timesteps = frozen.lifetime_trained_actions
            self._n_updates = frozen.optimizer_updates

    model = FakeModel()
    model_type = SimpleNamespace(load=lambda *_args, **_kwargs: model)
    measured = {"checkpoint": frozen.archive_sha256}
    monkeypatch.setattr(confirmation, "_measure_frozen_artifacts", lambda _frozen: measured)
    observed = {}

    def evaluate(_model, lesson, seeds, **kwargs):
        observed[lesson] = (tuple(seeds), kwargs["seed_access"])
        return lessons.U2LessonEvaluation(
            protocol=lessons.PROTOCOL,
            timestamp="2026-07-23T00:00:00Z",
            lesson_id=lesson.value,
            lesson_label=lesson.value,
            episodes=200,
            successes=180,
            success_rate=0.9,
            panel_successes=(90, 90),
            panel_success_rates=(0.9, 0.9),
            mean_steps=10.0,
        )

    monkeypatch.setattr(lessons, "evaluate_lesson", evaluate)
    access = _confirmation_access()
    seed_lists = {
        lesson: tuple(
            confirmation.candidate_seeds(lesson).start + index
            for index in range(200)
        )
        for lesson in lessons.LessonId
    }
    result = confirmation._evaluate_checkpoint(
        _verified_checkpoint(),
        model_type,
        seed_lists,
        confirmation_access=access,
    )
    assert set(observed) == set(lessons.LessonId)
    assert all(value[1] is access for value in observed.values())
    assert result["policy_updates"] is False
    assert result["policy_tensor_sha256_before"] == result["policy_tensor_sha256_after"]
    assert result["optimizer_state_sha256_before"] == result[
        "optimizer_state_sha256_after"
    ]
    assert result["artifact_sha256s_before"] == result["artifact_sha256s_after"]
    assert result["passed"] is True


def test_terminal_report_writes_checksum_and_final_attempt_ledger(tmp_path: Path) -> None:
    attempt = {
        "schema_version": 1,
        "protocol": confirmation.ATTEMPT_PROTOCOL,
        "attempt_id": confirmation.ATTEMPT_ID,
        "status": "claimed_before_candidate_access",
        "report_present": False,
    }
    confirmation.atomic_write_json(tmp_path / "attempt.json", attempt)
    identity = confirmation.finalize_attempt(
        tmp_path,
        {
            "protocol": confirmation.CONFIRMATION_PROTOCOL,
            "verdict": "qualification_failed",
            "checkpoint_scoring_performed": False,
        },
        evaluator_exit_status=1,
    )
    report = tmp_path / "report.json"
    assert identity["report_sha256"] == confirmation.file_sha256(report)
    assert (tmp_path / "report.json.sha256").read_text().split()[0] == identity[
        "report_sha256"
    ]
    ledger = json.loads((tmp_path / "attempt.json").read_text())
    assert ledger["status"] == "completed_with_report"
    assert ledger["report_verdict"] == "qualification_failed"
    assert ledger["checkpoint_scoring_performed"] is False
    assert (tmp_path / "terminal-bundle/report.json").read_bytes() == report.read_bytes()
    with pytest.raises(
        confirmation.U2ConfirmationError,
        match="terminal bundle already differs",
    ):
        confirmation.finalize_attempt(
            tmp_path,
            {
                "protocol": confirmation.CONFIRMATION_PROTOCOL,
                "verdict": "confirmed",
                "checkpoint_scoring_performed": True,
            },
            evaluator_exit_status=0,
        )


def test_claim_pre_rename_failure_leaves_no_canonical_or_pending_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "attempt"
    monkeypatch.setattr(confirmation, "CANONICAL_ATTEMPT_DIRECTORY", target)
    monkeypatch.setattr(
        confirmation.os,
        "rename",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fault")),
    )
    with pytest.raises(OSError, match="fault"):
        confirmation._claim_attempt(
            source={"commit": "a" * 40, "dirty": False},
            plan_path=tmp_path / "plan.md",
            plan_sha256="b" * 64,
            checkpoint_set_digest="c" * 64,
            verified=[_verified_checkpoint()],
            confirmation_anchor={
                "tag": confirmation_anchor.ANCHOR_TAG,
                "tag_object": "d" * 40,
                "remote_url": confirmation_anchor.EXPECTED_ORIGIN_URL,
            },
        )
    assert not target.exists()
    assert list(tmp_path.glob("*.claim-pending")) == []


def test_claim_post_rename_failure_is_still_durable_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "attempt"
    monkeypatch.setattr(confirmation, "CANONICAL_ATTEMPT_DIRECTORY", target)
    monkeypatch.setattr(
        confirmation,
        "_new_confirmation_launch_claim",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("after rename")),
    )
    with pytest.raises(RuntimeError, match="after rename"):
        confirmation._claim_attempt(
            source={"commit": "a" * 40, "dirty": False},
            plan_path=tmp_path / "plan.md",
            plan_sha256="b" * 64,
            checkpoint_set_digest="c" * 64,
            verified=[_verified_checkpoint()],
            confirmation_anchor={
                "tag": confirmation_anchor.ANCHOR_TAG,
                "tag_object": "d" * 40,
                "remote_url": confirmation_anchor.EXPECTED_ORIGIN_URL,
            },
        )
    assert confirmation._verified_on_disk_claim(target) is not None
    assert all(
        (
            target
            / "selection-journal"
            / lesson.value.replace("/", "__")
        ).is_dir()
        for lesson in lessons.LessonId
    )


def test_terminal_bundle_recovers_after_projection_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    confirmation.atomic_write_json(
        tmp_path / "attempt.json",
        {
            "schema_version": 1,
            "protocol": confirmation.ATTEMPT_PROTOCOL,
            "attempt_id": confirmation.ATTEMPT_ID,
            "status": "claimed_before_candidate_access",
        },
    )
    report = {
        "protocol": confirmation.CONFIRMATION_PROTOCOL,
        "verdict": "integrity_failed",
        "checkpoint_scoring_performed": False,
    }
    original = confirmation._publish_immutable_bytes

    def fail_projection(path, payload):
        if path == tmp_path / "report.json":
            raise OSError("projection fault")
        return original(path, payload)

    monkeypatch.setattr(confirmation, "_publish_immutable_bytes", fail_projection)
    with pytest.raises(OSError, match="projection fault"):
        confirmation.finalize_attempt(
            tmp_path,
            report,
            evaluator_exit_status=1,
        )
    authoritative = confirmation._authoritative_terminal(tmp_path)
    assert authoritative == (report, 1)
    monkeypatch.setattr(confirmation, "_publish_immutable_bytes", original)
    identity = confirmation.finalize_attempt(
        tmp_path,
        authoritative[0],
        evaluator_exit_status=authoritative[1],
    )
    assert Path(identity["report"]).is_file()


def test_selection_journal_retains_opened_seed_on_mid_inspection_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        confirmation,
        "candidate_seeds",
        lambda _lesson: range(4_000, 4_010),
    )
    monkeypatch.setattr(
        confirmation,
        "_inspect_candidate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        confirmation.select_confirmation_block(
            lessons.LessonId.NAVIGATE,
            _exclusions(),
            confirmation_access=_confirmation_access(),
            journal_directory=tmp_path,
            journal_identity={"claim_sha256": "a" * 64},
        )
    lesson_dir = tmp_path / "navigate__full"
    opened = json.loads((lesson_dir / "00001-4000-opened.json").read_text())
    outcome = json.loads((lesson_dir / "00001-4000-outcome.json").read_text())
    assert opened["seed"] == 4_000
    assert opened["base_identity"]["claim_sha256"] == "a" * 64
    assert outcome["status"] == "inspection_failed"


@pytest.mark.parametrize("lesson", list(lessons.LessonId))
def test_real_nonprotected_training_case_meets_candidate_contract(
    lesson: lessons.LessonId,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        confirmation.CANDIDATE_ROLES,
        lesson,
        U2SeedRole.TRAINING,
    )
    evidence = confirmation._inspect_candidate(
        lesson,
        12_345,
        confirmation_access=None,
    )
    assert evidence["seed"] == 12_345
    assert evidence["contracts"]["mechanically_solved"] is True
    assert evidence["contracts"]["reward_unchanged"] is True
    assert evidence["contracts"]["terminal_semantics_unchanged"] is True
    if lesson is lessons.LessonId.LOCAL_UNLOCK:
        assert evidence["contracts"]["divider_contract"] is True
        assert evidence["contracts"]["post_key_turn_present"] is True
    if lesson is lessons.LessonId.SEPARATED_UNLOCK:
        assert evidence["contracts"]["goal_blocked_while_locked"] is True
        assert evidence["contracts"]["matching_key_required"] is True


def _anchor_runner(
    *,
    message: str,
    remote_url: str = confirmation_anchor.EXPECTED_ORIGIN_URL,
    remote_object: str = "1" * 40,
):
    outputs = {
        ("rev-parse", "--verify", "HEAD^{commit}"): "a" * 40,
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
        ("remote", "get-url", confirmation_anchor.ANCHOR_REMOTE): remote_url,
        (
            "cat-file",
            "-t",
            f"refs/tags/{confirmation_anchor.ANCHOR_TAG}",
        ): "tag",
        (
            "rev-parse",
            "--verify",
            f"refs/tags/{confirmation_anchor.ANCHOR_TAG}",
        ): "1" * 40,
        (
            "rev-list",
            "-n",
            "1",
            f"refs/tags/{confirmation_anchor.ANCHOR_TAG}",
        ): "a" * 40,
        (
            "ls-remote",
            "--tags",
            confirmation_anchor.ANCHOR_REMOTE,
            f"refs/tags/{confirmation_anchor.ANCHOR_TAG}",
        ): (
            f"{remote_object}\trefs/tags/{confirmation_anchor.ANCHOR_TAG}"
        ),
        (
            "for-each-ref",
            "--format=%(contents)",
            f"refs/tags/{confirmation_anchor.ANCHOR_TAG}",
        ): message,
    }

    def run(command, **_kwargs):
        key = tuple(command[1:])
        if key not in outputs:
            raise AssertionError(f"unexpected git call: {command}")
        return subprocess.CompletedProcess(command, 0, outputs[key] + "\n", "")

    return run


def test_remote_anchor_binds_commit_plan_checkpoints_and_exact_origin(
    tmp_path: Path,
) -> None:
    message = confirmation_anchor.anchor_message(
        source_commit="a" * 40,
        plan_sha256="b" * 64,
        checkpoint_set_sha256="c" * 64,
    )
    verified = confirmation_anchor.verify_external_anchor(
        tmp_path,
        expected_source_commit="a" * 40,
        expected_plan_sha256="b" * 64,
        expected_checkpoint_set_sha256="c" * 64,
        runner=_anchor_runner(message=message),
    )
    assert verified.tag == "u2-confirmation-v0.2-u2-20260723"
    assert verified.tag_object == "1" * 40
    assert verified.remote_url == confirmation_anchor.EXPECTED_ORIGIN_URL
    assert verified.plan_sha256 == "b" * 64
    assert verified.checkpoint_set_sha256 == "c" * 64
    with pytest.raises(
        confirmation_anchor.U2ConfirmationAnchorError,
        match="frozen GitHub origin",
    ):
        confirmation_anchor.verify_external_anchor(
            tmp_path,
            expected_source_commit="a" * 40,
            expected_plan_sha256="b" * 64,
            expected_checkpoint_set_sha256="c" * 64,
            runner=_anchor_runner(
                message=message,
                remote_url="https://example.invalid/repository.git",
            ),
        )
    with pytest.raises(
        confirmation_anchor.U2ConfirmationAnchorError,
        match="not frozen on origin",
    ):
        confirmation_anchor.verify_external_anchor(
            tmp_path,
            expected_source_commit="a" * 40,
            expected_plan_sha256="b" * 64,
            expected_checkpoint_set_sha256="c" * 64,
            runner=_anchor_runner(message=message, remote_object="2" * 40),
        )


def test_cli_exposes_no_seed_count_path_checkpoint_or_threshold_override() -> None:
    parser = confirmation.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert destinations == {"help", "acknowledgement"}
    assert confirmation.ACKNOWLEDGEMENT.endswith("15200000-15239999 ONCE")
