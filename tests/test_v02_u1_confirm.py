import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import dungeon_apprentice.v02_u1_confirm as confirmation
from dungeon_apprentice.v02_lessons import (
    PROTOCOL,
    LessonEvaluation,
    LessonId,
)
from dungeon_apprentice.v02_u1 import CHECKPOINT_SCHEMA_VERSION, EVALUATION_INTERVAL


def _counts(targets: dict[str, float]) -> dict[str, int]:
    lessons = [lesson.value for lesson in LessonId]
    result = {lesson: math.floor(EVALUATION_INTERVAL * targets[lesson]) for lesson in lessons[:-1]}
    result[lessons[-1]] = EVALUATION_INTERVAL - sum(result.values())
    return result


def _write_transition_history(path: Path) -> tuple[dict, dict]:
    profiles = [
        "normal",
        "recovery_navigate",
        "recovery_visible_unlock",
        "recovery_navigate_and_visible_unlock",
        "normal",
        "normal",
    ]
    lifetime = {lesson.value: 0 for lesson in LessonId}
    events = []
    revision = 0
    previous = profiles[0]
    for index, profile in enumerate(profiles):
        if index and profile != previous:
            revision += 1
        targets = confirmation.TARGET_PROFILES[profile]
        counts = _counts(targets)
        lifetime = {lesson: lifetime[lesson] + counts[lesson] for lesson in lifetime}
        decision = "held"
        if index == len(profiles) - 2:
            decision = confirmation.FIRST_PASS_DECISION
        elif index == len(profiles) - 1:
            decision = confirmation.MASTERY_DECISION
        events.append(
            {
                "type": "curriculum_decision",
                "decision": decision,
                "child_trained_timesteps": (index + 1) * EVALUATION_INTERVAL,
                "lesson_passes": {lesson.value: index >= len(profiles) - 2 for lesson in LessonId},
                "allocation_within_tolerance": True,
                "allocation": {
                    "revision": revision,
                    "target_shares": targets,
                    "window_transitions": counts,
                    "realized_shares": {
                        lesson: count / EVALUATION_INTERVAL for lesson, count in counts.items()
                    },
                    "lifetime_transitions": lifetime,
                },
            }
        )
        previous = profile
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    final = events[-1]["allocation"]
    return final, final | {"rng_state": {"state": "opaque"}}


def _evaluation(lesson: LessonId, panel_successes: tuple[int, int]) -> LessonEvaluation:
    successes = sum(panel_successes)
    return LessonEvaluation(
        protocol=PROTOCOL,
        timestamp="2026-07-22T00:00:00Z",
        lesson_id=lesson.value,
        lesson_label=lesson.value,
        episodes=confirmation.CONFIRMATION_CASES,
        successes=successes,
        success_rate=successes / confirmation.CONFIRMATION_CASES,
        panel_successes=panel_successes,
        panel_success_rates=tuple(
            value / confirmation.CONFIRMATION_PANEL_SIZE for value in panel_successes
        ),
        mean_steps=10.0,
    )


def _qualification_cases(
    *,
    first_panel_full_unique: int = 100,
    first_panel_geometry_unique: int = 100,
) -> list[dict]:
    cases = []
    for index in range(confirmation.CONFIRMATION_CASES):
        panel_index = index % confirmation.CONFIRMATION_PANEL_SIZE
        full_index = (
            panel_index
            if index >= confirmation.CONFIRMATION_PANEL_SIZE
            else panel_index % first_panel_full_unique
        )
        geometry_index = (
            panel_index
            if index >= confirmation.CONFIRMATION_PANEL_SIZE
            else panel_index % first_panel_geometry_unique
        )
        panel = index // confirmation.CONFIRMATION_PANEL_SIZE
        cases.append(
            {
                "seed": index,
                "layout_sha256": f"layout-{panel}-{full_index}",
                "geometry_sha256": f"geometry-{panel}-{geometry_index}",
                "oracle_actions": 12,
            }
        )
    return cases


def _verified_checkpoint() -> confirmation.VerifiedU1Checkpoint:
    allocation = confirmation.AllocationVerification(
        windows=2,
        profile_windows={"normal": 2},
        maximum_deviation=0.0,
        final_target_shares=dict(confirmation.TARGET_PROFILES["normal"]),
        final_window_transitions=_counts(confirmation.TARGET_PROFILES["normal"]),
        final_realized_shares=dict(confirmation.TARGET_PROFILES["normal"]),
        whole_child_transitions=_counts(confirmation.TARGET_PROFILES["normal"]),
        whole_child_shares=dict(confirmation.TARGET_PROFILES["normal"]),
    )
    return confirmation.VerifiedU1Checkpoint(
        checkpoint="mastered-local-unlock.zip",
        checkpoint_sha256="a" * 64,
        expected_checkpoint_sha256="a" * 64,
        digest_verified=True,
        sidecar="mastered-local-unlock.json",
        manifest="manifest.json",
        child_algorithm_seed=20260725,
        parent_training_seed=20260725,
        parent_checkpoint_sha256="b" * 64,
        parent_manifest="parent-manifest.json",
        parent_confirmation_sha256="c" * 64,
        source_commit="d" * 40,
        expected_source_commit="d" * 40,
        source_verified=True,
        inherited_trained_timesteps=491_520,
        child_trained_timesteps=393_216,
        trained_timesteps=884_736,
        n_updates=1_728,
        artifact_sha256s={},
        allocation=allocation,
    )


def test_confirmation_seed_mapping_and_frozen_lineages_are_exact() -> None:
    expected_bases = {
        LessonId.NAVIGATE: 15_030_000,
        LessonId.VISIBLE_UNLOCK: 15_040_000,
        LessonId.LOCAL_UNLOCK: 15_020_000,
    }
    blocks = {}
    for lesson, base in expected_bases.items():
        block = confirmation.confirmation_seeds(lesson)
        assert block == tuple(range(base, base + 200))
        blocks[lesson] = set(block)
    assert all(
        blocks[left].isdisjoint(blocks[right])
        for left in LessonId
        for right in LessonId
        if left != right
    )
    assert CHECKPOINT_SCHEMA_VERSION == 3
    assert {
        child: (
            selected["parent_training_seed"],
            selected["sha256"],
            selected["source_commit"],
        )
        for child, selected in confirmation.EXPECTED_CHECKPOINTS.items()
    } == {
        20260725: (
            20260725,
            "bcce9b8251e97ed4fddda32871c891c3783c057bbb1f89deedb3a3d32058102a",
            "e2e765131bc24d22dc8a328ff0c462afc9a74c79",
        ),
        20260729: (
            20260726,
            "2a300927b48f966d5f6ddfeefe13d2e444da1e5c70bcd54e86abd6a9b2d1830b",
            "2bd274e6b89b8e20f36fb7831256d6683e961ffc",
        ),
        20260733: (
            20260727,
            "3d2950e63491d07d3e483660469b8bec869fa137fa61d6b4d22b3d9f0ded2104",
            "2bd274e6b89b8e20f36fb7831256d6683e961ffc",
        ),
    }
    audit = confirmation.seed_partition_audit()
    assert audit["passed"] is True
    assert audit["collisions"] == []
    assert all(
        partition["end"] < 20_000_000
        for partition in audit["partitions"]
        if partition["role"] == "new_confirmation"
    )
    assert confirmation.U1_FULL_LAYOUT_UNIQUENESS_FLOOR == 0.99
    assert confirmation.U1_GEOMETRY_UNIQUENESS_FLOOR == 0.95
    assert all(
        len(selection["artifact_sha256s"]) == 6
        for selection in confirmation.EXPECTED_CHECKPOINTS.values()
    )


def test_checkpoint_verification_rejects_non_schema_three_and_wrong_digest(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints" / "mastered-local-unlock.zip"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    sidecar = checkpoint.with_suffix(".json")
    sidecar.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(confirmation.U1ConfirmationError, match="schema"):
        confirmation.verify_u1_checkpoint(checkpoint)

    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "protocol": PROTOCOL,
                "kind": "mastery",
                "checkpoint_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(confirmation.U1ConfirmationError, match="digest"):
        confirmation.verify_u1_checkpoint(checkpoint)


def test_transition_history_accepts_normal_and_every_recovery_profile(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events.jsonl"
    final, scheduler = _write_transition_history(events)
    verified = confirmation.verify_transition_history(
        events,
        child_trained_timesteps=6 * EVALUATION_INTERVAL,
        final_allocation=final,
        scheduler_state=scheduler,
    )
    assert verified.windows == 6
    assert verified.profile_windows == {
        "normal": 3,
        "recovery_navigate": 1,
        "recovery_navigate_and_visible_unlock": 1,
        "recovery_visible_unlock": 1,
    }
    assert verified.maximum_deviation < 1 / EVALUATION_INTERVAL


@pytest.mark.parametrize("tamper", ["lifetime", "revision", "mastery"])
def test_transition_history_rejects_non_authoritative_history(tmp_path: Path, tamper: str) -> None:
    events = tmp_path / "events.jsonl"
    final, scheduler = _write_transition_history(events)
    records = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    if tamper == "lifetime":
        records[2]["allocation"]["lifetime_transitions"][LessonId.NAVIGATE.value] += 1
    elif tamper == "revision":
        records[2]["allocation"]["revision"] += 1
    else:
        records[-2]["lesson_passes"][LessonId.LOCAL_UNLOCK.value] = False
    events.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(confirmation.U1ConfirmationError):
        confirmation.verify_transition_history(
            events,
            child_trained_timesteps=6 * EVALUATION_INTERVAL,
            final_allocation=final,
            scheduler_state=scheduler,
        )


@pytest.mark.parametrize(
    ("lesson", "panels", "passed"),
    [
        (LessonId.NAVIGATE, (85, 85), True),
        (LessonId.NAVIGATE, (84, 86), False),
        (LessonId.VISIBLE_UNLOCK, (80, 90), True),
        (LessonId.VISIBLE_UNLOCK, (79, 91), False),
        (LessonId.LOCAL_UNLOCK, (80, 90), True),
        (LessonId.LOCAL_UNLOCK, (79, 91), False),
    ],
)
def test_confirmation_gates_include_overall_and_panel_floors(
    lesson: LessonId, panels: tuple[int, int], passed: bool
) -> None:
    assert confirmation.grade_evaluation(_evaluation(lesson, panels))["passed"] is passed


def test_anchor_geometry_is_diagnostic_but_u1_geometry_and_visual_panels_gate() -> None:
    repeated_geometry = _qualification_cases(first_panel_geometry_unique=1)
    anchor = confirmation._panel_uniqueness(
        repeated_geometry,
        full_floor=0.80,
        geometry_floor=None,
    )
    assert all(panel["passed"] for panel in anchor)
    assert all(panel["geometry_affects_verdict"] is False for panel in anchor)

    weak_u1_visuals = _qualification_cases(first_panel_full_unique=98)
    u1 = confirmation._panel_uniqueness(
        weak_u1_visuals,
        full_floor=0.99,
        geometry_floor=0.95,
    )
    assert u1[0]["passed"] is False
    assert u1[1]["passed"] is True


def test_u1_qualification_preserves_validation_collision_as_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = _qualification_cases()
    report = {
        "protocol": PROTOCOL,
        "lesson_id": LessonId.LOCAL_UNLOCK.value,
        "solved": 200,
        "full_unique_layouts": 200,
        "geometry_unique_layouts": 200,
        "cases": cases,
        "failures": [
            {
                "seed": None,
                "error": "qualification and validation share 1 exact visual layouts",
            }
        ],
        "result": "failed",
    }
    monkeypatch.setattr(confirmation, "qualify_local_unlock", lambda *args, **kwargs: report)
    qualified = confirmation.qualify_confirmation_block(LessonId.LOCAL_UNLOCK)
    assert qualified["result"] == "failed"
    assert qualified["validation_disjointness_affects_verdict"] is True


def _fake_model(torch_module):
    class FakePolicy(torch_module.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = torch_module.nn.Linear(2, 2)
            self.lstm_actor = torch_module.nn.LSTM(512, 256)
            self.optimizer = torch_module.optim.Adam(self.parameters())
            for parameter in self.parameters():
                self.optimizer.state[parameter] = {
                    "step": torch_module.tensor(1.0),
                    "exp_avg": torch_module.zeros_like(parameter),
                    "exp_avg_sq": torch_module.zeros_like(parameter),
                }

    return SimpleNamespace(
        policy=FakePolicy(),
        observation_space=SimpleNamespace(shape=(3, 56, 56)),
        action_space=SimpleNamespace(n=7),
        num_timesteps=884_736,
        _n_updates=1_728,
    )


def test_policy_evaluation_is_deterministic_and_no_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    model = _fake_model(torch)
    model_type = SimpleNamespace(load=lambda *_args, **_kwargs: model)
    monkeypatch.setattr(confirmation, "file_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(
        confirmation,
        "evaluate_lesson",
        lambda _model, lesson, _seeds, **_kwargs: _evaluation(lesson, (90, 90)),
    )
    result = confirmation._evaluate_selected_checkpoint(_verified_checkpoint(), model_type)
    assert result["policy_updates"] is False
    assert result["policy_tensor_sha256_before"] == result["policy_tensor_sha256_after"]
    assert result["passed"] is True

    def mutate(_model, lesson, _seeds, **_kwargs):
        with torch.no_grad():
            model.policy.head.weight.add_(1)
        return _evaluation(lesson, (90, 90))

    model = _fake_model(torch)
    model_type = SimpleNamespace(load=lambda *_args, **_kwargs: model)
    monkeypatch.setattr(confirmation, "evaluate_lesson", mutate)
    with pytest.raises(confirmation.U1ConfirmationError, match="changed policy"):
        confirmation._evaluate_selected_checkpoint(_verified_checkpoint(), model_type)


def test_milestone_rates_are_published_as_counts_and_conditionals() -> None:
    result = replace(
        _evaluation(LessonId.LOCAL_UNLOCK, (80, 90)),
        milestone_rates={
            "key_picked_up": 0.90,
            "door_opened": 0.85,
            "success": 0.85,
        },
    )
    evidence = confirmation.milestone_evidence(result)
    assert evidence["key_picked_up"] == 180
    assert evidence["door_opened"] == 170
    assert evidence["quest_completed"] == 170
    assert evidence["door_given_key"] == pytest.approx(170 / 180)
    assert evidence["success_given_door"] == 1.0


def test_main_refuses_to_overwrite_before_any_confirmation_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "report.json"
    output.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dungeon-confirm-v02-u1",
            "--checkpoint",
            "first.zip",
            "--checkpoint",
            "second.zip",
            "--checkpoint",
            "third.zip",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        confirmation.main()


def test_qualification_failure_writes_terminal_report_without_policy_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "qualification-failed.json"
    selected = [
        replace(
            _verified_checkpoint(),
            child_algorithm_seed=child,
            parent_training_seed=int(values["parent_training_seed"]),
            checkpoint_sha256=str(values["sha256"]),
        )
        for child, values in confirmation.EXPECTED_CHECKPOINTS.items()
    ]
    verified = iter(selected)
    qualified_lessons = []
    monkeypatch.setattr(
        confirmation,
        "git_snapshot",
        lambda _path: {"commit": "f" * 40, "branch": "test", "dirty": False},
    )
    monkeypatch.setattr(confirmation, "runtime_snapshot", lambda: {"runtime": "test"})
    monkeypatch.setattr(
        confirmation,
        "verify_u1_checkpoint",
        lambda _path: next(verified),
    )

    def fail_qualification(lesson: LessonId) -> dict:
        qualified_lessons.append(lesson)
        return {
            "lesson_id": lesson.value,
            "result": "failed" if lesson is LessonId.LOCAL_UNLOCK else "passed",
        }

    monkeypatch.setattr(confirmation, "qualify_confirmation_block", fail_qualification)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dungeon-confirm-v02-u1",
            "--checkpoint",
            "first.zip",
            "--checkpoint",
            "second.zip",
            "--checkpoint",
            "third.zip",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        confirmation.main()
    assert stopped.value.code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert qualified_lessons == list(LessonId)
    assert report["verdict"] == "qualification_failed"
    assert report["checkpoint_scoring_performed"] is False
    assert report["checkpoints"] == []
