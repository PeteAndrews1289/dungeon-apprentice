import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import dungeon_apprentice.v02_u1_confirm_v2 as confirmation
from dungeon_apprentice import v02_u1_confirm as frozen
from dungeon_apprentice.v02_lessons import PROTOCOL, LessonEvaluation, LessonId


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _candidate(seed: int, *, layout: str | None = None, geometry: str | None = None) -> dict:
    return {
        "seed": seed,
        "layout_sha256": layout or _sha(f"layout-{seed}"),
        "geometry_sha256": geometry or _sha(f"geometry-{seed}"),
        "oracle_actions": 12,
        "visibility": {"key_visible": True, "door_visible": False},
        "contracts": {
            "mechanically_solved": True,
            "observation": True,
            "visibility": True,
            "reward": True,
            "action_length": True,
        },
    }


def _exclusions(**overrides: frozenset[str]) -> confirmation.LessonReferenceExclusions:
    values = {
        "validation_exact": frozenset(),
        "prior_attempt_exact": frozenset(),
        "engineering_qualification_exact": frozenset(),
        "frozen_child_history_exact": frozenset(),
    }
    values.update(overrides)
    return confirmation.LessonReferenceExclusions(**values)


def _selection(
    lesson: LessonId,
    cases: list[dict],
    *,
    result: str = "passed",
) -> dict:
    seeds = [case["seed"] for case in cases]
    return {
        "lesson_id": lesson.value,
        "result": result,
        "cases": cases,
        "accepted_seed_sha256": confirmation._seed_list_sha256(seeds),
    }


def _verified_checkpoint() -> frozen.VerifiedU1Checkpoint:
    allocation = frozen.AllocationVerification(
        windows=2,
        profile_windows={"normal": 2},
        maximum_deviation=0.0,
        final_target_shares=dict(frozen.TARGET_PROFILES["normal"]),
        final_window_transitions={
            lesson.value: 0 for lesson in LessonId
        },
        final_realized_shares={
            lesson.value: 0.0 for lesson in LessonId
        },
        whole_child_transitions={
            lesson.value: 0 for lesson in LessonId
        },
        whole_child_shares={
            lesson.value: 0.0 for lesson in LessonId
        },
    )
    return frozen.VerifiedU1Checkpoint(
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


def test_candidate_populations_are_exact_and_disjoint_without_generation() -> None:
    expected = {
        LessonId.NAVIGATE: 15_060_000,
        LessonId.VISIBLE_UNLOCK: 15_070_000,
        LessonId.LOCAL_UNLOCK: 15_050_000,
    }
    populations = {}
    for lesson, base in expected.items():
        seeds = confirmation.candidate_seeds(lesson)
        assert isinstance(seeds, range)
        assert len(seeds) == 10_000
        assert seeds.start == base
        assert seeds.stop == base + 10_000
        populations[lesson] = set(seeds)
    assert all(
        populations[left].isdisjoint(populations[right])
        for left in LessonId
        for right in LessonId
        if left != right
    )
    audit = confirmation.candidate_partition_audit()
    assert audit["passed"] is True
    assert audit["collisions"] == []
    assert confirmation.CONFIRMATION_PROTOCOL.endswith("-confirmation-v2")
    assert confirmation.PRIOR_ATTEMPT_SHA256 == (
        "d2fa53308f7488cc08f5ee67b86a125ad91c6ba3790fcd9433c35aab1215b25c"
    )


def test_prior_attempt_requires_exact_digest_protocol_seed_lists_and_all_lessons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "report.json"
    path.write_text("immutable", encoding="utf-8")
    qualifications = []
    expected_hashes = {}
    for lesson in LessonId:
        cases = [
            {
                "seed": seed,
                "layout_sha256": _sha(f"{lesson.value}-{seed}"),
                "geometry_sha256": _sha(f"geometry-{lesson.value}-{seed}"),
            }
            for seed in frozen.confirmation_seeds(lesson)
        ]
        expected_hashes[lesson] = frozenset(case["layout_sha256"] for case in cases)
        qualifications.append({"lesson_id": lesson.value, "cases": cases})
    report = {
        "protocol": frozen.CONFIRMATION_PROTOCOL,
        "checkpoint_scoring_performed": False,
        "verdict": "qualification_failed",
        "layout_qualification": qualifications,
    }
    monkeypatch.setattr(
        confirmation,
        "file_sha256",
        lambda _path: confirmation.PRIOR_ATTEMPT_SHA256,
    )
    monkeypatch.setattr(frozen, "_read_json", lambda *_args: report)

    evidence, hashes, geometries = confirmation.load_prior_attempt(path)
    assert hashes == expected_hashes
    assert all(len(geometries[lesson]) == 200 for lesson in LessonId)
    assert evidence["sha256"] == confirmation.PRIOR_ATTEMPT_SHA256
    assert evidence["checkpoint_scoring_performed"] is False

    monkeypatch.setattr(confirmation, "file_sha256", lambda _path: "0" * 64)
    with pytest.raises(confirmation.U1ConfirmationV2Error, match="digest changed"):
        confirmation.load_prior_attempt(path)


def test_reference_builder_applies_training_exclusion_only_to_u1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = {lesson: frozenset({_sha(f"prior-{lesson.value}")}) for lesson in LessonId}
    monkeypatch.setattr(
        confirmation,
        "load_prior_attempt",
        lambda _path: (
            {"sha256": confirmation.PRIOR_ATTEMPT_SHA256},
            prior,
            {lesson: frozenset({_sha(f"geometry-{lesson.value}")}) for lesson in LessonId},
        ),
    )
    monkeypatch.setattr(
        confirmation,
        "_validation_exact_hashes",
        lambda lesson: frozenset({_sha(f"validation-{lesson.value}")}),
    )
    engineering = frozenset({_sha("engineering")})
    history = frozenset({_sha("history")})
    monkeypatch.setattr(
        confirmation,
        "_engineering_qualification_hashes",
        lambda _verified: (engineering, [{"fake": True}]),
    )
    monkeypatch.setattr(
        confirmation,
        "_frozen_child_u1_history_hashes",
        lambda _verified: (history, [{"fake": True}]),
    )

    exclusions, evidence, prior_geometries = confirmation.build_reference_exclusions(
        [_verified_checkpoint()],
        prior_attempt_report=Path("unused.json"),
    )
    assert exclusions[LessonId.LOCAL_UNLOCK].engineering_qualification_exact == engineering
    assert exclusions[LessonId.LOCAL_UNLOCK].frozen_child_history_exact == history
    assert exclusions[LessonId.NAVIGATE].engineering_qualification_exact == frozenset()
    assert exclusions[LessonId.VISIBLE_UNLOCK].frozen_child_history_exact == frozenset()
    assert evidence["anchor_training_overlap_affects_verdict"] is False
    assert evidence["u1_training_overlap_affects_verdict"] is True
    assert set(prior_geometries) == set(LessonId)


def test_selector_filters_every_declared_collision_then_assigns_panels_by_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = _sha("validation")
    prior = _sha("prior")
    engineering = _sha("engineering")
    history = _sha("history")
    first_accepted = _sha("first-accepted")
    exclusions = _exclusions(
        validation_exact=frozenset({validation}),
        prior_attempt_exact=frozenset({prior}),
        engineering_qualification_exact=frozenset({engineering}),
        frozen_child_history_exact=frozenset({history}),
    )
    population = range(1_000, 1_206)
    monkeypatch.setattr(confirmation, "candidate_seeds", lambda _lesson: population)

    def inspect(_lesson: LessonId, seed: int) -> dict:
        offset = seed - population.start
        if offset == 0:
            raise confirmation.CandidateRejected("oracle_contract", "fake failure")
        collisions = {
            1: validation,
            2: prior,
            3: engineering,
            4: history,
            5: first_accepted,
            6: first_accepted,
        }
        layout = collisions.get(offset)
        return _candidate(seed, layout=layout)

    monkeypatch.setattr(confirmation, "_inspect_candidate", inspect)
    report = confirmation.select_confirmation_block(
        LessonId.LOCAL_UNLOCK,
        exclusions,
    )

    assert report["result"] == "passed"
    assert report["accepted_count"] == 200
    assert report["candidates_examined"] == 206
    assert report["rejected_count"] == 6
    assert report["rejection_reason_counts"] == {
        "duplicate_accepted_exact_layout": 1,
        "engineering_qualification_exact_overlap": 1,
        "frozen_child_history_exact_overlap": 1,
        "oracle_contract": 1,
        "prior_attempt_exact_overlap": 1,
        "validation_exact_overlap": 1,
    }
    assert report["cases"][0]["seed"] == 1_005
    assert report["cases"][0]["status"] == "accepted"
    assert report["cases"][99]["panel_label"] == "A"
    assert report["cases"][100]["panel_label"] == "B"
    assert report["cases"][-1]["seed"] == 1_205
    assert all(case["accepted_index"] == index for index, case in enumerate(report["cases"]))
    assert all(panel["passed"] for panel in report["panel_qualification"])
    assert all(
        item["passed"] for item in report["accepted_vs_exclusion_overlap"].values()
    )
    reference_rejection = report["rejections"][1]
    assert reference_rejection["status"] == "rejected"
    assert reference_rejection["oracle_actions"] == 12
    assert reference_rejection["visibility"] == {
        "key_visible": True,
        "door_visible": False,
    }


def test_anchor_ignores_training_sets_but_still_excludes_validation_and_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = _sha("anchor-training-repeat")
    exclusions = _exclusions(
        engineering_qualification_exact=frozenset({shared}),
        frozen_child_history_exact=frozenset({shared}),
    )
    population = range(2_000, 2_200)
    monkeypatch.setattr(confirmation, "candidate_seeds", lambda _lesson: population)
    monkeypatch.setattr(
        confirmation,
        "_inspect_candidate",
        lambda _lesson, seed: _candidate(
            seed,
            layout=shared if seed == population.start else None,
        ),
    )
    report = confirmation.select_confirmation_block(LessonId.NAVIGATE, exclusions)
    assert report["result"] == "passed"
    assert report["accepted_count"] == 200
    assert report["rejected_count"] == 0
    assert report["cases"][0]["layout_sha256"] == shared
    assert report["geometry_affects_verdict"] is False


def test_u1_geometry_requires_190_overall_and_95_in_each_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population = range(3_000, 3_200)
    monkeypatch.setattr(confirmation, "candidate_seeds", lambda _lesson: population)
    weak = [
        _candidate(
            seed,
            geometry=_sha(
                f"geometry-{index if index < 95 else 94}"
                if index < 100
                else f"geometry-b-{index - 100 if index - 100 < 94 else 93}"
            ),
        )
        for index, seed in enumerate(population)
    ]
    report = confirmation._selection_result(
        LessonId.LOCAL_UNLOCK,
        weak,
        [],
        candidates_examined=200,
        exclusions=_exclusions(),
    )
    assert report["geometry_unique_layouts"] == 189
    assert report["panel_qualification"][0]["geometry_unique_layouts"] == 95
    assert report["panel_qualification"][1]["geometry_unique_layouts"] == 94
    assert report["result"] == "failed"

    strong = [
        dict(case)
        | {
            "geometry_sha256": _sha(
                f"geometry-{index if index < 95 else 94}"
                if index < 100
                else f"geometry-b-{index - 100 if index - 100 < 95 else 94}"
            )
        }
        for index, case in enumerate(weak)
    ]
    report = confirmation._selection_result(
        LessonId.LOCAL_UNLOCK,
        strong,
        [],
        candidates_examined=200,
        exclusions=_exclusions(),
    )
    assert report["geometry_unique_layouts"] == 190
    assert [panel["geometry_unique_layouts"] for panel in report["panel_qualification"]] == [
        95,
        95,
    ]
    assert report["result"] == "passed"


def test_selected_seed_lists_preserve_exact_acceptance_order_and_detect_tampering() -> None:
    selections = []
    for lesson_index, lesson in enumerate(LessonId):
        cases = [
            _candidate(100_000 * (lesson_index + 1) + offset)
            for offset in range(confirmation.CONFIRMATION_CASES)
        ]
        selections.append(_selection(lesson, cases))
    selected = confirmation.selected_seed_lists(selections)
    assert selected[LessonId.NAVIGATE][0] == 100_000
    assert selected[LessonId.NAVIGATE][-1] == 100_199

    tampered = [dict(selection) for selection in selections]
    tampered[0] = dict(tampered[0]) | {"accepted_seed_sha256": "0" * 64}
    with pytest.raises(confirmation.U1ConfirmationV2Error, match="seed evidence"):
        confirmation.selected_seed_lists(tampered)


def test_attempt1_geometry_matrix_is_protocol_independent_and_diagnostic() -> None:
    shared = _sha("shared-geometry")
    selections = []
    prior = {}
    for lesson_index, lesson in enumerate(LessonId):
        cases = [
            _candidate(
                400_000 * (lesson_index + 1) + offset,
                geometry=shared if offset == 0 else None,
            )
            for offset in range(confirmation.CONFIRMATION_CASES)
        ]
        selections.append(_selection(lesson, cases))
        prior[lesson] = frozenset(
            {
                shared,
                _sha(f"attempt1-{lesson.value}"),
            }
        )
    diagnostics = confirmation.attempt1_geometry_diagnostics(selections, prior)
    assert diagnostics["geometry_overlap_affects_verdict"] is False
    assert diagnostics["attempt1_report_sha256"] == confirmation.PRIOR_ATTEMPT_SHA256
    assert all(
        cell["overlap_unique_geometries"] == 1
        for row in diagnostics["matrix"].values()
        for cell in row.values()
    )


def test_policy_scoring_uses_exact_seed_lists_and_preserves_all_learned_state(
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

    class FakeModel:
        def __init__(self) -> None:
            self.policy = FakePolicy()
            self.observation_space = SimpleNamespace(shape=(3, 56, 56))
            self.action_space = SimpleNamespace(n=7)
            self.num_timesteps = 884_736
            self._n_updates = 1_728

    observed = {}

    def evaluate(_model, lesson: LessonId, seeds, **_kwargs) -> LessonEvaluation:
        observed[lesson] = tuple(seeds)
        return LessonEvaluation(
            protocol=PROTOCOL,
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

    model = FakeModel()
    model_type = SimpleNamespace(load=lambda *_args, **_kwargs: model)
    monkeypatch.setattr(confirmation, "evaluate_lesson", evaluate)
    monkeypatch.setattr(confirmation, "file_sha256", lambda _path: "a" * 64)
    seed_lists = {
        lesson: tuple(
            confirmation.CANDIDATE_BASES[lesson] + offset
            for offset in range(confirmation.CONFIRMATION_CASES)
        )
        for lesson in LessonId
    }
    result = confirmation._evaluate_selected_checkpoint(
        _verified_checkpoint(),
        model_type,
        seed_lists,
    )

    assert observed == seed_lists
    assert result["policy_updates"] is False
    assert result["integrity"]["passed"] is True
    assert result["integrity"]["archive_digest_verified"] is True
    assert result["evaluation_settings"] == {
        "device": "cpu",
        "environment_size": 9,
        "deterministic_actions": True,
        "recurrent_state_reset_each_case": True,
        "curiosity_enabled": False,
        "inference_mode": True,
    }
    assert result["policy_tensor_sha256_before"] == result["policy_tensor_sha256_after"]
    assert result["optimizer_state_sha256_before"] == result["optimizer_state_sha256_after"]
    assert result["checkpoint_sha256_before_load"] == result[
        "checkpoint_sha256_after_evaluation"
    ]
    assert result["passed"] is True


def test_terminal_qualification_failure_never_contains_scored_checkpoints() -> None:
    report = confirmation._terminal_qualification_report(
        {"protocol": confirmation.CONFIRMATION_PROTOCOL},
        wall_started=0.0,
        selections=[{"lesson_id": LessonId.LOCAL_UNLOCK.value, "result": "failed"}],
        diagnostics=None,
    )
    assert report["verdict"] == "qualification_failed"
    assert report["checkpoint_scoring_performed"] is False
    assert report["checkpoints"] == []
