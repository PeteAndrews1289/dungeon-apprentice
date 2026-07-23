from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from dungeon_apprentice import v02_u2s_qualify

REPOSITORY = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY / "scripts" / "run_v02_u2s_ablation.sh"
HELPER = REPOSITORY / "scripts" / "u2s_ablation_manifest.py"


def _load_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("u2s_ablation_manifest", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _initial_rng_identity(helper: ModuleType) -> dict[str, object]:
    components = {
        "python_random_sha256": "1" * 64,
        "numpy_global_sha256": "2" * 64,
        "torch_cpu_sha256": "3" * 64,
        "torch_mps_sha256": None,
        "torch_cuda_sha256": None,
        "model_action_space_sha256": "4" * 64,
        "scheduler_sha256": "5" * 64,
        "workers": [
            {
                "worker_index": index,
                "worker_stream": stream,
                "environment_np_random_sha256": "6" * 64,
                "action_space_sha256": "7" * 64,
                "observation_space_sha256": "8" * 64,
            }
            for index, stream in enumerate(helper.WORKER_STREAMS)
        ],
    }
    return {
        "captured_before_action_one": True,
        "algorithm_seed": helper.ALGORITHM_SEED,
        "components": components,
        "aggregate_sha256": helper._canonical_sha256(components),
    }


def _qualification(
    helper: ModuleType,
    *,
    source_commit: str,
    tag_object: str,
) -> dict[str, object]:
    return {
        "report": helper.QUALIFICATION_REPORT,
        "report_sha256": "f" * 64,
        "checksum": {"path": f"{helper.QUALIFICATION_REPORT}.sha256"},
        "source_commit": source_commit,
        "tag": helper.TAG_NAME,
        "tag_object": tag_object,
        "tag_payload_sha256": "9" * 64,
        "verdict": "qualified",
        "protocol_document": helper.PROTOCOL_DOCUMENT,
        "protocol_document_sha256": "e" * 64,
        "guard_mapping_sha256": helper.QUALIFIED_GUARD_MAPPING_SHA256,
        "u2r_terminal_active_records_sha256": (
            helper.U2R_TERMINAL_ACTIVE_RECORDS_SHA256
        ),
        "sampler_preflight_sha256": "8" * 64,
        "arm_contract_sha256": "7" * 64,
        "protected_partitions_sha256": "6" * 64,
        "resume_contract_sha256": "5" * 64,
        "storage_preflight_sha256": "4" * 64,
        "initial_rng_identity": _initial_rng_identity(helper),
        "guard_sets": {
            lesson: {
                "count": index + 1,
                "sha256": f"{index + 1:064x}",
                "rule": None,
            }
            for index, lesson in enumerate(
                (
                    "navigate/full",
                    "unlock/u0-visible",
                    "unlock/u1-local",
                    "unlock/u2-separated",
                )
            )
        },
        "storage_caps": {
            "lineage_cap_bytes": helper.LINEAGE_CAP_BYTES,
            "cohort_scientific_cap_bytes": helper.COHORT_SCIENTIFIC_CAP_BYTES,
            "media_cap_bytes": helper.MEDIA_CAP_BYTES,
            "combined_planned_cap_bytes": helper.COMBINED_PLANNED_CAP_BYTES,
        },
    }


def test_u2s_launcher_has_valid_zsh_syntax_without_running() -> None:
    zsh = shutil.which("zsh")
    if zsh is None:
        pytest.skip("zsh is unavailable")
    subprocess.run(
        [zsh, "-n", str(LAUNCHER)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert os.access(LAUNCHER, os.X_OK)
    assert os.access(HELPER, os.X_OK)


def test_u2s_launcher_is_fixed_sequential_and_fail_closed() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for text in (
        "set -euo pipefail",
        'run_root="$dungeon_root/u2s-ablation-r1-20260723"',
        'media_root="$dungeon_root/u2s-ablation-r1-media-20260723"',
        'training_protocol="dungeon-apprentice-v0.2-u2s-stability-ablation"',
        'training_tag="u2s-stability-ablation-v0.2-u2s-r1-20260723"',
        'trainer_module="dungeon_apprentice.v02_u2s"',
        'dashboard_module="dungeon_apprentice.v02_u2s_dashboard"',
        "dashboard_port=8787",
        "git status --porcelain=v1 --untracked-files=all",
        "git cat-file -t",
        "os.path.ismount(volume)",
        "metadata.st_dev == parent_metadata.st_dev",
        "assert_no_active_neural_trainer",
        "assert_u2s_qualification",
        "u2_trainer_supervisor.py",
        "/usr/bin/caffeinate -ims",
        "trap record_launcher_exit EXIT",
        "trap request_launcher_stop INT TERM",
        "--cohort-contract",
        "--qualification-report",
        "--dashboard-pid",
        '[[ -e "$media_directory" || -L "$media_directory" ]]',
        "accepts no options; interrupted cohorts must restart all four arms",
        "http://127.0.0.1:$dashboard_port/",
    ):
        assert text in source
    assert source.count("assert_source_unchanged") >= 3
    assert source.count("assert_free_space") >= 3
    assert source.count("assert_frozen_file") >= 6
    assert "--resume" not in source
    qualification = source.index("assert_u2s_qualification")
    root_creation = source.index('/bin/mkdir -m 0755 "$run_root"')
    manifest_start = source.index('manifest_command "${start_arguments[@]}"')
    media_reuse_guard = source.index(
        '[[ -e "$media_directory" || -L "$media_directory" ]]'
    )
    media_creation = source.index('/bin/mkdir -m 0755 "$media_directory"')
    media_ancestor_check = source.index(
        'assert_regular_ancestor_chain "$media_directory/placeholder" "$dungeon_root"'
    )
    trainer_start = source.index('"$repository/.venv/bin/python" "$trainer_supervisor"')
    assert qualification < root_creation < trainer_start
    assert (
        manifest_start
        < media_reuse_guard
        < media_creation
        < media_ancestor_check
        < trainer_start
    )
    final_stop = source.rindex("stop_caffeine\nmanifest_command finalize")
    finalization = source.rindex("manifest_command finalize")
    final_trap_release = source.rindex("launcher_finalized=true")
    assert final_stop < finalization < final_trap_release


def test_manifest_freezes_deep_matched_contract_and_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    root = tmp_path / "cohort"
    media = tmp_path / "media"
    root.mkdir()
    media.mkdir()
    source_commit = "a" * 40
    tag_object = "b" * 40
    qualification = _qualification(
        helper,
        source_commit=source_commit,
        tag_object=tag_object,
    )
    monkeypatch.setattr(
        helper,
        "_verified_qualification_binding",
        lambda **_kwargs: qualification,
    )

    state = helper.create_cohort(
        root,
        media_root=media,
        source_commit=source_commit,
        tag_object=tag_object,
    )
    contract = json.loads((root / "cohort-contract.json").read_text())
    assert state["phase"] == "ready"
    assert [arm["id"] for arm in state["arms"]] == list(helper.ARM_ORDER)
    assert contract["parent"]["checkpoint_sha256"] == helper.PARENT_CHECKPOINT_SHA256
    assert (
        contract["parent"]["policy_tensor_sha256"]
        == helper.PARENT_POLICY_TENSOR_SHA256
    )
    assert (
        contract["parent"]["optimizer_state_sha256"]
        == helper.PARENT_OPTIMIZER_STATE_SHA256
    )
    assert contract["qualification"] == qualification
    assert contract["matched_design"]["algorithm_seed"] == helper.ALGORITHM_SEED
    assert contract["matched_design"]["worker_streams"] == list(
        helper.WORKER_STREAMS
    )
    assert contract["matched_design"]["storage_caps"] == {
        "lineage_cap_bytes": 2 * 1024**3,
        "cohort_scientific_cap_bytes": 6 * 1024**3,
        "media_cap_bytes": 10 * 1024**3,
        "combined_planned_cap_bytes": 16 * 1024**3,
        "minimum_free_gib": 25,
    }
    assert [arm["intervention"]["name"] for arm in contract["arms"]] == list(
        helper.ARM_ORDER
    )
    assert contract["arms"][0]["intervention"]["clip_range"] == 0.2
    assert contract["arms"][1]["intervention"]["clip_range"] == 0.1
    assert contract["arms"][2]["intervention"]["no_effect_reward"]["amount"] == -0.01
    assert contract["arms"][3]["intervention"]["target_kl"] == 0.015
    assert contract["matched_design"]["checkpoint_promotable"] is False
    assert contract["matched_design"]["interruption"] == {
        "resumable": False,
        "cohort_disposition": "operationally_incomplete",
        "replacement_requires_all_four_fresh_arms": True,
        "replacement_requires_new_source_commit": True,
        "replacement_requires_new_annotated_tag": True,
        "replacement_requires_new_protocol_attempt_id": True,
        "replacement_requires_new_cohort_root": True,
    }

    helper.start_arm(
        root,
        source_commit=source_commit,
        tag_object=tag_object,
        arm_id="control",
    )
    with pytest.raises(helper.U2sManifestError, match="already active"):
        helper.start_arm(
            root,
            source_commit=source_commit,
            tag_object=tag_object,
            arm_id="conservative",
        )
    arm_dir = root / "control"
    arm_dir.mkdir()
    (arm_dir / "status.json").write_text(
        json.dumps(
            {
                "protocol": helper.PROTOCOL,
                "phase": "completed",
                "arm": "control",
                "source": {"commit": source_commit, "dirty": False},
                "parent_checkpoint_sha256": helper.PARENT_CHECKPOINT_SHA256,
                "action_cap": helper.ACTION_CAP,
                "child_trained_actions": helper.ACTION_CAP,
                "remaining_action_budget": 0,
                "terminal_eligible": True,
            }
        )
    )
    monkeypatch.setattr(
        helper,
        "_verified_arm_terminal",
        lambda *_args, **_kwargs: (
            {
                "arm": "control",
                "verdict": "stable_mechanism_candidate",
                "mechanism_selection_eligible": True,
            },
            {},
        ),
    )
    helper.finish_arm(
        root,
        source_commit=source_commit,
        tag_object=tag_object,
        arm_id="control",
        outcome="completed",
        trainer_exit_code=0,
    )
    plan = helper.next_plan(
        root,
        source_commit=source_commit,
        tag_object=tag_object,
    )
    assert plan["arm"] == "conservative"


def test_manifest_rejects_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    root = tmp_path / "cohort"
    media = tmp_path / "media"
    root.mkdir()
    media.mkdir()
    source_commit = "c" * 40
    tag_object = "d" * 40
    qualification = _qualification(
        helper,
        source_commit=source_commit,
        tag_object=tag_object,
    )
    monkeypatch.setattr(
        helper,
        "_verified_qualification_binding",
        lambda **_kwargs: qualification,
    )
    helper.create_cohort(
        root,
        media_root=media,
        source_commit=source_commit,
        tag_object=tag_object,
    )
    contract_path = root / "cohort-contract.json"
    contract = json.loads(contract_path.read_text())
    contract["arms"][0]["intervention"]["clip_range"] = 0.19
    contract_path.write_text(json.dumps(contract))
    with pytest.raises(helper.U2sManifestError, match="contract differs"):
        helper.next_plan(
            root,
            source_commit=source_commit,
            tag_object=tag_object,
        )


def test_manifest_pre_status_crash_closes_the_nonresumable_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    root = tmp_path / "cohort"
    media = tmp_path / "media"
    root.mkdir()
    media.mkdir()
    source_commit = "a" * 40
    tag_object = "b" * 40
    qualification = _qualification(
        helper,
        source_commit=source_commit,
        tag_object=tag_object,
    )
    monkeypatch.setattr(
        helper,
        "_verified_qualification_binding",
        lambda **_kwargs: qualification,
    )
    helper.create_cohort(
        root,
        media_root=media,
        source_commit=source_commit,
        tag_object=tag_object,
    )
    helper.start_arm(
        root,
        source_commit=source_commit,
        tag_object=tag_object,
        arm_id="control",
    )

    helper.finish_arm(
        root,
        source_commit=source_commit,
        tag_object=tag_object,
        arm_id="control",
        outcome="crashed",
        trainer_exit_code=1,
    )

    state = json.loads((root / "cohort.json").read_text())
    assert state["phase"] == "operationally_incomplete"
    assert state["active_arm"] is None
    assert state["arms"][0]["state"] == "crashed"
    assert state["arms"][0]["attempts"] == [
        {
            "index": 0,
            "state": "crashed",
            "started_at": state["arms"][0]["attempts"][0]["started_at"],
            "finished_at": state["arms"][0]["attempts"][0]["finished_at"],
            "trainer_exit_code": 1,
        }
    ]
    assert not (root / "control" / "status.json").exists()
    with pytest.raises(helper.U2sManifestError, match="terminal operationally_incomplete"):
        helper.next_plan(
            root,
            source_commit=source_commit,
            tag_object=tag_object,
        )


def test_manifest_normalizes_real_qualification_public_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    source_commit = "e" * 40
    tag_object = "f" * 40
    protocol_path = REPOSITORY / helper.PROTOCOL_DOCUMENT
    protocol_digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    counts = {
        "navigate/full": 76_565,
        "unlock/u0-visible": 79,
        "unlock/u1-local": 7_419,
        "unlock/u2-separated": 15_368,
    }

    class Evidence:
        def public_dict(self) -> dict[str, object]:
            return {
                "report": helper.QUALIFICATION_REPORT,
                "report_sha256": "1" * 64,
                "checksum": f"{helper.QUALIFICATION_REPORT}.sha256",
                "source_commit": source_commit,
                "tag": helper.TAG_NAME,
                "tag_object": tag_object,
                "tag_payload_sha256": "9" * 64,
                "protocol_document_sha256": protocol_digest,
                "verdict": "qualified",
                "guard_mapping_sha256": (
                    helper.QUALIFIED_GUARD_MAPPING_SHA256
                ),
                "u2r_terminal_active_records_sha256": (
                    helper.U2R_TERMINAL_ACTIVE_RECORDS_SHA256
                ),
                "sampler_preflight_sha256": "8" * 64,
                "arm_contract_sha256": "7" * 64,
                "protected_partitions_sha256": "6" * 64,
                "resume_contract_sha256": "5" * 64,
                "storage_preflight_sha256": "4" * 64,
                "initial_rng_identity": _initial_rng_identity(helper),
                "guard_sets": {
                    lesson: {
                        "exact_layouts": count,
                        "exact_layout_set_sha256": f"{index + 2:064x}",
                        "rule": f"fixed-rule-{index}",
                    }
                    for index, (lesson, count) in enumerate(counts.items())
                },
                "storage_caps": {
                    "per_arm_bytes": helper.LINEAGE_CAP_BYTES,
                    "scientific_cohort_bytes": helper.COHORT_SCIENTIFIC_CAP_BYTES,
                    "media_bytes": helper.MEDIA_CAP_BYTES,
                    "combined_bytes": helper.COMBINED_PLANNED_CAP_BYTES,
                },
            }

        def verified_report(self) -> dict[str, object]:
            return {
                "guards": {
                    "applied_mapping_sha256": (
                        helper.QUALIFIED_GUARD_MAPPING_SHA256
                    ),
                    "u2r_r1_extension": {
                        "terminal_active_records_sha256": (
                            helper.U2R_TERMINAL_ACTIVE_RECORDS_SHA256
                        )
                    },
                }
            }

    monkeypatch.setattr(
        v02_u2s_qualify,
        "verify_u2s_qualification",
        lambda *_args, **_kwargs: Evidence(),
    )
    binding = helper._verified_qualification_binding(
        source_commit=source_commit,
        tag_object=tag_object,
    )
    assert {
        lesson: guard["count"]
        for lesson, guard in binding["guard_sets"].items()
    } == counts
    assert binding["storage_caps"] == {
        "lineage_cap_bytes": helper.LINEAGE_CAP_BYTES,
        "cohort_scientific_cap_bytes": helper.COHORT_SCIENTIFIC_CAP_BYTES,
        "media_cap_bytes": helper.MEDIA_CAP_BYTES,
        "combined_planned_cap_bytes": helper.COMBINED_PLANNED_CAP_BYTES,
    }


def test_terminal_summary_preserves_fixed_tail_counts() -> None:
    helper = _load_helper()
    exams: list[dict[str, object]] = []
    for index in range(1, helper.EXAM_COUNT + 1):
        thresholds = {
            "at_least_1": index,
            "at_least_3": index + 1,
            "at_least_10": index + 2,
            "at_least_32": index + 3,
        }
        exams.append(
            {
                "child_trained_actions": index * helper.EVALUATION_EVERY,
                "practice_profile": "normal",
                "allocation_valid": True,
                "lessons": {
                    lesson_id: {
                        "successes": 80,
                        "panel_successes": [40, 40],
                        "mean_ineffective_interactions": 1.0,
                        "ineffective_tail": {
                            "ineffective_threshold_counts": thresholds
                        },
                        "max_repeated_identical_interaction_run": 2,
                        "max_identical_visible_no_effect_streak": 1,
                    }
                    for lesson_id in helper.LESSON_IDS
                },
                "case_diagnostics": {
                    "max_case_ineffective_interactions": 4,
                    "cases_with_ineffective_at_least_10": (
                        thresholds["at_least_10"]
                    ),
                    "max_repeated_identical_interaction_run": 2,
                    "max_identical_visible_no_effect_streak": 1,
                    "ineffective_tail": {
                        "ineffective_threshold_counts": thresholds
                    },
                },
            }
        )

    curve = helper._exam_curve(exams)
    summary = helper._terminal_summary(
        curve,
        grade={"eligible": True},
    )

    assert summary["ineffective_threshold_counts"] == {
        "at_least_1": 31 + 32 + 30,
        "at_least_3": 32 + 33 + 31,
        "at_least_10": 33 + 34 + 32,
        "at_least_32": 34 + 35 + 33,
    }
    assert summary["cases_with_ineffective_at_least_10"] == 33 + 34 + 32


def test_factorial_contrasts_use_exact_frozen_formulas() -> None:
    helper = _load_helper()
    offsets = {
        "control": 0.0,
        "conservative": 2.0,
        "no-effect": 4.0,
        "combined": 10.0,
    }
    metric_bases = {
        metric: float(index + 1)
        for index, metric in enumerate(helper.FACTORIAL_METRIC_ORDER)
    }

    terminal: dict[str, dict[str, object]] = {}
    for arm_id in helper.ARM_ORDER:
        values = {
            metric: base + offsets[arm_id]
            for metric, base in metric_bases.items()
        }
        terminal[arm_id] = {
            "u2_successes_mean": values["u2_successes_mean"],
            "u0_u1_u2_mean_ineffective_interactions": {
                "unlock/u0-visible": values[
                    "u0_mean_ineffective_interactions"
                ],
                "unlock/u1-local": values[
                    "u1_mean_ineffective_interactions"
                ],
                "unlock/u2-separated": values[
                    "u2_mean_ineffective_interactions"
                ],
            },
            "ineffective_threshold_counts": {
                "at_least_1": values["ineffective_cases_at_least_1"],
                "at_least_3": values["ineffective_cases_at_least_3"],
                "at_least_10": values[
                    "ineffective_cases_at_least_10"
                ],
                "at_least_32": values[
                    "ineffective_cases_at_least_32"
                ],
            },
            "worst_case_ineffective_interactions": values[
                "worst_case_ineffective_interactions"
            ],
            "worst_repeated_identical_interaction_run": values[
                "worst_repeated_identical_interaction_run"
            ],
            "worst_identical_visible_no_effect_streak": values[
                "worst_identical_visible_no_effect_streak"
            ],
        }

    result = helper._factorial_contrasts(terminal)

    assert result["schema_version"] == 1
    assert result["descriptive_only"] is True
    assert result["population_inference_authorized"] is False
    assert result["cell_order"] == list(helper.ARM_ORDER)
    assert result["metric_order"] == list(helper.FACTORIAL_METRIC_ORDER)
    assert result["effect_order"] == list(helper.FACTORIAL_EFFECT_ORDER)
    assert result["effect_definitions"] == {
        "conservative_main": (
            "mean(conservative,combined)-mean(control,no-effect)"
        ),
        "no_effect_main": (
            "mean(no-effect,combined)-mean(control,conservative)"
        ),
        "interaction": "(combined-conservative)-(no-effect-control)",
    }
    assert list(result["metrics"]) == list(helper.FACTORIAL_METRIC_ORDER)
    for metric, base in metric_bases.items():
        assert result["metrics"][metric]["cells"] == {
            arm_id: base + offsets[arm_id] for arm_id in helper.ARM_ORDER
        }
        assert result["metrics"][metric]["effects"] == {
            "conservative_main": 4.0,
            "no_effect_main": 6.0,
            "interaction": 4.0,
        }


def test_matched_initial_rng_identity_requires_exact_four_arm_equality() -> None:
    helper = _load_helper()
    identity = _initial_rng_identity(helper)
    arms = {
        arm_id: {"initial_rng_identity": identity}
        for arm_id in helper.ARM_ORDER
    }
    matched = helper._matched_initial_rng_identity(arms)
    assert matched["identical_across_all_arms"] is True
    assert set(matched["arm_aggregate_sha256"].values()) == {
        identity["aggregate_sha256"]
    }

    changed = json.loads(json.dumps(arms))
    changed["combined"]["initial_rng_identity"]["components"][
        "python_random_sha256"
    ] = "f" * 64
    changed["combined"]["initial_rng_identity"]["aggregate_sha256"] = (
        helper._canonical_sha256(
            changed["combined"]["initial_rng_identity"]["components"]
        )
    )
    with pytest.raises(helper.U2sManifestError, match="identical RNG"):
        helper._matched_initial_rng_identity(changed)


def test_process_closeout_rejects_supervisor_or_caffeinate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper()
    root = tmp_path / "cohort"
    root.mkdir()
    dashboard_pid = 41001
    dashboard_command = (
        f"/venv/python -m dungeon_apprentice.v02_u2s_dashboard {root} "
        "--port 8787"
    )

    def completed(stdout: str) -> SimpleNamespace:
        return SimpleNamespace(stdout=stdout)

    def clean_run(arguments: list[str], **_kwargs: object) -> SimpleNamespace:
        if arguments[0] == "/bin/ps":
            return completed(f"{dashboard_pid} 1 {dashboard_command}\n")
        return completed(f"{dashboard_pid}\n")

    monkeypatch.setattr(helper.subprocess, "run", clean_run)
    evidence = helper._process_closeout_evidence(
        root=root,
        dashboard_pid=dashboard_pid,
    )
    assert evidence["orphan_free"] is True
    assert evidence["dashboard"]["explicitly_excepted"] is True

    def dirty_run(arguments: list[str], **_kwargs: object) -> SimpleNamespace:
        if arguments[0] == "/bin/ps":
            return completed(
                f"{dashboard_pid} 1 {dashboard_command}\n"
                "41002 1 /venv/python scripts/u2_trainer_supervisor.py\n"
                "41003 1 /usr/bin/caffeinate -ims\n"
            )
        return completed(f"{dashboard_pid}\n")

    monkeypatch.setattr(helper.subprocess, "run", dirty_run)
    with pytest.raises(helper.U2sManifestError, match="remains at closeout"):
        helper._process_closeout_evidence(
            root=root,
            dashboard_pid=dashboard_pid,
        )
