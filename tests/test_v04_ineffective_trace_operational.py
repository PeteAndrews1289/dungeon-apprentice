from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
HELPER = REPOSITORY / "scripts" / "v04_ineffective_trace_manifest.py"


def _load_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "v04_ineffective_trace_manifest_fixture",
        HELPER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _cohort(
    tmp_path: Path,
) -> tuple[ModuleType, Path, Path, str, str, dict[str, object]]:
    helper = _load_helper()
    root = tmp_path / "cohort"
    media = tmp_path / "media"
    qualification_root = tmp_path / "qualification"
    root.mkdir()
    media.mkdir()
    qualification_root.mkdir()
    qualification = qualification_root / "report.json"
    _write_json(qualification, {"fixture": True})
    digest = hashlib.sha256(qualification.read_bytes()).hexdigest()
    checksum = qualification.with_name("report.json.sha256")
    checksum.write_text(f"{digest}  report.json\n", encoding="ascii")
    source = "a" * 40
    tag_object = "b" * 40
    public: dict[str, object] = {
        "report": str(qualification),
        "report_sha256": digest,
        "checksum": str(checksum),
        "source_commit": source,
        "tag": helper.TAG_NAME,
        "tag_object": tag_object,
        "tag_payload_sha256": "1" * 64,
        "verdict": "qualified",
        "protocol_document_sha256": "2" * 64,
        "guard_mapping_sha256": "3" * 64,
        "sampler_preflight_sha256": "4" * 64,
        "architecture_contract_sha256": "5" * 64,
        "smoke_evidence_sha256": "6" * 64,
        "protected_partitions_sha256": "7" * 64,
        "r3_terminal_evidence_sha256": "8" * 64,
        "storage_caps": {
            "per_arm_bytes": helper.LINEAGE_CAP_BYTES,
            "scientific_cohort_bytes": helper.COHORT_SCIENTIFIC_CAP_BYTES,
            "media_bytes": helper.MEDIA_CAP_BYTES,
            "combined_bytes": helper.COMBINED_PLANNED_CAP_BYTES,
        },
    }
    helper._verified_qualification_binding = (
        lambda *_args, **_kwargs: json.loads(json.dumps(public))
    )
    helper.create_cohort(
        root,
        media_root=media,
        qualification_report=qualification,
        source_commit=source,
        tag_object=tag_object,
    )
    return helper, root, media, source, tag_object, public


def test_v04_manifest_freezes_new_identity_and_r3_boundary(
    tmp_path: Path,
) -> None:
    helper, root, media, source, tag_object, public = _cohort(tmp_path)
    contract = json.loads(
        (root / "cohort-contract.json").read_text(encoding="utf-8")
    )

    assert helper.PROTOCOL == (
        "dungeon-apprentice-v0.4-ineffective-trace-architecture"
    )
    assert helper.COHORT_ID == "v0.4-ineffective-trace-stage-a-20260724"
    assert helper.TAG_NAME == (
        "ineffective-trace-architecture-v0.4-stage-a-20260724"
    )
    assert helper.DEFAULT_DASHBOARD_PORT == 8792
    assert helper.ARM_ORDER == ("trace-sham", "ineffective-trace")
    assert contract["roots"] == {
        "cohort": str(root),
        "media": str(media),
    }
    assert contract["predecessor_boundary"] == {
        "r3_terminal_evidence_sha256": public[
            "r3_terminal_evidence_sha256"
        ],
        "r3_checkpoint_reuse_authorized": False,
        "restarts_both_arms_from_confirmed_u1": True,
    }
    assert contract["observation"]["ineffective_trace"] == {
        "shape": [1],
        "dtype": "float32",
        "trace_cap": 9,
        "normalization_divisor": 9.0,
        "action_identity_exposed": False,
        "visible_outcome_identity_exposed": False,
        "reset_sentinel": 0.0,
        "privileged_state": False,
    }
    assert contract["matched_design"]["resumable"] is False
    assert contract["matched_design"]["checkpoint_promotable"] is False
    assert helper.next_plan(
        root,
        source_commit=source,
        tag_object=tag_object,
    )["arm"] == "trace-sham"


def test_v04_manifest_enforces_fixed_order_and_terminal_interruption(
    tmp_path: Path,
) -> None:
    helper, root, _media, source, tag_object, _public = _cohort(tmp_path)

    with pytest.raises(helper.V04ManifestError, match="fixed sham-first"):
        helper.start_arm(
            root,
            source_commit=source,
            tag_object=tag_object,
            arm_id="ineffective-trace",
        )

    helper.start_arm(
        root,
        source_commit=source,
        tag_object=tag_object,
        arm_id="trace-sham",
    )
    result = helper.finish_arm(
        root,
        source_commit=source,
        tag_object=tag_object,
        arm_id="trace-sham",
        outcome="crashed",
        trainer_exit_code=17,
    )
    assert result["phase"] == "operationally_incomplete"
    with pytest.raises(helper.V04ManifestError, match="non-resumable"):
        helper.next_plan(
            root,
            source_commit=source,
            tag_object=tag_object,
        )


def test_v04_manifest_aborts_post_arm_closeout_without_resume(
    tmp_path: Path,
) -> None:
    helper, root, _media, source, tag_object, _public = _cohort(tmp_path)
    state_path = root / "cohort.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["phase"] = "awaiting_closeout"
    for arm in state["arms"]:
        arm["state"] = "completed"
    _write_json(state_path, state)

    result = helper.abort_closeout(
        root,
        source_commit=source,
        tag_object=tag_object,
        launcher_exit_code=23,
    )
    assert result == {
        "phase": "operationally_incomplete",
        "launcher_exit_code": 23,
    }
    closed = json.loads(state_path.read_text(encoding="utf-8"))
    assert closed["closeout_abort"] == {
        "reason": "launcher_abnormal_exit_before_terminal_finalization",
        "launcher_exit_code": 23,
        "at": closed["closeout_abort"]["at"],
        "process_closeout_already_sealed": False,
    }
    with pytest.raises(helper.V04ManifestError, match="non-resumable"):
        helper.next_plan(
            root,
            source_commit=source,
            tag_object=tag_object,
        )
    with pytest.raises(helper.V04ManifestError, match="complete arms"):
        helper.finalize_cohort(
            root,
            source_commit=source,
            tag_object=tag_object,
        )


def test_v04_manifest_rejects_contract_tamper(
    tmp_path: Path,
) -> None:
    helper, root, _media, source, tag_object, _public = _cohort(tmp_path)
    path = root / "cohort-contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract["predecessor_boundary"]["r3_checkpoint_reuse_authorized"] = True
    _write_json(path, contract)

    with pytest.raises(helper.V04ManifestError, match="contract changed"):
        helper.next_plan(
            root,
            source_commit=source,
            tag_object=tag_object,
        )


@pytest.mark.parametrize(
    ("command", "role"),
    (
        (
            "/repo/.venv/bin/python -m "
            "dungeon_apprentice.v04_ineffective_trace_train train-arm",
            "trainer",
        ),
        (
            "/repo/.venv/bin/python -m "
            "dungeon_apprentice.v03_action_effect_train train-arm",
            "trainer",
        ),
        (
            "/repo/.venv/bin/dungeon-train-v04-ineffective-trace "
            "--arm ineffective-trace",
            "trainer",
        ),
        (
            "/usr/bin/python3 /repo/.venv/bin/dungeon-train-v03-action-effect "
            "--arm action-effect",
            "trainer",
        ),
        (
            "/repo/.venv/bin/python /repo/scripts/u2_trainer_supervisor.py "
            "-- /repo/.venv/bin/python -m "
            "dungeon_apprentice.v04_ineffective_trace_train",
            "supervisor",
        ),
        ("/usr/bin/caffeinate -ims", "caffeinate"),
        (
            "/repo/.venv/bin/python -m "
            "dungeon_apprentice.v04_ineffective_trace_dashboard",
            "dashboard",
        ),
        (
            "/bin/zsh -c echo dungeon_apprentice.v04_ineffective_trace_train",
            "unrelated",
        ),
    ),
)
def test_v04_process_roles_use_leading_argv(
    command: str,
    role: str,
) -> None:
    helper = _load_helper()
    assert helper._process_role(command) == role


def test_v04_process_closeout_rejects_training_roles() -> None:
    helper = _load_helper()
    with pytest.raises(helper.V04ManifestError, match="prohibited"):
        helper._process_closeout_evidence(
            [
                {
                    "pid": 101,
                    "ppid": 1,
                    "command": (
                        "/repo/.venv/bin/python -m "
                        "dungeon_apprentice.v04_ineffective_trace_train"
                    ),
                }
            ]
        )

    evidence = helper._process_closeout_evidence(
        [
            {
                "pid": 102,
                "ppid": 1,
                "command": (
                    "/repo/.venv/bin/python -m "
                    "dungeon_apprentice.v04_ineffective_trace_dashboard"
                ),
            }
        ]
    )
    assert evidence["verdict"] == "clear"
    assert evidence["dashboard_processes"][0]["role"] == "dashboard"
