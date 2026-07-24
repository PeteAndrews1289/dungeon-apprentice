# ruff: noqa: E501
"""Read-only dashboard for the v0.3 matched action-effect study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from dungeon_apprentice import v03_action_effect as v03

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8789
PROTOCOL = v03.PROTOCOL
QUALIFICATION_PROTOCOL = PROTOCOL
COHORT_ID = "v0.3-action-effect-stage-a-r1-20260724"
TAG_NAME = "action-effect-architecture-v0.3-stage-a-r1-20260724"
ACTION_CAP = v03.CHILD_ACTION_BUDGET
ARM_ORDER = ("sham", "action-effect")
LESSONS = (
    ("navigate/full", "Navigate"),
    ("unlock/u0-visible", "Visible Unlock"),
    ("unlock/u1-local", "Local Unlock"),
    ("unlock/u2-separated", "Separated Unlock"),
)
LESSON_IDS = frozenset(lesson for lesson, _label in LESSONS)
STATUS_MAX_BYTES = 8 * 1024 * 1024
REPORT_MAX_BYTES = 64 * 1024 * 1024
FRAME_MAX_BYTES = 16 * 1024 * 1024
PROCESS_CLOSEOUT_NAME = "process-closeout.json"
PROCESS_CLOSEOUT_MAX_BYTES = 4 * 1024 * 1024
PROCESS_INVENTORY_MAX_BYTES = 8 * 1024 * 1024
PROCESS_INVENTORY_MAX_ROWS = 4096
PROCESS_SCAN_TIMEOUT_SECONDS = 10
PROCESS_SCAN_COMMAND = ["/bin/ps", "-axo", "pid=,ppid=,command="]
PROCESS_ROLES = frozenset(
    {"unrelated", "dashboard", "trainer", "supervisor", "caffeinate"}
)
PROHIBITED_PROCESS_ROLES = frozenset(
    {"trainer", "supervisor", "caffeinate"}
)
HEARTBEAT_STALE_SECONDS = 180
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

V03_DASHBOARD_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dungeon Apprentice — v0.3</title>
<style>:root{color-scheme:dark;--bg:#071019;--panel:#101f2a;--line:#2b485d;--ink:#eef8ff;
--muted:#92aabc;--cyan:#60dceb;--green:#63df98;--amber:#efc46b;--red:#f07979;--violet:#a995ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#173d54,var(--bg) 42%);
color:var(--ink);font:15px/1.45 ui-rounded,system-ui,sans-serif}main{max-width:1220px;margin:auto;padding:28px 20px 58px}
header{display:flex;justify-content:space-between;gap:18px;align-items:flex-end;margin-bottom:18px}.eyebrow{color:var(--cyan);
font-weight:850;letter-spacing:.15em;text-transform:uppercase;font-size:12px}h1{font-size:clamp(30px,5vw,52px);
line-height:1;margin:7px 0 0;letter-spacing:-.05em}.fresh{color:var(--muted)}.dot{display:inline-block;width:9px;height:9px;
border-radius:50%;background:var(--amber);margin-right:8px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.card{background:color-mix(in srgb,var(--panel) 95%,transparent);border:1px solid var(--line);border-radius:17px;
padding:17px;box-shadow:0 18px 48px #0004}.metric span,.tiny{font-size:11px;color:var(--muted);text-transform:uppercase;
letter-spacing:.08em}.metric b{display:block;font-size:23px;margin-top:5px}.boundary{margin:12px 0;display:grid;
grid-template-columns:2fr 1fr;gap:12px}.arms{display:grid;grid-template-columns:1fr 1fr;gap:12px}.arm h2{display:flex;
justify-content:space-between;margin:0 0 8px}.badge{font-size:11px;color:var(--muted);border:1px solid var(--line);
padding:4px 8px;border-radius:999px}.bar{height:10px;background:#06111a;border-radius:20px;overflow:hidden;margin:13px 0 5px}
.fill{height:100%;background:linear-gradient(90deg,var(--violet),var(--cyan))}.numbers{display:grid;
grid-template-columns:repeat(3,1fr);gap:7px;margin:13px 0}.number{background:#091720;border-radius:10px;padding:8px}
.number b{font-size:18px;display:block}.frame{width:100%;aspect-ratio:1;background:#06111a;border:1px solid var(--line);
border-radius:11px;object-fit:contain;image-rendering:pixelated}.checks{display:grid;gap:5px;margin-top:12px}.check{display:flex;
justify-content:space-between;border-top:1px solid var(--line);padding-top:6px;font-size:12px}.good{color:var(--green)}
.bad{color:var(--red)}.note{color:var(--muted)}table{width:100%;border-collapse:collapse;min-width:760px}th,td{
padding:9px;border-bottom:1px solid var(--line);text-align:left;font-variant-numeric:tabular-nums}th{color:var(--muted);
font-size:11px;text-transform:uppercase}.table{margin-top:12px;overflow:auto}@media(max-width:800px){.grid,.arms,
.boundary{grid-template-columns:1fr 1fr}.numbers{grid-template-columns:1fr 1fr}}@media(max-width:540px){
main{padding:20px 12px 42px}.grid,.arms,.boundary{grid-template-columns:1fr}header{align-items:flex-start;flex-direction:column}}
</style></head><body><main><header><div><div class="eyebrow">v0.3 · matched architecture study</div>
<h1>Did action-effect context help?</h1></div><div class="fresh"><i class="dot"></i><span id="fresh">Loading…</span></div>
</header><section class="grid"><div class="card metric"><span>Cohort</span><b id="phase">—</b></div>
<div class="card metric"><span>Active twin</span><b id="active">—</b></div><div class="card metric">
<span>Actions trained</span><b id="actions">—</b></div><div class="card metric"><span>Frozen exams</span>
<b id="exams">—</b></div></section><section class="boundary"><div class="card"><span class="tiny">Stage-A decision</span>
<h2 id="decision">Pending both full-budget twins</h2><div id="decisionNote" class="note"></div></div><div class="card">
<span class="tiny">Scientific boundary</span><div class="note">Sham calibrates the comparison. Only the action-effect
architecture can qualify, no Stage-A checkpoint can be reused, and this study cannot open U3.</div></div></section>
<section id="arms" class="arms"></section><section class="card table"><h2>Exam curve</h2><table><thead><tr>
<th>Twin</th><th>Action boundary</th><th>U2</th><th>Worst ineffective</th><th>Interaction run</th><th>Gate</th>
</tr></thead><tbody id="rows"></tbody></table></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const f=n=>Number.isFinite(Number(n))?Number(n).toLocaleString():'—';let delay=2000;
function arm(a){let p=Math.max(0,Math.min(100,100*Number(a.trained_actions||0)/1048576)),c=a.context_encoder||{},
x=a.context_summary||{},r=a.first_rollout||{},checks=(a.terminal_checks||[]).map((z,i)=>`<div class="check"><span>Terminal exam ${i+1}</span>
<b class="${z.passed?'good':'bad'}">${z.passed?'Pass':'Fail'}</b></div>`).join('')||'<div class="note">Terminal window pending.</div>';
let frame=a.frame_url?`<img class="frame" src="${e(a.frame_url)}&v=${f(a.frame_revision)}" alt="${e(a.label)} live frame">`:
'<div class="frame note" style="display:grid;place-items:center">Frame pending</div>';return `<article class="card arm"><h2>${e(a.label)}
<span class="badge">${e(a.state)}</span></h2><div class="tiny">${a.context_enabled?'Real previous-action + visible outcome':'All-zero sham context'}</div>
<div class="bar"><div class="fill" style="width:${p}%"></div></div><div class="note">${f(a.trained_actions)} / 1,048,576 actions ·
${f(a.exams_completed)} exams</div><div class="numbers"><div class="number"><span class="tiny">Encoder norm</span><b>${f(c.terminal_weight_norm??
c.weight_l2_norm)}</b></div><div class="number"><span class="tiny">Context active</span><b>${Number.isFinite(Number(x.activation_rate))?
(100*Number(x.activation_rate)).toFixed(1)+'%':'—'}</b></div><div class="number"><span class="tiny">First divergence</span><b>${e(
r.first_divergence||'—')}</b></div></div>${frame}<div class="checks">${checks}</div></article>`}
function render(d){document.getElementById('phase').textContent=String(d.phase||'').replaceAll('_',' ');
document.getElementById('active').textContent=d.active_arm||'None';document.getElementById('actions').textContent=f(d.total_trained_actions);
document.getElementById('exams').textContent=f(d.total_exams);document.getElementById('arms').innerHTML=d.arms.map(arm).join('');
let terminal=d.arms.flatMap(a=>(a.exams||[]).map(x=>({a,x})));document.getElementById('rows').innerHTML=terminal.map(({a,x})=>{
let u=(x.lessons||[]).find(z=>z.lesson_id==='unlock/u2-separated');return `<tr><td>${e(a.label)}</td><td>${f(x.boundary)}</td>
<td>${f(u?.successes)}/80</td><td>${f(x.max_case_ineffective)}</td><td>${f(x.max_interaction_run)}</td>
<td class="${x.passed?'good':'bad'}">${x.passed?'Pass':'Fail'}</td></tr>`}).join('')||
'<tr><td colspan="6" class="note">No frozen exam has completed.</td></tr>';let selected=d.selected_architecture;
document.getElementById('decision').textContent=selected?'Action-effect architecture selected':
(d.terminal_verdict==='architecture_failed'?'Architecture did not qualify':'Pending both full-budget twins');
document.getElementById('decisionNote').textContent=selected?'The architecture definition may proceed only to a separately preregistered replication.':
(d.terminal_verdict==='architecture_failed'?'Stage A stops here; U3 remains closed.':'No intermediate peak can decide this study.');
let dot=document.querySelector('.dot'),bad=['operationally_incomplete','integrity_failed'].includes(d.phase),stale=d.heartbeat_stale;
let color=bad?'var(--red)':stale?'var(--amber)':'var(--green)';dot.style.background=color;document.getElementById('fresh').textContent=
stale?'Active heartbeat delayed':`Updated ${f(Math.floor(Number(d.heartbeat_age_seconds||0)))}s ago`;
delay=['completed','operationally_incomplete','integrity_failed'].includes(d.phase)?60000:2000}
let failures=0;async function poll(){try{let q=await fetch(`/api/v03.json?v=${Date.now()}`,{cache:'no-store'});
if(!q.ok)throw Error(q.status);render(await q.json());failures=0}catch(_){failures++;document.getElementById('fresh').textContent=`Dashboard retry ${failures}`}
setTimeout(poll,delay)}poll();</script></body></html>"""


class V03DashboardError(RuntimeError):
    """Raised when dashboard evidence is unsafe or inconsistent."""


class V03DashboardServer(ThreadingHTTPServer):
    """HTTP server bound to one authenticated v0.3 cohort root."""

    daemon_threads = True
    allow_reuse_address = False
    run_root: Path
    snapshot_lock: threading.Lock
    authentication: V03DashboardAuthentication


@dataclass(frozen=True)
class V03DashboardAuthentication:
    """Immutable evidence authenticated once before the server binds."""

    root: Path
    contract_sha256: str
    qualification_binding_sha256: str
    qualification_report: Path
    qualification_report_sha256: str
    qualification_checksum: Path
    source_commit: str
    tag_object: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_regular_bytes(
    path: Path,
    *,
    maximum: int,
    label: str,
) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise V03DashboardError(f"{label} is missing or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > maximum
        ):
            raise V03DashboardError(f"{label} has an invalid file bound")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise V03DashboardError(f"{label} changed while read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise V03DashboardError(f"{label} grew while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _absolute_directory(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        raise ValueError("v0.3 dashboard root must be absolute")
    try:
        metadata = candidate.lstat()
    except FileNotFoundError as error:
        raise ValueError("v0.3 dashboard root does not exist") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("v0.3 dashboard root must be a regular directory")
    return candidate.resolve(strict=True)


def _read_json(
    path: Path,
    *,
    maximum: int = STATUS_MAX_BYTES,
) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise V03DashboardError(
            f"dashboard evidence is missing: {path.name}"
        ) from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or metadata.st_size > maximum
    ):
        raise V03DashboardError(
            f"dashboard evidence is unsafe: {path.name}"
        )
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V03DashboardError(
            f"cannot read dashboard evidence {path.name}"
        ) from error
    if not isinstance(payload, dict):
        raise V03DashboardError(
            f"dashboard evidence is not an object: {path.name}"
        )
    return payload


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _timestamp_age(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        return None
    return max(
        0.0,
        (
            datetime.now(UTC) - timestamp.astimezone(UTC)
        ).total_seconds(),
    )


def _source_commit(payload: Mapping[str, Any]) -> Any:
    source = payload.get("source")
    return source.get("commit") if isinstance(source, Mapping) else source


def _lesson_rows(exam: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    lessons = exam.get("lessons")
    if isinstance(lessons, Mapping):
        return [
            {**value, "lesson_id": value.get("lesson_id", lesson_id)}
            for lesson_id, value in lessons.items()
            if isinstance(value, Mapping)
        ]
    if isinstance(lessons, list):
        return [
            value for value in lessons if isinstance(value, Mapping)
        ]
    return []


def _group_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        boundary = _integer(
            record.get(
                "child_trained_actions",
                record.get("trained_actions"),
            )
        )
        if boundary is None or record.get("lesson_id") not in LESSON_IDS:
            continue
        grouped.setdefault(boundary, []).append(record)
    exams: list[dict[str, Any]] = []
    for boundary, rows in sorted(grouped.items()):
        exams.append(
            {
                "child_trained_actions": boundary,
                "allocation_valid": (
                    len(rows) == len(LESSON_IDS)
                    and all(row.get("allocation_valid") is True for row in rows)
                ),
                "practice_profile": (
                    rows[0].get("practice_profile")
                    if rows
                    and len({row.get("practice_profile") for row in rows}) == 1
                    else None
                ),
                "lessons": {
                    str(row["lesson_id"]): dict(row) for row in rows
                },
                "case_diagnostics": {
                    "max_case_ineffective_interactions": max(
                        (
                            int(row.get("max_ineffective_interactions", -1))
                            for row in rows
                        ),
                        default=-1,
                    ),
                    "cases_with_ineffective_at_least_10": sum(
                        int(
                            (
                                row.get("ineffective_tail")
                                if isinstance(
                                    row.get("ineffective_tail"), Mapping
                                )
                                else {}
                            )
                            .get("ineffective_threshold_counts", {})
                            .get("at_least_10", 0)
                        )
                        for row in rows
                    ),
                    "max_repeated_identical_interaction_run": max(
                        (
                            int(
                                row.get(
                                    "max_repeated_identical_interaction_run",
                                    -1,
                                )
                            )
                            for row in rows
                        ),
                        default=-1,
                    ),
                },
            }
        )
    return exams


def _evaluation_records(
    directory: Path,
    status: Mapping[str, Any],
) -> list[dict[str, Any]]:
    for key in ("exam_records", "evaluation_history", "evaluations"):
        value = status.get(key)
        if isinstance(value, list):
            records = [
                item for item in value if isinstance(item, Mapping)
            ]
            grouped = _group_records(records)
            return grouped or [dict(item) for item in records]
    ledger = directory / "evaluations.jsonl"
    if ledger.exists() or ledger.is_symlink():
        metadata = ledger.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > REPORT_MAX_BYTES
        ):
            raise V03DashboardError("v0.3 evaluations ledger is unsafe")
        rows: list[Mapping[str, Any]] = []
        try:
            for line in ledger.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise V03DashboardError(
                        "v0.3 evaluation row is not an object"
                    )
                rows.append(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise V03DashboardError(
                "cannot read v0.3 evaluation ledger"
            ) from error
        return _group_records(rows)
    latest = status.get("latest_evaluation")
    return [dict(latest)] if isinstance(latest, Mapping) else []


def _public_exam(exam: Mapping[str, Any]) -> dict[str, Any]:
    rows = _lesson_rows(exam)
    lessons = [
        {
            "lesson_id": lesson_id,
            "lesson_label": next(
                label for item, label in LESSONS if item == lesson_id
            ),
            "successes": _integer(row.get("successes")),
            "episodes": _integer(row.get("episodes")),
            "panel_successes": (
                [_integer(value) for value in row["panel_successes"]]
                if isinstance(row.get("panel_successes"), list)
                else []
            ),
            "mean_ineffective_interactions": _number(
                row.get("mean_ineffective_interactions")
            ),
        }
        for lesson_id in (item[0] for item in LESSONS)
        if (
            row := next(
                (
                    value
                    for value in rows
                    if value.get("lesson_id") == lesson_id
                ),
                None,
            )
        )
        is not None
    ]
    diagnostics = exam.get("case_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    try:
        passed = all(v03.frozen_u2s.exam_stability_checks(exam).values())
    except (RuntimeError, TypeError, ValueError):
        passed = False
    return {
        "boundary": _integer(
            exam.get(
                "child_trained_actions",
                exam.get("boundary"),
            )
        ),
        "lessons": lessons,
        "max_case_ineffective": _integer(
            diagnostics.get("max_case_ineffective_interactions")
        ),
        "max_interaction_run": _integer(
            diagnostics.get(
                "max_repeated_identical_interaction_run"
            )
        ),
        "passed": passed,
    }


def _safe_relative_frame(
    root: Path,
    arm_id: str,
    status: Mapping[str, Any],
    *,
    allow_missing: bool,
) -> Path:
    if arm_id not in ARM_ORDER:
        raise FileNotFoundError("unknown v0.3 arm")
    frame = status.get("latest_frame")
    if isinstance(frame, Mapping):
        relative_value = frame.get("path", "frames/latest.png")
        declared_sha256 = frame.get("sha256")
    else:
        relative_value = "frames/latest.png"
        declared_sha256 = None
    if not isinstance(relative_value, str):
        raise V03DashboardError("v0.3 frame path is invalid")
    relative = Path(relative_value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.suffix.lower() != ".png"
    ):
        raise V03DashboardError("v0.3 frame path escapes its arm")
    arm_root = root / arm_id
    candidate = arm_root / relative
    if not candidate.exists() and not candidate.is_symlink():
        if allow_missing:
            return candidate
        raise FileNotFoundError("v0.3 frame is not yet available")
    metadata = candidate.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 8
        or metadata.st_size > FRAME_MAX_BYTES
    ):
        raise V03DashboardError("v0.3 frame is unsafe")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(arm_root.resolve(strict=True))
    except ValueError as error:
        raise V03DashboardError("v0.3 frame escapes its arm") from error
    with resolved.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise V03DashboardError("v0.3 frame is not a PNG")
    if (
        declared_sha256 is not None
        and (
            not isinstance(declared_sha256, str)
            or SHA256_PATTERN.fullmatch(declared_sha256) is None
            or _sha256(resolved) != declared_sha256
        )
    ):
        raise V03DashboardError("v0.3 frame checksum changed")
    return resolved


def resolve_v03_frame(run_root: Path, *, arm_id: str) -> Path:
    """Resolve only one fixed status-bound latest PNG."""

    root = _absolute_directory(run_root)
    if arm_id not in ARM_ORDER:
        raise FileNotFoundError("unknown v0.3 arm")
    status = _read_json(root / arm_id / "status.json")
    return _safe_relative_frame(
        root,
        arm_id,
        status,
        allow_missing=False,
    )


def _authenticate_live_qualification(
    report_path: Path,
    *,
    source_commit: str,
    tag_object: str,
) -> dict[str, Any]:
    try:
        from dungeon_apprentice.v03_action_effect_qualify import (
            verify_action_effect_qualification,
        )
    except ImportError as error:
        raise V03DashboardError(
            "v0.3 qualification verifier is unavailable"
        ) from error
    repository = Path(__file__).resolve().parents[2]
    try:
        evidence = verify_action_effect_qualification(
            report_path,
            expected_source_commit=source_commit,
            expected_tag_object=tag_object,
            repository=repository,
        )
        public = evidence.public_dict()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise V03DashboardError(
            f"live v0.3 qualification authentication failed: {error}"
        ) from error
    if not isinstance(public, dict):
        raise V03DashboardError(
            "v0.3 qualification verifier returned no public evidence"
        )
    return json.loads(json.dumps(public))


def _validate_contract(
    contract: Mapping[str, Any],
    *,
    root: Path,
) -> None:
    source = contract.get("source")
    preregistration = contract.get("preregistration")
    qualification = contract.get("qualification")
    replacement = contract.get("replacement")
    parent = contract.get("parent")
    roots = contract.get("roots")
    matched = contract.get("matched_design")
    process_contract = contract.get("process_closeout")
    arms = contract.get("arms")
    if (
        contract.get("schema_version") != 1
        or contract.get("protocol") != PROTOCOL
        or contract.get("cohort_id") != COHORT_ID
        or not isinstance(source, Mapping)
        or source.get("dirty") is not False
        or not isinstance(preregistration, Mapping)
        or preregistration.get("tag") != TAG_NAME
        or preregistration.get("peeled_commit") != source.get("commit")
        or not isinstance(qualification, Mapping)
        or qualification.get("verdict") != "qualified"
        or qualification.get("source_commit") != source.get("commit")
        or qualification.get("tag") != TAG_NAME
        or qualification.get("tag_object")
        != preregistration.get("tag_object")
        or not isinstance(replacement, Mapping)
        or replacement.get("failed_attempt_evidence_sha256")
        != qualification.get("failed_stage_a_attempt_sha256")
        or replacement.get("failed_attempt_resume_authorized") is not False
        or replacement.get("failed_attempt_root_reuse_authorized")
        is not False
        or replacement.get("restarts_both_arms_from_confirmed_u1")
        is not True
        or not isinstance(parent, Mapping)
        or parent.get("checkpoint_sha256")
        != v03.PARENT_CHECKPOINT_SHA256
        or not isinstance(roots, Mapping)
        or roots.get("cohort") != str(root)
        or not isinstance(matched, Mapping)
        or matched.get("arm_order") != list(ARM_ORDER)
        or matched.get("fresh_only") is not True
        or matched.get("resumable") is not False
        or matched.get("checkpoint_promotable") is not False
        or not isinstance(process_contract, Mapping)
        or process_contract.get("required_before_finalization") is not True
        or process_contract.get("evidence_path") != PROCESS_CLOSEOUT_NAME
        or process_contract.get("collector") != PROCESS_SCAN_COMMAND
        or process_contract.get("maximum_rows")
        != PROCESS_INVENTORY_MAX_ROWS
        or process_contract.get("maximum_inventory_bytes")
        != PROCESS_INVENTORY_MAX_BYTES
        or process_contract.get("prohibited_roles")
        != sorted(PROHIBITED_PROCESS_ROLES)
        or process_contract.get("dashboard_may_remain") is not True
        or not isinstance(arms, list)
        or [arm.get("id") for arm in arms if isinstance(arm, Mapping)]
        != list(ARM_ORDER)
    ):
        raise V03DashboardError("immutable v0.3 cohort contract is invalid")
    qualification_path = Path(str(qualification.get("report")))
    live = _authenticate_live_qualification(
        qualification_path,
        source_commit=str(source.get("commit")),
        tag_object=str(preregistration.get("tag_object")),
    )
    if live != dict(qualification):
        raise V03DashboardError("live v0.3 qualification binding changed")


def _startup_authentication(root: Path) -> V03DashboardAuthentication:
    """Deep-authenticate immutable source evidence before serving."""

    contract_path = root / "cohort-contract.json"
    contract = _read_json(contract_path)
    _validate_contract(contract, root=root)
    qualification = contract["qualification"]
    authentication = V03DashboardAuthentication(
        root=root,
        contract_sha256=_sha256(contract_path),
        qualification_binding_sha256=_canonical_sha256(qualification),
        qualification_report=Path(str(qualification["report"])),
        qualification_report_sha256=str(
            qualification["report_sha256"]
        ),
        qualification_checksum=Path(str(qualification["checksum"])),
        source_commit=str(contract["source"]["commit"]),
        tag_object=str(contract["preregistration"]["tag_object"]),
    )
    _verify_cached_authentication(
        contract,
        root=root,
        contract_sha256=authentication.contract_sha256,
        authentication=authentication,
    )
    return authentication


def _verify_cached_authentication(
    contract: Mapping[str, Any],
    *,
    root: Path,
    contract_sha256: str,
    authentication: V03DashboardAuthentication,
) -> None:
    """Recheck immutable bytes without repeating the deep verifier."""

    qualification = contract.get("qualification")
    source = contract.get("source")
    preregistration = contract.get("preregistration")
    if (
        root != authentication.root
        or contract_sha256 != authentication.contract_sha256
        or not isinstance(qualification, Mapping)
        or _canonical_sha256(qualification)
        != authentication.qualification_binding_sha256
        or Path(str(qualification.get("report")))
        != authentication.qualification_report
        or qualification.get("report_sha256")
        != authentication.qualification_report_sha256
        or Path(str(qualification.get("checksum")))
        != authentication.qualification_checksum
        or not isinstance(source, Mapping)
        or source.get("commit") != authentication.source_commit
        or not isinstance(preregistration, Mapping)
        or preregistration.get("tag_object")
        != authentication.tag_object
    ):
        raise V03DashboardError(
            "cached v0.3 qualification contract binding changed"
        )
    report_bytes = _bounded_regular_bytes(
        authentication.qualification_report,
        maximum=STATUS_MAX_BYTES,
        label="v0.3 qualification report",
    )
    digest = hashlib.sha256(report_bytes).hexdigest()
    if digest != authentication.qualification_report_sha256:
        raise V03DashboardError(
            "cached v0.3 qualification report checksum changed"
        )
    try:
        checksum_fields = (
            _bounded_regular_bytes(
                authentication.qualification_checksum,
                maximum=256,
                label="v0.3 qualification checksum",
            )
            .decode("ascii")
            .strip()
            .split()
        )
    except UnicodeDecodeError as error:
        raise V03DashboardError(
            "cached v0.3 qualification checksum is not ASCII"
        ) from error
    if checksum_fields != [
        digest,
        authentication.qualification_report.name,
    ]:
        raise V03DashboardError(
            "cached v0.3 qualification checksum changed"
        )


def _verified_process_closeout(
    root: Path,
    *,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    binding = state.get("process_closeout")
    if not isinstance(binding, Mapping):
        raise V03DashboardError(
            "terminal process closeout evidence is missing"
        )
    path = root / PROCESS_CLOSEOUT_NAME
    evidence = _read_json(path, maximum=PROCESS_CLOSEOUT_MAX_BYTES)
    inventory = evidence.get("inventory")
    dashboards = evidence.get("dashboard_processes")
    collector = evidence.get("collector")
    policy = evidence.get("policy")
    if (
        set(evidence)
        != {
            "schema_version",
            "protocol",
            "cohort_id",
            "kind",
            "captured_at",
            "collector",
            "policy",
            "inventory",
            "inventory_sha256",
            "prohibited_matches",
            "dashboard_processes",
            "verdict",
        }
        or evidence.get("schema_version") != 1
        or evidence.get("protocol") != PROTOCOL
        or evidence.get("cohort_id") != COHORT_ID
        or evidence.get("kind") != "terminal_process_closeout"
        or evidence.get("verdict") != "clear"
        or not isinstance(evidence.get("captured_at"), str)
        or not isinstance(collector, Mapping)
        or collector.get("command") != PROCESS_SCAN_COMMAND
        or collector.get("timeout_seconds")
        != PROCESS_SCAN_TIMEOUT_SECONDS
        or collector.get("maximum_inventory_bytes")
        != PROCESS_INVENTORY_MAX_BYTES
        or collector.get("maximum_rows") != PROCESS_INVENTORY_MAX_ROWS
        or not isinstance(inventory, list)
        or not inventory
        or len(inventory) > PROCESS_INVENTORY_MAX_ROWS
        or collector.get("observed_rows") != len(inventory)
        or not isinstance(policy, Mapping)
        or policy.get("prohibited_roles")
        != sorted(PROHIBITED_PROCESS_ROLES)
        or policy.get("dashboard_may_remain") is not True
        or policy.get("commands_redacted_to_sha256") is not True
        or evidence.get("prohibited_matches") != []
        or not isinstance(dashboards, list)
        or _canonical_sha256(inventory)
        != evidence.get("inventory_sha256")
    ):
        raise V03DashboardError(
            "terminal process closeout evidence changed"
        )
    seen: set[int] = set()
    expected_dashboards: list[dict[str, Any]] = []
    for item in inventory:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"pid", "ppid", "command_sha256", "role"}
            or isinstance(item.get("pid"), bool)
            or not isinstance(item.get("pid"), int)
            or item["pid"] <= 0
            or item["pid"] in seen
            or isinstance(item.get("ppid"), bool)
            or not isinstance(item.get("ppid"), int)
            or item["ppid"] < 0
            or item.get("role") not in PROCESS_ROLES
            or item.get("role") in PROHIBITED_PROCESS_ROLES
            or SHA256_PATTERN.fullmatch(
                str(item.get("command_sha256", ""))
            )
            is None
        ):
            raise V03DashboardError(
                "terminal process closeout inventory changed"
            )
        seen.add(item["pid"])
        if item["role"] == "dashboard":
            expected_dashboards.append(dict(item))
    if (
        [item["pid"] for item in inventory] != sorted(seen)
        or dashboards != expected_dashboards
    ):
        raise V03DashboardError(
            "terminal process closeout ordering changed"
        )
    expected_binding = {
        "path": PROCESS_CLOSEOUT_NAME,
        "sha256": _sha256(path),
        "inventory_sha256": evidence["inventory_sha256"],
        "captured_at": evidence["captured_at"],
        "observed_rows": len(inventory),
        "dashboard_pids": [item["pid"] for item in dashboards],
        "prohibited_count": 0,
        "verdict": "clear",
    }
    if dict(binding) != expected_binding:
        raise V03DashboardError(
            "terminal process closeout binding changed"
        )
    return expected_binding


def _verified_closeout(
    root: Path,
    *,
    contract: Mapping[str, Any],
    state: Mapping[str, Any],
    contract_sha256: str,
    process_closeout: Mapping[str, Any],
) -> dict[str, Any]:
    terminal = state.get("terminal_report")
    if not isinstance(terminal, Mapping):
        raise V03DashboardError("v0.3 terminal state has no report binding")
    report_path = root / "report.json"
    integrity_path = root / "report.integrity.json"
    report = _read_json(report_path, maximum=REPORT_MAX_BYTES)
    integrity = _read_json(integrity_path)
    finalization_recheck = report.get("process_finalization_recheck")
    if (
        _sha256(report_path) != terminal.get("sha256")
        or _sha256(integrity_path) != terminal.get("integrity_sha256")
        or terminal.get("process_closeout_sha256")
        != process_closeout.get("sha256")
        or integrity.get("protocol") != PROTOCOL
        or integrity.get("cohort_id") != COHORT_ID
        or integrity.get("report_sha256") != terminal.get("sha256")
        or integrity.get("cohort_contract_sha256") != contract_sha256
        or integrity.get("process_closeout_sha256")
        != process_closeout.get("sha256")
        or integrity.get("process_inventory_sha256")
        != process_closeout.get("inventory_sha256")
        or report.get("protocol") != PROTOCOL
        or report.get("cohort_id") != COHORT_ID
        or report.get("cohort_contract_sha256") != contract_sha256
        or report.get("process_closeout") != dict(process_closeout)
        or not isinstance(finalization_recheck, Mapping)
        or _canonical_sha256(finalization_recheck)
        != integrity.get("process_finalization_recheck_sha256")
        or finalization_recheck.get("prohibited_count") != 0
        or finalization_recheck.get("verdict") != "clear"
        or _source_commit(report) != contract["source"]["commit"]
        or report.get("qualification", {}).get("report_sha256")
        != contract["qualification"]["report_sha256"]
        or report.get("checkpoint_rule", {}).get(
            "stage_a_checkpoint_reuse_authorized"
        )
        is not False
        or report.get("checkpoint_rule", {}).get("u3_authorized")
        is not False
    ):
        raise V03DashboardError("v0.3 terminal closeout is unauthenticated")
    arm_records: dict[str, list[Mapping[str, Any]]] = {}
    for arm_id in ARM_ORDER:
        arm_report = _read_json(
            root / arm_id / "report.json",
            maximum=REPORT_MAX_BYTES,
        )
        controller = arm_report.get("controller")
        exams = (
            controller.get("exam_records")
            if isinstance(controller, Mapping)
            else None
        )
        if not isinstance(exams, list) or not all(
            isinstance(exam, Mapping) for exam in exams
        ):
            raise V03DashboardError(
                f"{arm_id} terminal exams are unavailable"
            )
        arm_records[arm_id] = exams
    selection = v03.select_architecture(arm_records)
    if (
        report.get("selection") != selection
        or report.get("verdict") != selection["verdict"]
        or (
            (report.get("selected_architecture") is None)
            != (selection["selected_architecture"] is None)
        )
    ):
        raise V03DashboardError(
            "dashboard recomputation differs from v0.3 closeout"
        )
    return report


def load_v03_snapshot(
    run_root: Path,
    *,
    authentication: V03DashboardAuthentication | None = None,
) -> dict[str, Any]:
    """Load one bounded public v0.3 view and reject provenance drift."""

    root = _absolute_directory(run_root)
    contract_path = root / "cohort-contract.json"
    contract = _read_json(contract_path)
    state = _read_json(root / "cohort.json")
    contract_sha256 = _sha256(contract_path)
    if authentication is None:
        _validate_contract(contract, root=root)
    else:
        _verify_cached_authentication(
            contract,
            root=root,
            contract_sha256=contract_sha256,
            authentication=authentication,
        )
    arms = state.get("arms")
    process_binding = state.get("process_closeout")
    if (
        state.get("schema_version") != 1
        or state.get("protocol") != PROTOCOL
        or state.get("cohort_id") != COHORT_ID
        or state.get("contract_sha256") != contract_sha256
        or state.get("source_commit") != contract["source"]["commit"]
        or state.get("tag") != TAG_NAME
        or state.get("tag_object")
        != contract["preregistration"]["tag_object"]
        or state.get("qualification_sha256")
        != contract["qualification"]["report_sha256"]
        or state.get("phase")
        not in {
            "ready",
            "training",
            "awaiting_closeout",
            "completed",
            "operationally_incomplete",
            "integrity_failed",
        }
        or not isinstance(arms, list)
        or [arm.get("id") for arm in arms if isinstance(arm, Mapping)]
        != list(ARM_ORDER)
        or sum(arm.get("state") == "training" for arm in arms) > 1
        or (
            state.get("phase") == "completed"
            and not isinstance(process_binding, Mapping)
        )
        or (
            process_binding is not None
            and (
                not isinstance(process_binding, Mapping)
                or state.get("phase")
                not in {
                    "awaiting_closeout",
                    "completed",
                    "integrity_failed",
                }
            )
        )
    ):
        raise V03DashboardError(
            "mutable v0.3 state differs from its contract"
        )
    process_closeout = (
        _verified_process_closeout(root, state=state)
        if process_binding is not None
        else None
    )
    closeout = (
        _verified_closeout(
            root,
            contract=contract,
            state=state,
            contract_sha256=contract_sha256,
            process_closeout=process_closeout,
        )
        if state["phase"] == "completed"
        else None
    )
    public_arms: list[dict[str, Any]] = []
    ages: list[float] = []
    active_age: float | None = None
    for definition, arm_state in zip(
        contract["arms"], arms, strict=True
    ):
        arm_id = definition["id"]
        directory = root / arm_id
        status: dict[str, Any] = {}
        if directory.exists() or directory.is_symlink():
            if directory.is_symlink() or not directory.is_dir():
                raise V03DashboardError(
                    f"unsafe v0.3 arm directory: {arm_id}"
                )
            status = _read_json(directory / "status.json")
            if (
                status.get("protocol") != PROTOCOL
                or status.get("cohort_id") != COHORT_ID
                or status.get("arm") != arm_id
                or _source_commit(status) != contract["source"]["commit"]
                or status.get("qualification_sha256")
                != contract["qualification"]["report_sha256"]
                or status.get("cohort_contract_sha256")
                != contract_sha256
                or status.get("parent_checkpoint_sha256")
                != v03.PARENT_CHECKPOINT_SHA256
                or _integer(status.get("action_cap")) != ACTION_CAP
            ):
                raise V03DashboardError(
                    f"{arm_id} status differs from the cohort contract"
                )
            age = _timestamp_age(status.get("updated_at"))
            if age is not None:
                ages.append(age)
                if state.get("active_arm") == arm_id:
                    active_age = age
        elif arm_state.get("state") != "pending":
            raise V03DashboardError(
                f"{arm_id} state has no arm directory"
            )
        if arm_state.get("state") == "completed":
            report = _read_json(
                directory / "report.json",
                maximum=REPORT_MAX_BYTES,
            )
            controller = report.get("controller")
            raw_exams = (
                controller.get("exam_records")
                if isinstance(controller, Mapping)
                else None
            )
            if (
                not isinstance(raw_exams, list)
                or not all(
                    isinstance(exam, Mapping) for exam in raw_exams
                )
            ):
                raise V03DashboardError(
                    f"{arm_id} report has no exam records"
                )
            terminal = arm_state.get("terminal")
            if (
                not isinstance(terminal, Mapping)
                or _sha256(directory / "report.json")
                != terminal.get("report_sha256")
            ):
                raise V03DashboardError(
                    f"{arm_id} terminal report changed"
                )
            context_encoder = report.get("context_encoder")
            context_summary = report.get("context_summary")
            first_rollout = report.get("first_rollout")
        else:
            raw_exams = _evaluation_records(directory, status) if status else []
            context_encoder = status.get("encoder_latest", {})
            context_summary = status.get("context_metrics", {})
            first_rollout = {
                "aggregate_sha256": status.get(
                    "first_rollout_identity_sha256"
                ),
                "verified": status.get("first_rollout_verified") is True,
                "first_divergence": (
                    "after_first_optimizer"
                    if isinstance(context_encoder, Mapping)
                    and int(
                        context_encoder.get(
                            "weight_nonzero_parameters",
                            0,
                        )
                    )
                    > 0
                    else "not_observed"
                ),
            }
        exams = [_public_exam(exam) for exam in raw_exams]
        eligible: bool | None = None
        terminal_checks = exams[-3:]
        if arm_state.get("state") == "completed":
            try:
                eligible = v03.grade_terminal(
                    arm_id,
                    raw_exams,
                ).eligible
            except (RuntimeError, TypeError, ValueError) as error:
                raise V03DashboardError(
                    f"{arm_id} terminal grade is invalid"
                ) from error
        frame_available = False
        if status and _integer(status.get("frame_revision")):
            frame_available = _safe_relative_frame(
                root,
                arm_id,
                status,
                allow_missing=True,
            ).exists()
        public_arms.append(
            {
                "id": arm_id,
                "label": definition.get("label", arm_id),
                "state": arm_state.get("state"),
                "phase": status.get("phase"),
                "context_enabled": definition.get("context_enabled"),
                "updated_at": status.get("updated_at"),
                "trained_actions": (
                    _integer(status.get("child_trained_actions")) or 0
                ),
                "collected_actions": (
                    _integer(status.get("child_collected_actions")) or 0
                ),
                "remaining_action_budget": _integer(
                    status.get("remaining_action_budget")
                ),
                "optimizer_updates": _integer(
                    status.get("optimizer_updates")
                ),
                "exams_completed": len(exams),
                "exams": exams,
                "terminal_checks": terminal_checks,
                "eligible": eligible,
                "context_encoder": (
                    dict(context_encoder)
                    if isinstance(context_encoder, Mapping)
                    else {}
                ),
                "context_summary": (
                    dict(context_summary)
                    if isinstance(context_summary, Mapping)
                    else {}
                ),
                "first_rollout": (
                    dict(first_rollout)
                    if isinstance(first_rollout, Mapping)
                    else {}
                ),
                "frame_revision": (
                    _integer(status.get("frame_revision")) or 0
                ),
                "frame_url": (
                    f"/api/frame?arm={arm_id}"
                    if frame_available
                    else None
                ),
            }
        )
    heartbeat_age = (
        active_age if state.get("active_arm") is not None else max(ages, default=0.0)
    )
    selected = (
        closeout.get("selected_architecture")
        if isinstance(closeout, Mapping)
        else None
    )
    return {
        "protocol": PROTOCOL,
        "cohort_id": COHORT_ID,
        "loaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "phase": state["phase"],
        "active_arm": state.get("active_arm"),
        "source_commit": contract["source"]["commit"],
        "tag": TAG_NAME,
        "tag_object": contract["preregistration"]["tag_object"],
        "qualification_report_sha256": contract["qualification"][
            "report_sha256"
        ],
        "contract_sha256": contract_sha256,
        "heartbeat_age_seconds": heartbeat_age,
        "heartbeat_stale": (
            state.get("active_arm") is not None
            and (
                active_age is None
                or active_age > HEARTBEAT_STALE_SECONDS
            )
        ),
        "total_trained_actions": sum(
            arm["trained_actions"] for arm in public_arms
        ),
        "total_exams": sum(
            arm["exams_completed"] for arm in public_arms
        ),
        "arms": public_arms,
        "selected_architecture": selected,
        "terminal_verdict": (
            closeout.get("verdict")
            if isinstance(closeout, Mapping)
            else None
        ),
        "process_closeout": process_closeout,
        "checkpoint_promotable": False,
        "u3_authorized": False,
        "interruption_disclosure": (
            "Stage A has no resume path. An interruption makes this matched "
            "root operationally incomplete; replacement requires a new "
            "source, tag, attempt identity, root, and two fresh twins."
        ),
    }


class V03DashboardHandler(BaseHTTPRequestHandler):
    server: V03DashboardServer
    server_version = "DungeonApprenticeV03Dashboard/1"

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _send(
        self,
        content: bytes,
        *,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        send_body: bool,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self'; "
            "style-src 'unsafe-inline'; script-src 'unsafe-inline'",
        )
        self.end_headers()
        if send_body:
            self.wfile.write(content)

    def _error(
        self,
        status: HTTPStatus,
        message: str,
        *,
        send_body: bool,
    ) -> None:
        self._send(
            json.dumps({"error": message}).encode(),
            content_type="application/json; charset=utf-8",
            status=status,
            send_body=send_body,
        )

    def _handle(self, *, send_body: bool) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send(
                V03_DASHBOARD_HTML.encode(),
                content_type="text/html; charset=utf-8",
                send_body=send_body,
            )
            return
        if parsed.path == "/api/v03.json":
            try:
                with self.server.snapshot_lock:
                    payload = load_v03_snapshot(
                        self.server.run_root,
                        authentication=self.server.authentication,
                    )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                self._error(
                    HTTPStatus.CONFLICT,
                    str(error),
                    send_body=send_body,
                )
                return
            self._send(
                json.dumps(payload, separators=(",", ":")).encode(),
                content_type="application/json; charset=utf-8",
                send_body=send_body,
            )
            return
        if parsed.path == "/api/frame":
            arm = parse_qs(parsed.query).get("arm", [""])[0]
            try:
                with self.server.snapshot_lock:
                    frame = resolve_v03_frame(
                        self.server.run_root,
                        arm_id=arm,
                    )
                    content = frame.read_bytes()
            except FileNotFoundError:
                self._error(
                    HTTPStatus.NOT_FOUND,
                    "frame unavailable",
                    send_body=send_body,
                )
                return
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                self._error(
                    HTTPStatus.CONFLICT,
                    str(error),
                    send_body=send_body,
                )
                return
            self._send(
                content,
                content_type="image/png",
                send_body=send_body,
            )
            return
        self._error(
            HTTPStatus.NOT_FOUND,
            "not found",
            send_body=send_body,
        )

    def do_GET(self) -> None:
        self._handle(send_body=True)

    def do_HEAD(self) -> None:
        self._handle(send_body=False)


def start_v03_dashboard(
    run_root: Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> V03DashboardServer:
    """Build a read-only dashboard server after one authenticated load."""

    root = _absolute_directory(run_root)
    authentication = _startup_authentication(root)
    load_v03_snapshot(root, authentication=authentication)
    server = V03DashboardServer(
        (host, int(port)),
        V03DashboardHandler,
    )
    server.run_root = root
    server.snapshot_lock = threading.Lock()
    server.authentication = authentication
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = start_v03_dashboard(
        args.run_root,
        host=args.host,
        port=args.port,
    )
    print(f"http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
