# ruff: noqa: E501
"""Read-only comparison dashboard for the four-arm U2-S mechanism study."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import threading
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
PROTOCOL = "dungeon-apprentice-v0.2-u2s-stability-ablation"
COHORT_ID = "v0.2-u2s-ablation-r1-20260723"
ACTION_CAP = 1_048_576
ARM_ORDER = ("control", "conservative", "no-effect", "combined")
LESSONS = (
    ("navigate/full", "Navigate", 68, 34, False),
    ("unlock/u0-visible", "Visible Unlock", 68, 32, True),
    ("unlock/u1-local", "Local Unlock", 68, 32, True),
    ("unlock/u2-separated", "Separated Unlock", 72, 34, True),
)
LESSON_IDS = frozenset(item[0] for item in LESSONS)
STATUS_MAX_BYTES = 8 * 1024 * 1024
EVALUATIONS_MAX_BYTES = 64 * 1024 * 1024

U2S_DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dungeon Apprentice — U2-S Stability Lab</title>
<style>
:root{color-scheme:dark;--bg:#071018;--panel:#101d28;--line:#294256;--ink:#edf7ff;
--muted:#91a8bb;--cyan:#5dd7e7;--green:#63df93;--amber:#f1c66d;--red:#ef7777;--violet:#a992ff}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 12% 0,
#15364c 0,var(--bg) 38%);font:15px/1.45 ui-rounded,system-ui,sans-serif}main{max-width:1380px;
margin:auto;padding:28px 20px 60px}header{display:flex;justify-content:space-between;align-items:flex-end;
gap:18px;margin-bottom:20px}.eyebrow{color:var(--cyan);font-size:12px;font-weight:850;letter-spacing:
.16em;text-transform:uppercase;margin-bottom:8px}h1{font-size:clamp(30px,6vw,54px);letter-spacing:
-.05em;line-height:1;margin:0}h2{font-size:18px;margin:0 0 12px}.fresh{display:flex;gap:8px;
align-items:center;color:var(--muted)}.dot{width:9px;height:9px;border-radius:50%;background:var(--amber);
box-shadow:0 0 14px var(--amber)}.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;
margin-bottom:12px}.card{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid
var(--line);border-radius:17px;padding:17px;box-shadow:0 18px 48px #0004}.metric span,.tiny{
color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}.metric b{display:block;
font-size:23px;margin-top:4px}.banner{display:grid;grid-template-columns:2fr 1fr;gap:12px;margin-bottom:
12px}.decision b{font-size:24px}.decision .note{margin-top:7px}.arm-grid{display:grid;
grid-template-columns:repeat(4,1fr);gap:12px}.arm{min-width:0}.arm h2{display:flex;justify-content:
space-between;gap:8px}.badge{font-size:11px;padding:4px 8px;border:1px solid var(--line);border-radius:
999px;color:var(--muted);white-space:nowrap}.badge.good{color:var(--green);border-color:var(--green)}
.badge.bad{color:var(--red);border-color:var(--red)}.bar{height:10px;background:#06101a;border-radius:
20px;overflow:hidden;margin:13px 0 5px}.fill{height:100%;width:0;background:linear-gradient(90deg,
var(--violet),var(--cyan));transition:width .3s}.numbers{display:grid;grid-template-columns:1fr 1fr;
gap:8px;margin:13px 0}.number{background:#09151e;border-radius:10px;padding:9px}.number b{display:block;
font-size:19px}.spark{height:58px;display:flex;align-items:flex-end;gap:2px;margin:12px 0}.spark i{
display:block;flex:1;min-width:2px;background:var(--cyan);border-radius:3px 3px 0 0;opacity:.8}
.checks{display:grid;gap:5px}.check{display:flex;justify-content:space-between;border-top:1px solid
var(--line);padding-top:6px;font-size:12px}.good{color:var(--green)}.bad{color:var(--red)}.pending{
color:var(--muted)}.table-card{margin-top:12px;overflow:auto}table{width:100%;border-collapse:collapse;
min-width:900px}th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums}th{color:var(--muted);font-size:11px;text-transform:uppercase;
letter-spacing:.06em}.note{color:var(--muted)}code{color:var(--cyan)}@media(max-width:1050px){
.arm-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:720px){header{align-items:flex-start;
flex-direction:column}.summary,.banner,.arm-grid{grid-template-columns:1fr 1fr}}@media(max-width:520px){
main{padding:22px 13px 44px}.summary,.banner,.arm-grid{grid-template-columns:1fr}}
</style></head><body><main><header><div><div class="eyebrow">U2-S · matched 2 x 2 mechanism study</div>
<h1>Can stability be learned?</h1></div><div class="fresh"><i class="dot"></i>
<span id="freshness">Loading the cohort…</span></div></header>
<section class="summary"><div class="card metric"><span>Cohort state</span><b id="phase">—</b></div>
<div class="card metric"><span>Arms complete</span><b id="complete">—</b></div>
<div class="card metric"><span>Total learned</span><b id="actions">—</b></div>
<div class="card metric"><span>Active arm</span><b id="active">—</b></div></section>
<section class="banner"><div class="card decision"><span class="tiny">Fixed priority decision</span>
<b id="decision">Pending all four arms</b><div id="decisionNote" class="note"></div></div>
<div class="card"><span class="tiny">Scientific boundary</span><div class="note">This study chooses a
learner configuration only. No arm checkpoint is promotable or eligible to open U3. Any
interruption makes the whole matched cohort operationally incomplete; there is no resume.</div></div>
</section>
<section id="arms" class="arm-grid"></section><section class="card table-card">
<h2>Terminal-window comparison</h2><table><thead><tr><th>Arm</th><th>Boundary</th><th>U2</th>
<th>U2 ineffective</th><th>p95 ineffective</th><th>Worst case</th><th>Cases ≥10</th>
<th>Interaction run</th><th>Visible no-effect streak</th>
<th>Top-two tail share</th><th>Exam gate</th>
</tr></thead><tbody id="rows"></tbody></table></section></main>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',
"'":'&#39;'}[c]));const fmt=n=>Number.isFinite(Number(n))?Number(n).toLocaleString():'—';
const pct=x=>Number.isFinite(Number(x))?`${(100*Number(x)).toFixed(1)}%`:'—';
const phase=p=>String(p||'waiting').replaceAll('_',' ');const dot=document.querySelector('.dot');
function armCard(a){let p=Math.max(0,Math.min(100,100*Number(a.trained_actions||0)/1048576));
let exams=a.exams||[],last=exams.at(-1),u2=last?.lessons?.find(x=>x.lesson_id==='unlock/u2-separated');
let bars=exams.map(x=>{let z=x.lessons?.find(v=>v.lesson_id==='unlock/u2-separated');
return `<i title="${fmt(x.boundary)}: ${fmt(z?.successes)}/80" style="height:${Math.max(3,
100*Number(z?.successes||0)/80)}%"></i>`}).join('');let checks=(a.terminal_checks||[]).map((x,i)=>
`<div class="check"><span>Terminal exam ${i+1}</span><b class="${x.passed?'good':'bad'}">${
x.passed?'Pass':'Fail'}</b></div>`).join('')||'<div class="note">Three terminal exams pending.</div>';
let cls=a.eligible===true?'good':a.eligible===false&&a.state==='completed'?'bad':'';
let interrupted=['interrupted','crashed','integrity_failed'].includes(a.state)?`<div class="note bad">This matched
cohort is terminal and cannot resume. A replacement needs new committed protocol evidence and four fresh arms.</div>`:'';
return `<article class="card arm"><h2>${esc(a.label)}<span class="badge ${cls}">${esc(phase(a.state))}
</span></h2><div class="tiny">${esc(a.ppo_profile)} PPO · feedback ${a.no_effect_penalty?'on':'off'}
</div><div class="bar"><div class="fill" style="width:${p}%"></div></div><div class="note">${
fmt(a.trained_actions)} / 1,048,576 actions</div><div class="numbers"><div class="number"><span
class="tiny">Latest U2</span><b>${fmt(u2?.successes)}/80</b></div><div class="number"><span
class="tiny">Ineffective</span><b>${u2?Number(u2.mean_ineffective_interactions).toFixed(2):'—'}
</b></div><div class="number"><span class="tiny">Worst case</span><b>${fmt(last?.max_case_ineffective)}
</b></div><div class="number"><span class="tiny">Interaction run</span><b>${fmt(
last?.max_interaction_run)}</b></div><div class="number"><span class="tiny">Visible no-effect</span><b>${
fmt(last?.max_no_effect_streak)}</b></div></div><div class="spark">${bars}</div><div class="checks">
${checks}</div>${interrupted}</article>`}
function render(d){document.getElementById('phase').textContent=phase(d.phase);
pollDelay=['completed','operationally_incomplete','integrity_failed'].includes(d.phase)?60000:2000;
document.getElementById('complete').textContent=`${d.arms.filter(x=>x.state==='completed').length}/4`;
document.getElementById('actions').textContent=fmt(d.total_trained_actions);
document.getElementById('active').textContent=d.active_arm||'None';
document.getElementById('arms').innerHTML=d.arms.map(armCard).join('');
let selected=d.selected_mechanism;document.getElementById('decision').textContent=selected?
`${selected.label} selected`:(d.phase==='completed'?'No mechanism qualified':'Pending all four arms');
document.getElementById('decisionNote').textContent=selected?
'Selected by the preregistered simplest-intervention priority. The checkpoint remains development-only.':
(d.phase==='completed'?'The current learner family reached its declared stopping point.':
'The decision remains sealed until every arm reaches its full budget.');
let terminal=d.arms.flatMap(a=>(a.exams||[]).slice(-3).map(x=>({a,x})));
document.getElementById('rows').innerHTML=terminal.map(({a,x})=>{let u2=x.lessons.find(
v=>v.lesson_id==='unlock/u2-separated');return `<tr><td>${esc(a.label)}</td><td>${fmt(x.boundary)}
</td><td>${fmt(u2?.successes)}/80</td><td>${u2?Number(u2.mean_ineffective_interactions).toFixed(2):
'—'}</td><td>${fmt(u2?.ineffective_quantiles?.p95)}</td><td>${fmt(x.max_case_ineffective)}</td><td>${
fmt(x.case_threshold_counts?.at_least_10)}</td><td>${fmt(x.max_interaction_run)}</td><td>${
fmt(x.max_no_effect_streak)}</td><td>${pct(
x.top_two_tail_share)}</td><td class="${x.passed?'good':'bad'}">${x.passed?'Pass':'Fail'}
</td></tr>`}).join('')||'<tr><td colspan="11" class="note">No exam has completed.</td></tr>';
let stale=Boolean(d.heartbeat_stale),bad=['operationally_incomplete','integrity_failed'].includes(d.phase);let color=bad?'var(--red)':
stale?'var(--amber)':'var(--green)';dot.style.background=color;dot.style.boxShadow=`0 0 14px ${
color}`;document.getElementById('freshness').textContent=stale?'Active heartbeat delayed':
`Updated ${fmt(Math.floor(Number(d.heartbeat_age_seconds||0)))}s ago`}
let failures=0,pollDelay=2000;async function poll(){try{let r=await fetch(`/api/u2s.json?v=${Date.now()}`,
{cache:'no-store'});if(!r.ok)throw new Error(r.status);render(await r.json());failures=0}catch(e){
failures++;document.getElementById('freshness').textContent=`Dashboard retry ${failures}`}
setTimeout(poll,pollDelay)}poll();
</script></body></html>"""


class U2sDashboardError(RuntimeError):
    """Raised when dashboard evidence is unsafe or internally inconsistent."""


class U2sDashboardServer(ThreadingHTTPServer):
    """HTTP server carrying one authenticated U2-S cohort root."""

    daemon_threads = True
    allow_reuse_address = False
    run_root: Path
    snapshot_lock: threading.Lock


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _absolute_regular_directory(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        raise ValueError("U2-S dashboard root must be absolute")
    try:
        metadata = candidate.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"U2-S dashboard root does not exist: {candidate}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("U2-S dashboard root must be a regular directory")
    return candidate.resolve(strict=True)


def _read_json(path: Path, *, maximum: int = STATUS_MAX_BYTES) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise U2sDashboardError(f"dashboard evidence is missing: {path.name}") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or metadata.st_size > maximum
    ):
        raise U2sDashboardError(f"dashboard evidence is unsafe: {path.name}")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise U2sDashboardError(f"cannot read dashboard evidence {path.name}: {error}") from error
    if not isinstance(payload, dict):
        raise U2sDashboardError(f"dashboard evidence is not an object: {path.name}")
    return payload


def _status_source(status: Mapping[str, Any]) -> Any:
    source = status.get("source")
    return source.get("commit") if isinstance(source, Mapping) else source


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _lesson_rows(exam: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    lessons = exam.get("lessons")
    if isinstance(lessons, list):
        return [item for item in lessons if isinstance(item, Mapping)]
    if isinstance(lessons, Mapping):
        return [
            {**value, "lesson_id": value.get("lesson_id", key)}
            for key, value in lessons.items()
            if isinstance(value, Mapping)
        ]
    evaluations = exam.get("evaluations")
    if isinstance(evaluations, list):
        return [item for item in evaluations if isinstance(item, Mapping)]
    if exam.get("lesson_id") in LESSON_IDS:
        return [exam]
    return []


def _group_lesson_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        if record.get("counts_toward_arm_stability") is False:
            continue
        boundary = _integer(
            record.get("child_trained_actions", record.get("trained_actions"))
        )
        if (
            boundary is None
            or boundary <= 0
            or record.get("lesson_id") not in LESSON_IDS
        ):
            continue
        grouped.setdefault(boundary, []).append(record)
    exams: list[dict[str, Any]] = []
    for boundary, rows in sorted(grouped.items()):
        allocation_values = [row.get("allocation_valid") for row in rows]
        profiles = {row.get("practice_profile") for row in rows}
        exams.append(
            {
                "boundary": boundary,
                "lessons": rows,
                "allocation_valid": (
                    len(rows) == len(LESSONS)
                    and all(value is True for value in allocation_values)
                ),
                "practice_profile": (
                    next(iter(profiles)) if len(profiles) == 1 else None
                ),
            }
        )
    return exams


def _evaluation_records(directory: Path, status: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = directory / "evaluations.jsonl"
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > EVALUATIONS_MAX_BYTES
        ):
            raise U2sDashboardError("U2-S evaluations ledger is unsafe")
        rows: list[Mapping[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise U2sDashboardError("U2-S evaluation row is not an object")
                rows.append(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise U2sDashboardError(f"cannot read U2-S evaluations ledger: {error}") from error
        return _group_lesson_records(rows)
    for key in ("evaluation_history", "evaluations"):
        value = status.get(key)
        if isinstance(value, list):
            mappings = [item for item in value if isinstance(item, Mapping)]
            grouped = _group_lesson_records(mappings)
            if grouped:
                return grouped
            return [dict(item) for item in mappings]
    latest = status.get("latest_evaluation")
    if isinstance(latest, Mapping):
        rows = _lesson_rows(latest)
        grouped = _group_lesson_records(rows)
        return grouped or [dict(latest)]
    return []


def _tail_value(
    rows: Iterable[Mapping[str, Any]],
    *,
    keys: tuple[str, ...],
) -> float | None:
    found: list[float] = []
    for row in rows:
        tail = row.get("loop_tail")
        for key in keys:
            value = row.get(key)
            if value is None and isinstance(tail, Mapping):
                value = tail.get(key)
            number = _number(value)
            if number is not None:
                found.append(number)
                break
    return max(found) if found else None


def _tail_mapping(row: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in (
        "ineffective_tail",
        "ineffective_interaction_tail",
        "loop_tail",
        "tail_metrics",
    ):
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _nested_mapping(
    row: Mapping[str, Any],
    tail: Mapping[str, Any],
    *keys: str,
) -> Mapping[str, Any]:
    for key in keys:
        value = row.get(key, tail.get(key))
        if isinstance(value, Mapping):
            return value
    return {}


def _metric(
    row: Mapping[str, Any],
    tail: Mapping[str, Any],
    *keys: str,
) -> float | None:
    for key in keys:
        value = row.get(key, tail.get(key))
        number = _number(value)
        if number is not None:
            return number
    return None


def _public_tail(row: Mapping[str, Any]) -> dict[str, Any]:
    tail = _tail_mapping(row)
    quantiles = _nested_mapping(
        row,
        tail,
        "ineffective_quantiles",
        "quantiles",
        "percentiles",
    )
    counts = _nested_mapping(
        row,
        tail,
        "ineffective_threshold_counts",
        "case_threshold_counts",
        "counts",
        "cases_at_least",
    )
    shares = _nested_mapping(
        row,
        tail,
        "worst_case_shares",
        "tail_shares",
        "shares",
        "worst_share",
    )
    streak_counts = _nested_mapping(
        row,
        tail,
        "visible_no_effect_streak_counts",
        "streak_threshold_counts",
    )

    def quantile(name: str) -> float | None:
        return _number(
            quantiles.get(
                name,
                row.get(
                    f"ineffective_{name}",
                    tail.get(f"ineffective_{name}"),
                ),
            )
        )

    def threshold(mapping: Mapping[str, Any], threshold_value: int) -> int | None:
        for key in (
            f"at_least_{threshold_value}",
            f">={threshold_value}",
            str(threshold_value),
        ):
            value = _integer(mapping.get(key))
            if value is not None:
                return value
        direct = _integer(
            row.get(
                f"cases_ineffective_at_least_{threshold_value}",
                tail.get(f"cases_ineffective_at_least_{threshold_value}"),
            )
        )
        return direct

    def share(count: int) -> float | None:
        for key in (f"worst_{count}", f"top_{count}", str(count)):
            value = _number(shares.get(key))
            if value is not None:
                return value
        name = {1: "one", 2: "two", 5: "five"}[count]
        return _metric(
            row,
            tail,
            f"worst_{name}_ineffective_share",
            f"top_{name}_ineffective_share",
        )

    return {
        "ineffective_quantiles": {
            name: quantile(name) for name in ("p50", "p90", "p95", "p99")
        },
        "case_threshold_counts": {
            f"at_least_{value}": threshold(counts, value)
            for value in (1, 3, 10, 32)
        },
        "worst_case_shares": {
            f"worst_{value}": share(value) for value in (1, 2, 5)
        },
        "visible_no_effect_streak_counts": {
            f"at_least_{value}": threshold(streak_counts, value)
            for value in (3, 10, 32)
        },
        "max_ineffective_interactions": _metric(
            row,
            tail,
            "max_ineffective_interactions",
            "worst_case_ineffective_interactions",
            "max_case_ineffective_interactions",
        ),
        "max_identical_visible_no_effect_streak": _metric(
            row,
            tail,
            "max_identical_visible_no_effect_streak",
            "max_visible_no_effect_streak",
            "longest_identical_visibly_ineffective_streak",
        ),
        "max_repeated_identical_interaction_run": _metric(
            row,
            tail,
            "max_repeated_identical_interaction_run",
            "longest_repeated_identical_interaction_run",
            "max_identical_interaction_run",
        ),
    }


def _public_exam(exam: Mapping[str, Any]) -> dict[str, Any]:
    rows = _lesson_rows(exam)
    lessons: list[dict[str, Any]] = []
    for lesson_id, label, _overall, _panel, _ineffective in LESSONS:
        row = next((item for item in rows if item.get("lesson_id") == lesson_id), None)
        if row is None:
            continue
        panels = row.get("panel_successes")
        lessons.append(
            {
                "lesson_id": lesson_id,
                "lesson_label": row.get("lesson_label", label),
                "successes": _integer(row.get("successes")),
                "episodes": _integer(row.get("episodes")),
                "panel_successes": (
                    [_integer(value) for value in panels]
                    if isinstance(panels, list)
                    else []
                ),
                "mean_ineffective_interactions": _number(
                    row.get("mean_ineffective_interactions")
                ),
                "passed": row.get("passed"),
                **_public_tail(row),
            }
        )
    boundary = _integer(
        exam.get(
            "boundary",
            exam.get("child_trained_actions", exam.get("trained_actions")),
        )
    )
    allocation = exam.get("allocation")
    allocation_valid = exam.get("allocation_valid")
    practice_profile = exam.get("practice_profile")
    if isinstance(allocation, Mapping):
        if allocation_valid is None:
            allocation_valid = allocation.get("valid", allocation.get("verified"))
        if practice_profile is None:
            practice_profile = allocation.get("profile")
    max_case = _tail_value(
        rows,
        keys=(
            "max_ineffective_interactions",
            "worst_case_ineffective_interactions",
            "max_case_ineffective_interactions",
        ),
    )
    max_streak = _tail_value(
        rows,
        keys=(
            "max_identical_visible_no_effect_streak",
            "max_visible_no_effect_streak",
            "longest_identical_visibly_ineffective_streak",
        ),
    )
    max_interaction_run = _tail_value(
        rows,
        keys=(
            "max_repeated_identical_interaction_run",
            "longest_repeated_identical_interaction_run",
            "max_identical_interaction_run",
        ),
    )
    top_two = _tail_value(
        rows,
        keys=(
            "top_two_ineffective_share",
            "top_two_tail_share",
            "worst_two_ineffective_share",
        ),
    )
    if top_two is None:
        shares = [
            _number(lesson["worst_case_shares"].get("worst_2"))
            for lesson in lessons
            if isinstance(lesson.get("worst_case_shares"), Mapping)
        ]
        available_shares = [value for value in shares if value is not None]
        top_two = max(available_shares) if available_shares else None
    def sum_lesson_counts(field: str, threshold: int) -> int | None:
        key = f"at_least_{threshold}"
        values = [
            _integer(lesson[field].get(key))
            for lesson in lessons
            if isinstance(lesson.get(field), Mapping)
        ]
        if len(values) != len(lessons) or any(value is None for value in values):
            return None
        return sum(int(value) for value in values if value is not None)

    counts = {
        f"at_least_{threshold}": sum_lesson_counts(
            "case_threshold_counts",
            threshold,
        )
        for threshold in (1, 3, 10, 32)
    }
    streak_counts = {
        f"at_least_{threshold}": sum_lesson_counts(
            "visible_no_effect_streak_counts",
            threshold,
        )
        for threshold in (3, 10, 32)
    }
    public = {
        "boundary": boundary,
        "allocation_valid": allocation_valid is True,
        "practice_profile": practice_profile,
        "lessons": lessons,
        "max_case_ineffective": max_case,
        "max_no_effect_streak": max_streak,
        "max_interaction_run": max_interaction_run,
        "top_two_tail_share": top_two,
        "case_threshold_counts": counts,
        "visible_no_effect_streak_counts": streak_counts,
    }
    passed, reasons = _exam_pass(public)
    public["passed"] = passed
    public["reasons"] = reasons
    return public


def _exam_pass(exam: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if exam.get("allocation_valid") is not True:
        reasons.append("allocation is not authenticated")
    if exam.get("practice_profile") != "normal":
        reasons.append("window is not normal practice")
    lessons = exam.get("lessons")
    lesson_items = lessons if isinstance(lessons, list) else []
    lesson_map = {
        item.get("lesson_id"): item
        for item in lesson_items
        if isinstance(item, Mapping)
    }
    for lesson_id, label, overall, panel, ineffective_gate in LESSONS:
        row = lesson_map.get(lesson_id)
        if row is None:
            reasons.append(f"{label} is missing")
            continue
        successes = _integer(row.get("successes"))
        panels = row.get("panel_successes")
        if successes is None or successes < overall:
            reasons.append(f"{label} overall gate failed")
        if (
            not isinstance(panels, list)
            or len(panels) != 2
            or any(_integer(value) is None or int(value) < panel for value in panels)
        ):
            reasons.append(f"{label} panel gate failed")
        if row.get("passed") is False:
            reasons.append(f"{label} existing gate failed")
        if ineffective_gate:
            ineffective = _number(row.get("mean_ineffective_interactions"))
            if ineffective is None or ineffective > 3.0:
                reasons.append(f"{label} ineffective mean failed")
    max_case = _number(exam.get("max_case_ineffective"))
    if max_case is None or max_case >= 10:
        reasons.append("case ineffective tail failed")
    max_interaction_run = _number(exam.get("max_interaction_run"))
    if max_interaction_run is None or max_interaction_run >= 10:
        reasons.append("repeated identical interaction run failed")
    return not reasons, reasons


def _timestamp_age(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return max(0.0, (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds())


def _authenticate_live_qualification(
    *,
    source_commit: str,
    tag_object: str,
) -> dict[str, Any]:
    """Re-run the canonical qualification and annotated-tag checks."""

    try:
        from scripts.u2s_ablation_manifest import (
            _verified_qualification_binding,
        )
    except ImportError as error:
        raise U2sDashboardError(
            "U2-S qualification authenticator is unavailable"
        ) from error
    try:
        return _verified_qualification_binding(
            source_commit=source_commit,
            tag_object=tag_object,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise U2sDashboardError(
            f"live U2-S qualification authentication failed: {error}"
        ) from error


def _deep_verify_arm_terminal(
    directory: Path,
    *,
    arm_id: str,
    source_commit: str,
    contract_digest: str,
) -> dict[str, Any]:
    """Run the trainer's complete checkpoint, ledger, and case audit."""

    try:
        from dungeon_apprentice.v02_u2s import verify_terminal_report
    except ImportError as error:
        raise U2sDashboardError(
            "U2-S terminal authenticator is unavailable"
        ) from error
    try:
        evidence = verify_terminal_report(
            directory,
            expected_arm=arm_id,
            expected_source_commit=source_commit,
            expected_cohort_contract_sha256=contract_digest,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise U2sDashboardError(
            f"{arm_id} failed deep terminal authentication: {error}"
        ) from error
    if not isinstance(evidence, Mapping):
        raise U2sDashboardError(
            f"{arm_id} terminal authenticator returned no evidence"
        )
    return json.loads(json.dumps(evidence))


def _authenticate_live_process_closeout(
    root: Path,
    sealed: Mapping[str, Any],
) -> dict[str, Any]:
    dashboard = sealed.get("dashboard")
    if not isinstance(dashboard, Mapping):
        raise U2sDashboardError("U2-S sealed dashboard identity is missing")
    try:
        from scripts.u2s_ablation_manifest import _process_closeout_evidence
    except ImportError as error:
        raise U2sDashboardError(
            "U2-S process-closeout authenticator is unavailable"
        ) from error
    try:
        current = _process_closeout_evidence(
            root=root,
            dashboard_pid=int(dashboard.get("pid", -1)),
            dashboard_port=int(dashboard.get("port", -1)),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise U2sDashboardError(
            f"live U2-S process-closeout authentication failed: {error}"
        ) from error
    current_dashboard = current.get("dashboard")
    if (
        current.get("trainer_process_count") != 0
        or current.get("supervisor_process_count") != 0
        or current.get("caffeinate_process_count") != 0
        or current.get("orphan_free") is not True
        or not isinstance(current_dashboard, Mapping)
        or current_dashboard.get("pid") != dashboard.get("pid")
        or current_dashboard.get("command_sha256")
        != dashboard.get("command_sha256")
        or current_dashboard.get("host") != dashboard.get("host")
        or current_dashboard.get("port") != dashboard.get("port")
        or current_dashboard.get("listener_pids")
        != dashboard.get("listener_pids")
        or current_dashboard.get("explicitly_excepted") is not True
    ):
        raise U2sDashboardError(
            "live U2-S process closeout differs from its terminal seal"
        )
    return current


def _arm_evidence_matches(
    recorded: Mapping[str, Any],
    verified: Mapping[str, Any],
) -> bool:
    direct_keys = (
        "arm",
        "verdict",
        "mechanism_selection_eligible",
        "child_trained_actions",
        "lifetime_trained_actions",
        "optimizer_updates",
        "exam_count",
        "case_count",
        "case_evidence_sha256",
        "case_evidence_files",
        "initial_rng_identity",
        "qualification_sha256",
        "terminal_checkpoint_sha256",
        "report_sha256",
        "report_integrity_sha256",
    )
    if any(recorded.get(key) != verified.get(key) for key in direct_keys):
        return False
    aliases = (
        (
            "terminal_checkpoint_sidecar_sha256",
            "terminal_sidecar_sha256",
        ),
        (
            "terminal_checkpoint_integrity_sha256",
            "terminal_integrity_sha256",
        ),
    )
    return all(recorded.get(left) == verified.get(right) for left, right in aliases)


def _verified_cohort_closeout(
    root: Path,
    *,
    state: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_digest: str,
) -> dict[str, Any]:
    terminal = state.get("terminal_report")
    if not isinstance(terminal, Mapping):
        raise U2sDashboardError("completed U2-S cohort has no terminal report binding")
    if (
        terminal.get("path") != "report.json"
        or terminal.get("integrity") != "report.integrity.json"
        or terminal.get("checkpoint_promotable") is not False
    ):
        raise U2sDashboardError("U2-S terminal report pointer changed")
    report_path = root / "report.json"
    integrity_path = root / "report.integrity.json"
    report = _read_json(report_path, maximum=EVALUATIONS_MAX_BYTES)
    integrity = _read_json(integrity_path)
    report_digest = _sha256(report_path)
    integrity_digest = _sha256(integrity_path)
    arm_evidence = report.get("arm_evidence")
    selection = report.get("selection")
    checkpoint_rule = report.get("checkpoint_rule")
    matched_rng = report.get("matched_initial_rng_identity")
    process_closeout = report.get("process_closeout")
    factorial_contrasts = report.get("factorial_contrasts")
    if (
        terminal.get("sha256") != report_digest
        or terminal.get("integrity_sha256") != integrity_digest
        or terminal.get("verdict") != report.get("verdict")
        or terminal.get("selected_configuration")
        != report.get("selected_configuration")
        or report.get("schema_version") != 1
        or report.get("protocol") != PROTOCOL
        or report.get("cohort_id") != COHORT_ID
        or report.get("development_only") is not True
        or report.get("source") != contract.get("source")
        or report.get("preregistration") != contract.get("preregistration")
        or report.get("parent") != contract.get("parent")
        or report.get("qualification") != contract.get("qualification")
        or report.get("cohort_contract") != "cohort-contract.json"
        or report.get("cohort_contract_sha256") != contract_digest
        or not isinstance(arm_evidence, Mapping)
        or set(arm_evidence) != set(ARM_ORDER)
        or not isinstance(selection, Mapping)
        or selection.get("priority") != list(ARM_ORDER)
        or selection.get("ablation_checkpoint_reuse_authorized") is not False
        or selection.get("selected_arm") != report.get("selected_configuration")
        or report.get("verdict")
        != (
            "mechanism_selected"
            if report.get("selected_configuration") is not None
            else "ablation_failed"
        )
        or report.get("selected_configuration") not in {*ARM_ORDER, None}
        or not isinstance(checkpoint_rule, Mapping)
        or checkpoint_rule.get("ablation_checkpoint_reuse_authorized") is not False
        or checkpoint_rule.get("successor_checkpoint") is not None
        or checkpoint_rule.get(
            "successor_must_restart_from_confirmed_u1_parent"
        )
        is not True
        or integrity.get("schema_version") != 1
        or integrity.get("protocol") != PROTOCOL
        or integrity.get("cohort_id") != COHORT_ID
        or integrity.get("report") != "report.json"
        or integrity.get("report_sha256") != report_digest
        or integrity.get("cohort_contract_sha256") != contract_digest
        or not isinstance(matched_rng, Mapping)
        or integrity.get("initial_rng_identity_sha256")
        != _canonical_sha256(matched_rng)
        or terminal.get("initial_rng_identity_sha256")
        != integrity.get("initial_rng_identity_sha256")
        or not isinstance(process_closeout, Mapping)
        or integrity.get("process_closeout_sha256")
        != _canonical_sha256(process_closeout)
        or terminal.get("process_closeout_sha256")
        != integrity.get("process_closeout_sha256")
        or not isinstance(factorial_contrasts, Mapping)
        or integrity.get("factorial_contrasts_sha256")
        != _canonical_sha256(factorial_contrasts)
        or terminal.get("factorial_contrasts_sha256")
        != integrity.get("factorial_contrasts_sha256")
    ):
        raise U2sDashboardError("U2-S terminal closeout evidence differs")
    report_hashes = integrity.get("arm_report_sha256")
    case_hashes = integrity.get("arm_case_evidence_sha256")
    if (
        not isinstance(report_hashes, Mapping)
        or not isinstance(case_hashes, Mapping)
        or set(report_hashes) != set(ARM_ORDER)
        or set(case_hashes) != set(ARM_ORDER)
    ):
        raise U2sDashboardError("U2-S terminal arm integrity inventory is missing")
    for arm_id in ARM_ORDER:
        evidence = arm_evidence.get(arm_id)
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("arm") != arm_id
            or evidence.get("report_sha256") != report_hashes.get(arm_id)
            or evidence.get("case_evidence_sha256") != case_hashes.get(arm_id)
            or evidence.get("child_trained_actions") != ACTION_CAP
            or evidence.get("exam_count") != 32
            or evidence.get("case_count") != 10_240
        ):
            raise U2sDashboardError(
                f"U2-S terminal evidence differs for {arm_id}"
            )
    common_rng = matched_rng.get("identity")
    rng_aggregates = matched_rng.get("arm_aggregate_sha256")
    if (
        matched_rng.get("identical_across_all_arms") is not True
        or not isinstance(common_rng, Mapping)
        or not isinstance(rng_aggregates, Mapping)
        or set(rng_aggregates) != set(ARM_ORDER)
        or any(
            arm_evidence[arm_id].get("initial_rng_identity") != common_rng
            or rng_aggregates.get(arm_id) != common_rng.get("aggregate_sha256")
            for arm_id in ARM_ORDER
        )
        or process_closeout.get("trainer_process_count") != 0
        or process_closeout.get("supervisor_process_count") != 0
        or process_closeout.get("caffeinate_process_count") != 0
        or process_closeout.get("orphan_free") is not True
        or not isinstance(process_closeout.get("checked_at"), str)
        or not isinstance(process_closeout.get("dashboard"), Mapping)
        or process_closeout["dashboard"].get("explicitly_excepted") is not True
        or factorial_contrasts.get("schema_version") != 1
        or factorial_contrasts.get("descriptive_only") is not True
        or factorial_contrasts.get("population_inference_authorized") is not False
    ):
        raise U2sDashboardError(
            "U2-S matched RNG, process, or factorial seal is invalid"
        )
    _authenticate_live_process_closeout(root, process_closeout)
    source_commit = str(contract.get("source", {}).get("commit", ""))
    tag_object = str(contract.get("preregistration", {}).get("tag_object", ""))
    qualification = _authenticate_live_qualification(
        source_commit=source_commit,
        tag_object=tag_object,
    )
    if qualification != contract.get("qualification"):
        raise U2sDashboardError(
            "live U2-S qualification differs from the cohort contract"
        )
    arm_states = state.get("arms")
    if not isinstance(arm_states, list) or len(arm_states) != len(ARM_ORDER):
        raise U2sDashboardError("U2-S terminal arm state inventory is missing")
    for arm_id, arm_state in zip(ARM_ORDER, arm_states, strict=True):
        if not isinstance(arm_state, Mapping):
            raise U2sDashboardError(f"U2-S {arm_id} state is invalid")
        recorded = arm_evidence[arm_id]
        verified = _deep_verify_arm_terminal(
            root / arm_id,
            arm_id=arm_id,
            source_commit=source_commit,
            contract_digest=contract_digest,
        )
        state_terminal = arm_state.get("terminal")
        if (
            arm_state.get("id") != arm_id
            or arm_state.get("state") != "completed"
            or not isinstance(state_terminal, Mapping)
            or state_terminal.get("verified") is not True
            or not _arm_evidence_matches(recorded, verified)
            or not _arm_evidence_matches(state_terminal, verified)
        ):
            raise U2sDashboardError(
                f"U2-S completed claim is unauthenticated for {arm_id}"
            )
    return report


def load_u2s_snapshot(run_root: Path) -> dict[str, Any]:
    """Load a bounded public comparison and reject provenance drift."""

    root = _absolute_regular_directory(run_root)
    contract_path = root / "cohort-contract.json"
    state_path = root / "cohort.json"
    contract = _read_json(contract_path)
    state = _read_json(state_path)
    contract_digest = _sha256(contract_path)
    source = contract.get("source")
    parent = contract.get("parent")
    qualification = contract.get("qualification")
    preregistration = contract.get("preregistration")
    matched = contract.get("matched_design")
    contract_arms = contract.get("arms")
    if (
        contract.get("schema_version") != 1
        or contract.get("protocol") != PROTOCOL
        or contract.get("cohort_id") != COHORT_ID
        or not isinstance(source, Mapping)
        or source.get("dirty") is not False
        or not isinstance(parent, Mapping)
        or not isinstance(qualification, Mapping)
        or qualification.get("verdict") != "qualified"
        or not isinstance(preregistration, Mapping)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(preregistration.get("tag_payload_sha256", "")),
        )
        is None
        or not isinstance(matched, Mapping)
        or matched.get("arm_order") != list(ARM_ORDER)
        or matched.get("selection_priority") != list(ARM_ORDER)
        or matched.get("checkpoint_promotable") is not False
        or not isinstance(contract_arms, list)
        or [arm.get("id") for arm in contract_arms if isinstance(arm, Mapping)]
        != list(ARM_ORDER)
    ):
        raise U2sDashboardError("immutable U2-S cohort contract is invalid")
    if (
        state.get("schema_version") != 1
        or state.get("protocol") != PROTOCOL
        or state.get("cohort_id") != COHORT_ID
        or state.get("contract_sha256") != contract_digest
        or state.get("source_commit") != source.get("commit")
        or state.get("tag") != preregistration.get("tag")
        or state.get("tag_object") != preregistration.get("tag_object")
        or state.get("parent_checkpoint_sha256") != parent.get("checkpoint_sha256")
        or not isinstance(state.get("arms"), list)
        or [arm.get("id") for arm in state["arms"] if isinstance(arm, Mapping)]
        != list(ARM_ORDER)
    ):
        raise U2sDashboardError("mutable U2-S cohort provenance differs from its contract")
    phase = state.get("phase")
    if phase not in {
        "ready",
        "training",
        "awaiting_closeout",
        "completed",
        "operationally_incomplete",
        "integrity_failed",
    }:
        raise U2sDashboardError("mutable U2-S cohort phase is invalid")
    closeout = (
        _verified_cohort_closeout(
            root,
            state=state,
            contract=contract,
            contract_digest=contract_digest,
        )
        if phase == "completed"
        else None
    )

    public_arms: list[dict[str, Any]] = []
    ages: list[float] = []
    for definition, arm_state in zip(contract_arms, state["arms"], strict=True):
        arm_id = str(definition["id"])
        directory = root / str(definition["directory"])
        status: dict[str, Any] = {}
        if directory.exists() or directory.is_symlink():
            if directory.is_symlink() or not directory.is_dir():
                raise U2sDashboardError(f"unsafe U2-S arm directory: {arm_id}")
            status = _read_json(directory / "status.json")
            if (
                status.get("protocol") != PROTOCOL
                or status.get("arm") != arm_id
                or _status_source(status) != source.get("commit")
                or status.get("parent_checkpoint_sha256")
                != parent.get("checkpoint_sha256")
                or status.get("qualification_sha256")
                != qualification.get("report_sha256")
                or status.get("cohort_contract_sha256") != contract_digest
                or _integer(status.get("action_cap")) != ACTION_CAP
            ):
                raise U2sDashboardError(f"{arm_id} status differs from the cohort contract")
            age = _timestamp_age(status.get("updated_at"))
            if age is not None:
                ages.append(age)
        if phase == "completed":
            arm_report_path = directory / "report.json"
            arm_report = _read_json(
                arm_report_path,
                maximum=EVALUATIONS_MAX_BYTES,
            )
            expected_report_sha256 = closeout["arm_evidence"][arm_id][
                "report_sha256"
            ]
            raw_exam_value = arm_report.get("exam_records")
            if (
                _sha256(arm_report_path) != expected_report_sha256
                or not isinstance(raw_exam_value, list)
                or not all(
                    isinstance(exam, Mapping) for exam in raw_exam_value
                )
            ):
                raise U2sDashboardError(
                    f"{arm_id} authenticated exam report changed"
                )
            raw_exams = [dict(exam) for exam in raw_exam_value]
        else:
            raw_exams = _evaluation_records(directory, status) if status else []
        exams = [_public_exam(exam) for exam in raw_exams]
        terminal = exams[-3:]
        eligible: bool | None = None
        if arm_state.get("state") == "completed":
            eligible = (
                len(terminal) == 3
                and [exam.get("boundary") for exam in terminal]
                == [983_040, 1_015_808, 1_048_576]
                and all(exam["passed"] for exam in terminal)
            )
        public_arms.append(
            {
                "id": arm_id,
                "label": definition.get("label", arm_id),
                "ppo_profile": definition.get("ppo_profile"),
                "no_effect_penalty": definition.get("no_effect_penalty"),
                "state": arm_state.get("state"),
                "phase": status.get("phase"),
                "updated_at": status.get("updated_at"),
                "trained_actions": (
                    _integer(status.get("child_trained_actions")) or 0
                ),
                "collected_actions": _integer(status.get("collected_actions")) or 0,
                "remaining_action_budget": _integer(
                    status.get("remaining_action_budget")
                ),
                "optimizer_updates": _integer(status.get("optimizer_updates")),
                "exams_completed": len(exams),
                "exams": exams,
                "terminal_checks": terminal,
                "eligible": eligible,
            }
        )
    selected_id = (
        closeout.get("selected_configuration")
        if isinstance(closeout, Mapping)
        else None
    )
    selected = next(
        (arm for arm in public_arms if arm["id"] == selected_id),
        None,
    )
    if isinstance(closeout, Mapping):
        grades = closeout.get("selection", {}).get("grades")
        if not isinstance(grades, Mapping):
            raise U2sDashboardError("U2-S terminal selection grades are missing")
        for arm in public_arms:
            grade = grades.get(arm["id"])
            if (
                not isinstance(grade, Mapping)
                or grade.get("eligible") is not arm["eligible"]
            ):
                raise U2sDashboardError(
                    f"dashboard recomputation differs for {arm['id']}"
                )
        expected_selected = next(
            (arm["id"] for arm in public_arms if arm["eligible"] is True),
            None,
        )
        if selected_id != expected_selected:
            raise U2sDashboardError(
                "U2-S terminal selection violates its fixed priority"
            )
    active = state.get("active_arm")
    heartbeat_age = max(ages) if ages else None
    active_public = next((arm for arm in public_arms if arm["id"] == active), None)
    active_age = (
        _timestamp_age(active_public.get("updated_at"))
        if isinstance(active_public, Mapping)
        else None
    )
    return {
        "protocol": PROTOCOL,
        "cohort_id": COHORT_ID,
        "loaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "phase": phase,
        "active_arm": active,
        "source_commit": source.get("commit"),
        "tag": preregistration.get("tag"),
        "tag_object": preregistration.get("tag_object"),
        "tag_payload_sha256": preregistration.get("tag_payload_sha256"),
        "parent_checkpoint_sha256": parent.get("checkpoint_sha256"),
        "qualification_report_sha256": qualification.get("report_sha256"),
        "contract_sha256": contract_digest,
        "heartbeat_age_seconds": active_age if active is not None else heartbeat_age,
        "heartbeat_stale": (
            active is not None and (active_age is None or active_age > 180)
        ),
        "total_trained_actions": sum(arm["trained_actions"] for arm in public_arms),
        "interruption_disclosure": (
            "The matched cohort has no resume path. Any interruption or crash makes "
            "the complete attempt operationally_incomplete; a replacement requires "
            "a new source commit, annotated tag, protocol-attempt identity, root, "
            "and all four fresh arms."
        ),
        "arms": public_arms,
        "selected_mechanism": (
            None
            if selected is None
            else {
                "id": selected["id"],
                "label": selected["label"],
                "ppo_profile": selected["ppo_profile"],
                "no_effect_penalty": selected["no_effect_penalty"],
            }
        ),
        "checkpoint_promotable": False,
        "terminal_verdict": (
            closeout.get("verdict") if isinstance(closeout, Mapping) else None
        ),
        "terminal_report_sha256": (
            state.get("terminal_report", {}).get("sha256")
            if isinstance(state.get("terminal_report"), Mapping)
            else None
        ),
        "selection_note": (
            "The fixed priority selects only a learner configuration; "
            "no ablation checkpoint is promotable."
        ),
    }


class U2sDashboardHandler(BaseHTTPRequestHandler):
    server_version = "DungeonApprenticeU2sDashboard/1"

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
            "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'",
        )
        self.end_headers()
        if send_body:
            self.wfile.write(content)

    def _error(self, status: HTTPStatus, message: str, *, send_body: bool) -> None:
        self._send(
            json.dumps({"error": message}, sort_keys=True).encode(),
            content_type="application/json; charset=utf-8",
            status=status,
            send_body=send_body,
        )

    def _handle(self, *, send_body: bool) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send(
                U2S_DASHBOARD_HTML.encode(),
                content_type="text/html; charset=utf-8",
                send_body=send_body,
            )
            return
        if path == "/api/u2s.json":
            try:
                with self.server.snapshot_lock:
                    content = json.dumps(
                        load_u2s_snapshot(self.server.run_root),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
            except (OSError, U2sDashboardError, ValueError) as error:
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    str(error),
                    send_body=send_body,
                )
                return
            self._send(
                content,
                content_type="application/json; charset=utf-8",
                send_body=send_body,
            )
            return
        self._error(HTTPStatus.NOT_FOUND, "not found", send_body=send_body)

    def do_GET(self) -> None:
        self._handle(send_body=True)

    def do_HEAD(self) -> None:
        self._handle(send_body=False)

    def do_POST(self) -> None:
        self._error(
            HTTPStatus.METHOD_NOT_ALLOWED,
            "read-only dashboard",
            send_body=True,
        )

    do_DELETE = do_POST
    do_PATCH = do_POST
    do_PUT = do_POST


def start_u2s_dashboard(
    run_root: Path,
    *,
    port: int = DEFAULT_PORT,
) -> U2sDashboardServer:
    """Start the fixed local dashboard without a port fallback."""

    if port <= 0 or port > 65_535:
        raise ValueError("dashboard port must be between 1 and 65535")
    root = _absolute_regular_directory(run_root)
    load_u2s_snapshot(root)
    server = U2sDashboardServer((DEFAULT_HOST, port), U2sDashboardHandler)
    server.run_root = root
    server.snapshot_lock = threading.Lock()
    thread = threading.Thread(
        target=server.serve_forever,
        name="u2s-dashboard",
        daemon=True,
    )
    thread.start()
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = start_u2s_dashboard(args.run_root, port=args.port)
    print(
        f"U2-S dashboard: http://{DEFAULT_HOST}:{server.server_port}/",
        flush=True,
    )
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
