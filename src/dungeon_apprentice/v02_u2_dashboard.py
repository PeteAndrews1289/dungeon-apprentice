# ruff: noqa: E501
"""Read-only, cohort-wide live dashboard for Dungeon Apprentice U2."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8785
MANIFEST_NAME = "cohort.json"
MANIFEST_MAX_BYTES = 256 * 1024
STATUS_MAX_BYTES = 4 * 1024 * 1024
FRAME_MAX_BYTES = 16 * 1024 * 1024
LINEAGE_COUNT = 3
LINEAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

LESSONS = (
    ("navigate/full", "Navigate"),
    ("unlock/u0-visible", "Visible Unlock"),
    ("unlock/u1-local", "Local Unlock"),
    ("unlock/u2-separated", "Separated Unlock"),
)
LESSON_IDS = frozenset(lesson_id for lesson_id, _label in LESSONS)

_PUBLIC_STATUS_FIELDS = (
    "protocol",
    "phase",
    "setup_complete",
    "active_lineage",
    "parent",
    "qualification",
    "source",
    "parent_checkpoint_sha256",
    "qualification_sha256",
    "source_commit",
    "started_at",
    "updated_at",
    "elapsed_seconds",
    "actions_per_second",
    "fps",
    "collected_actions",
    "trained_actions",
    "inherited_trained_actions",
    "child_collected_actions",
    "child_trained_actions",
    "child_collected_timesteps",
    "child_trained_timesteps",
    "collected_timesteps",
    "trained_timesteps",
    "lifetime_trained_actions",
    "lifetime_trained_timesteps",
    "remaining_child_actions",
    "remaining_action_budget",
    "action_cap",
    "optimizer_updates",
    "episodes",
    "current_lesson_id",
    "current_lesson_label",
    "frame_revision",
    "latest_evaluations",
    "evaluation_history",
    "curriculum",
    "controller",
    "practice_allocation",
    "last_completed_practice_allocation",
    "last_completed_practice_allocation_valid",
    "recent_training",
    "recent_behavior",
    "latest_optimizer",
    "next_evaluation",
    "lesson_frames",
    "latest_exam_checkpoint",
    "latest_evaluated_checkpoint",
    "latest_safe_checkpoint",
    "latest_safe_checkpoint_sha256",
    "latest_safe_trained_actions",
    "segment",
    "crash_report",
    "storage",
    "storage_usage",
)

COHORT_DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dungeon Apprentice — U2 Cohort</title>
<style>
:root{color-scheme:dark;--bg:#08100d;--panel:#101d17;--panel2:#14251d;--line:#294035;
--ink:#eef9f1;--muted:#9ab1a2;--green:#69e18c;--blue:#6fcbea;--amber:#f0ca70;
--red:#ef7c7c;--purple:#bb9cff}*{box-sizing:border-box}body{margin:0;color:var(--ink);
background:radial-gradient(circle at 15% 0,#173a29 0,var(--bg) 37%);font:15px/1.5
ui-rounded,system-ui,-apple-system,sans-serif}main{max-width:1220px;margin:auto;padding:28px 20px 60px}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin-bottom:22px}
.eyebrow{color:var(--green);font-size:12px;font-weight:850;letter-spacing:.16em;
text-transform:uppercase;margin-bottom:8px}h1{font-size:clamp(30px,6vw,54px);letter-spacing:-.05em;
line-height:1;margin:0}h2{margin:0 0 14px;font-size:19px}h3{margin:0;font-size:17px}
.fresh{display:flex;align-items:center;gap:8px;color:var(--muted);text-align:right}.dot{width:9px;
height:9px;border-radius:50%;background:var(--amber);box-shadow:0 0 14px var(--amber)}
.metrics,.lineages,.detailGrid,.frames{display:grid;gap:13px}.metrics{grid-template-columns:repeat(5,1fr);
margin-bottom:13px}.card{background:color-mix(in srgb,var(--panel) 94%,transparent);
border:1px solid var(--line);border-radius:17px;padding:17px;box-shadow:0 18px 48px #0004}
.metric span,.kicker,.tiny{color:var(--muted);font-size:11px;text-transform:uppercase;
letter-spacing:.08em}.metric b{font-size:24px;display:block;letter-spacing:-.03em;margin-top:3px}
.lineages{grid-template-columns:repeat(3,1fr);margin-bottom:13px}.lineage{position:relative;min-height:150px}
.lineage.active{border-color:var(--green);background:linear-gradient(145deg,#183326,var(--panel))}
.lineageTop{display:flex;justify-content:space-between;gap:10px;align-items:start}.badge{border:1px solid
var(--line);border-radius:999px;padding:3px 8px;color:var(--muted);font-size:11px}.badge.active{
border-color:var(--green);color:var(--green)}.lineageStats{display:grid;grid-template-columns:1fr 1fr;
gap:10px;margin-top:18px}.lineageStats b{display:block;font-size:19px}.lineageError{color:var(--red);
font-size:12px;margin-top:12px}.detailGrid{grid-template-columns:1.2fr .8fr}.wide{grid-column:1/-1}
.actionRow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}.actionCell b{
display:block;font-size:22px}.bar{height:12px;background:#07100b;border-radius:20px;overflow:hidden}
.fill{height:100%;width:0;background:linear-gradient(90deg,var(--blue),var(--green));
border-radius:20px;transition:width .35s}.subline{display:flex;justify-content:space-between;
gap:15px;color:var(--muted);font-size:12px;margin-top:7px}.lesson{display:grid;
grid-template-columns:minmax(125px,1fr) minmax(120px,1.8fr) 150px;gap:12px;align-items:center;
padding:12px 0;border-bottom:1px solid var(--line)}.lesson:last-child{border-bottom:0}.score{
font-variant-numeric:tabular-nums;text-align:right}.panelText{color:var(--muted);font-size:12px}
.pills{display:flex;flex-wrap:wrap;gap:7px}.pill{padding:6px 10px;border:1px solid var(--line);
border-radius:999px;color:var(--muted);font-size:12px}.pill.recovery{border-color:var(--amber);
color:var(--amber)}.pill.mastered{border-color:var(--green);color:var(--green)}
.frames{grid-template-columns:repeat(5,1fr)}figure{margin:0}figure img{width:100%;aspect-ratio:1;
object-fit:contain;image-rendering:pixelated;border:1px solid var(--line);border-radius:10px;
background:#040806}figcaption{font-size:11px;color:var(--muted);margin-top:5px;overflow-wrap:anywhere}
.empty{color:var(--muted);padding:15px 0}.note{color:var(--muted);font-size:12px;margin-top:12px}
.timestamps{display:grid;grid-template-columns:1fr 1fr;gap:12px}.timestamps b{display:block;
font-size:13px;overflow-wrap:anywhere}.fatal{border-color:var(--red);color:#ffd6d6}
.evidence{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.evidence div{
padding:9px;background:#09130e;border-radius:9px;overflow-wrap:anywhere}.evidence b{display:block;
font-size:12px;margin-top:2px}.stale{color:var(--red)}.healthy{color:var(--green)}
@media(max-width:850px){.metrics{grid-template-columns:repeat(2,1fr)}.metrics .metric:last-child{
grid-column:1/-1}.detailGrid{grid-template-columns:1fr}.lineages{grid-template-columns:1fr}
.frames{grid-template-columns:repeat(2,1fr)}}@media(max-width:560px){main{padding:22px 13px 44px}
header{align-items:flex-start;flex-direction:column}.fresh{text-align:left}.actionRow{grid-template-columns:1fr 1fr}
.lesson{grid-template-columns:1fr}.score{text-align:left}.metrics{grid-template-columns:1fr 1fr}
.card{padding:15px}.timestamps{grid-template-columns:1fr}}
</style></head><body><main>
<header><div><div class="eyebrow">U2 · cumulative experiential learning</div>
<h1>Three apprentices, one experiment</h1></div>
<div class="fresh"><i class="dot"></i><span id="freshness">Loading the cohort…</span></div></header>
<section class="metrics">
<div class="card metric"><span>Cohort state</span><b id="cohortState">—</b></div>
<div class="card metric"><span>Active lineage</span><b id="activeName">—</b></div>
<div class="card metric"><span>New learned actions</span><b id="cohortActions">—</b></div>
<div class="card metric"><span>Cohort runtime</span><b id="cohortRuntime">—</b></div>
<div class="card metric"><span>Science + media</span><b id="cohortStorage">—</b></div>
</section>
<section id="lineages" class="lineages"></section>
<section id="detail" class="detailGrid">
<div class="card wide"><div class="lineageTop"><div><div class="kicker">Active lineage</div>
<h2 id="detailTitle">Waiting for a lineage</h2></div><span id="detailState" class="badge">Pending</span></div>
<div class="actionRow">
<div class="actionCell"><span class="tiny">Collected</span><b id="collected">—</b></div>
<div class="actionCell"><span class="tiny">Learned from</span><b id="trained">—</b></div>
<div class="actionCell"><span class="tiny">Remaining</span><b id="remaining">—</b></div>
<div class="actionCell"><span class="tiny">Current rate</span><b id="rate">—</b></div>
</div><div class="bar"><div id="actionFill" class="fill"></div></div>
<div class="subline"><span id="actionProgress">Waiting for training.</span>
<span id="storageProgress">Storage awaiting first report.</span></div></div>
<div class="card"><h2>Frozen four-lesson exams</h2><div id="exams"></div>
<div class="note">Each lesson shows its overall score and both independent 40-case panels.</div></div>
<div class="card"><h2>Practice, recovery, and mastery</h2><div id="practice" class="pills"></div>
<div id="curriculumNote" class="note">Waiting for curriculum state.</div>
<div class="timestamps"><div><span class="tiny">Started</span><b id="startedAt">—</b></div>
<div><span class="tiny">Last heartbeat</span><b id="updatedAt">—</b></div></div></div>
<div class="card"><h2>Evidence chain</h2><div id="provenance" class="evidence"></div></div>
<div class="card"><h2>Learning and behavior</h2><div id="telemetry" class="evidence"></div></div>
<div class="card wide"><h2>What the policy sees</h2><div id="frames" class="frames"></div>
<div class="note">These are read-only snapshots. Closing this page never pauses or stops a trainer.</div></div>
</section></main>
<script>
const LESSONS=[['navigate/full','Navigate'],['unlock/u0-visible','Visible Unlock'],
['unlock/u1-local','Local Unlock'],['unlock/u2-separated','Separated Unlock']];
const byId=id=>document.getElementById(id);
const esc=value=>String(value??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;',
'"':'&quot;',"'":'&#39;'}[ch]));
const num=value=>Number.isFinite(Number(value))?Number(value):0;
const fmt=value=>Number.isFinite(Number(value))?Number(value).toLocaleString():'—';
const pct=value=>Math.max(0,Math.min(100,Number(value)||0));
const duration=seconds=>{let n=Math.max(0,num(seconds)),h=Math.floor(n/3600),m=Math.floor(n%3600/60);
return h?`${h}h ${m}m`:m?`${m}m`:`${Math.floor(n)}s`};
const bytes=value=>{let n=Math.max(0,num(value)),units=['B','KiB','MiB','GiB'];let i=0;
while(n>=1024&&i<3){n/=1024;i++}return i?`${n.toFixed(2)} ${units[i]}`:`${n} B`};
const stateLabel=value=>({training:'Training',pending:'Waiting',queued:'Queued',mastered:'Mastered',
completed:'Completed',crashed:'Stopped',interrupted:'Interrupted',unavailable:'Status unavailable'}[value]||
String(value||'Pending').replaceAll('_',' '));
const findEvaluation=(status,id)=>(Array.isArray(status.latest_evaluations)?
status.latest_evaluations:[]).find(item=>item.lesson_id===id);
function panels(exam){let values=Array.isArray(exam?.panel_successes)?exam.panel_successes:null;
if(values)return [num(values[0]),num(values[1])];let p=exam?.panels||{};
return [num(p.a?.successes??exam?.panel_a_successes),num(p.b?.successes??exam?.panel_b_successes)]}
function storageOf(status){let value=status.storage||status.storage_usage||{};
if(value.lineage)value=value.lineage;return {used:num(value.used_bytes),cap:num(value.cap_bytes)}}
function updateFreshness(snapshot,active){let stamp=active?.status?.updated_at||snapshot.manifest.updated_at||
snapshot.loaded_at,parsed=Date.parse(stamp),fallback=(Date.now()-parsed)/1000,
age=Number.isFinite(Number(active?.status_age_seconds))?Number(active.status_age_seconds):fallback;
let stale=active?.status_stale===true||(active?.state==='training'&&(!Number.isFinite(age)||age>180));
document.querySelector('.dot').style.background=stale?'var(--red)':active?.state==='training'?
'var(--green)':'var(--amber)';byId('freshness').textContent=stale?'Heartbeat delayed':
`Updated ${Number.isFinite(age)?Math.max(0,Math.floor(age))+'s ago':'time unknown'}`}
function renderLineages(snapshot){let activeId=snapshot.manifest.active_lineage_id;
byId('lineages').innerHTML=snapshot.lineages.map(lineage=>{let s=lineage.status||{},trained=
num(s.child_trained_timesteps),rate=num(s.fps),active=lineage.id===activeId;
return `<article class="card lineage ${active?'active':''}"><div class="lineageTop"><div>
<div class="kicker">Lineage ${lineage.order}</div><h3>${esc(lineage.label)}</h3></div>
<span class="badge ${active?'active':''}">${active?'Active · ':''}${esc(stateLabel(lineage.state))}</span>
</div><div class="lineageStats"><div><span class="tiny">Learned actions</span><b>${fmt(trained)}</b></div>
<div><span class="tiny">Rate</span><b>${rate?Math.round(rate)+'/s':'—'}</b></div></div>
<div class="note ${lineage.status_stale?'stale':'healthy'}">Heartbeat ${
lineage.status_age_seconds===null||lineage.status_age_seconds===undefined?'not available':
fmt(lineage.status_age_seconds)+'s old'}${lineage.status_stale?' · stale':''}</div>
${lineage.status_error?`<div class="lineageError">${esc(lineage.status_error)}</div>`:''}</article>`}).join('')}
function renderExams(status){byId('exams').innerHTML=LESSONS.map(([id,label])=>{let exam=findEvaluation(status,id),
score=num(exam?.successes),episodes=num(exam?.episodes)||80,rate=exam?num(exam.success_rate)*100:0,
[a,b]=panels(exam);return `<div class="lesson"><strong>${label}</strong><div><div class="bar">
<div class="fill" style="width:${pct(rate)}%"></div></div><div class="panelText">Panel A ${exam?a+'/40':'—/40'}
 · Panel B ${exam?b+'/40':'—/40'}</div></div><div class="score">${exam?score+'/'+episodes+
' · '+Math.round(rate)+'%':'No exam yet'}</div></div>`}).join('')}
function renderPractice(status){let curriculum=status.curriculum||{},allocation=
status.last_completed_practice_allocation||status.practice_allocation||{},targets=
allocation.target_shares||{},actual=allocation.realized_shares||{},items=Object.keys(targets).map(id=>
`<span class="pill">${esc(id.split('/').pop())}: target ${Math.round(num(targets[id])*100)}% · actual
 ${Math.round(num(actual[id])*100)}%</span>`);if(curriculum.recovery)items.unshift(
`<span class="pill recovery">Recovery: ${esc(curriculum.recovery_cause||'prerequisite')}</span>`);
if(curriculum.mastered)items.unshift('<span class="pill mastered">Full U2 mastery reached</span>');
byId('practice').innerHTML=items.join('')||'<span class="empty">No completed practice window yet.</span>';
byId('curriculumNote').textContent=curriculum.mastered?`Mastery streak: ${fmt(curriculum.consecutive_passes)}`
:curriculum.recovery?'Automatic practice has shifted toward a weakened retained skill.':
'Normal transition-balanced practice is active.'}
function short(value){let text=String(value||'—');return text==='—'?text:text.slice(0,12)}
function renderEvidence(status){let evaluated=status.latest_evaluated_checkpoint||{},allocation=
status.last_completed_practice_allocation||status.practice_allocation||{},curriculum=status.curriculum||{},
behavior=status.recent_behavior||{},optimizer=status.latest_optimizer||{};
byId('provenance').innerHTML=[
['Source commit',short(status.source_commit)],['Parent checkpoint',short(status.parent_checkpoint_sha256)],
['Qualification',short(status.qualification_sha256)],['Evaluated checkpoint',
short(evaluated.checkpoint_sha256)],['Exam boundary',fmt(evaluated.child_trained_actions)],
['Safe checkpoint',short(status.latest_safe_checkpoint_sha256)]].map(([label,value])=>
`<div><span class="tiny">${esc(label)}</span><b>${esc(value)}</b></div>`).join('');
byId('telemetry').innerHTML=[
['Inherited actions',fmt(status.inherited_trained_actions)],['Child actions',fmt(status.child_trained_actions)],
['Lifetime actions',fmt(status.lifetime_trained_actions)],['Optimizer updates',fmt(status.optimizer_updates)],
['Allocation',status.last_completed_practice_allocation_valid===true?'Within tolerance':
status.last_completed_practice_allocation_valid===false?'Outside tolerance':allocation.profile||'Pending'],
['Recovery / mastery',curriculum.mastered?'Mastered':curriculum.recovery?
`Recovery ${fmt(curriculum.recovery_passes)}/2`:`Pass streak ${fmt(curriculum.consecutive_passes)}/2`],
['Recent success',behavior.success_rate===null||behavior.success_rate===undefined?'—':
`${Math.round(num(behavior.success_rate)*100)}% / ${fmt(behavior.window_episodes)} episodes`],
['Coverage / collisions',`${behavior.mean_coverage===null||behavior.mean_coverage===undefined?'—':
num(behavior.mean_coverage).toFixed(2)} / ${behavior.mean_collisions===null||
behavior.mean_collisions===undefined?'—':num(behavior.mean_collisions).toFixed(1)}`],
['Action concentration',behavior.mean_largest_action_share===null||
behavior.mean_largest_action_share===undefined?'—':
`${Math.round(num(behavior.mean_largest_action_share)*100)}%`],
['Policy loss',optimizer.policy_gradient_loss===null||
optimizer.policy_gradient_loss===undefined?'—':num(optimizer.policy_gradient_loss).toFixed(4)]
].map(([label,value])=>`<div><span class="tiny">${esc(label)}</span><b>${esc(value)}</b></div>`).join('')}
function renderFrames(lineage){let urls=lineage.frame_urls||{},labels={latest:'Current training view',
'navigate/full':'Navigate exam','unlock/u0-visible':'Visible Unlock exam',
'unlock/u1-local':'Local Unlock exam','unlock/u2-separated':'Separated Unlock exam'};
byId('frames').innerHTML=Object.entries(urls).map(([name,url])=>`<figure><img src="${esc(url)}"
alt="${esc(labels[name]||name)}"><figcaption>${esc(labels[name]||name)}</figcaption></figure>`).join('')||
'<div class="empty">No policy frames have been published yet.</div>'}
function render(snapshot){let activeId=snapshot.manifest.active_lineage_id,active=
snapshot.lineages.find(item=>item.id===activeId)||snapshot.lineages.find(item=>item.state==='training')||
snapshot.lineages[0],status=active?.status||{},cap=num(status.action_cap||
snapshot.manifest.action_cap_per_lineage)||1048576,trained=num(status.child_trained_timesteps),
collected=num(status.child_collected_timesteps),remaining=Number.isFinite(Number(status.remaining_action_budget))?
num(status.remaining_action_budget):Math.max(0,cap-trained),cohortActions=snapshot.lineages.reduce(
(sum,item)=>sum+num(item.status?.child_trained_timesteps),0),storage=storageOf(status);
byId('cohortState').textContent=stateLabel(snapshot.manifest.phase);
byId('activeName').textContent=active?active.label:'None';
byId('cohortActions').textContent=fmt(cohortActions);
byId('cohortRuntime').textContent=duration(snapshot.manifest.elapsed_seconds);
let combined=snapshot.manifest.storage||{};byId('cohortStorage').textContent=
combined.combined_used_bytes!==undefined?`${bytes(combined.combined_used_bytes)} / ${
bytes(combined.combined_cap_bytes||17179869184)}`:storage.used?bytes(storage.used):'—';
renderLineages(snapshot);byId('detailTitle').textContent=active?.label||'Waiting for a lineage';
byId('detailState').textContent=stateLabel(active?.state);byId('collected').textContent=fmt(collected);
byId('trained').textContent=fmt(trained);byId('remaining').textContent=fmt(remaining);
byId('rate').textContent=num(status.fps)?`${Math.round(num(status.fps))}/s`:'—';
byId('actionFill').style.width=`${pct(cap?100*trained/cap:0)}%`;
byId('actionProgress').textContent=`${fmt(trained)} of ${fmt(cap)} learned actions`;
byId('storageProgress').textContent=storage.cap?`${bytes(storage.used)} of ${bytes(storage.cap)} lineage storage`:
'Storage awaiting first report.';byId('startedAt').textContent=status.started_at||'—';
byId('updatedAt').textContent=status.updated_at||'—';renderExams(status);renderPractice(status);
renderEvidence(status);renderFrames(active||{});updateFreshness(snapshot,active)}
let failures=0;async function poll(){try{let response=await fetch(`/api/cohort.json?v=${Date.now()}`,
{cache:'no-store'});if(!response.ok)throw new Error(`status ${response.status}`);render(await response.json());
failures=0}catch(error){failures++;if(failures>1){document.querySelector('.dot').style.background='var(--red)';
byId('freshness').textContent='Dashboard cannot read the cohort right now'}}finally{setTimeout(poll,2000)}}poll();
</script></body></html>"""


class DashboardDataError(RuntimeError):
    """Raised when the live manifest or status contract is invalid."""


class UnsafeDashboardPathError(DashboardDataError):
    """Raised before a dashboard can follow a symlink or escape its cohort."""


class CohortDashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    cohort_root: Path


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_cohort_root(cohort_root: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(Path(cohort_root).expanduser())))
    try:
        mode = root.lstat().st_mode
    except FileNotFoundError as error:
        raise FileNotFoundError(f"cohort root does not exist: {root}") from error
    if stat.S_ISLNK(mode):
        raise UnsafeDashboardPathError(f"cohort root cannot be a symlink: {root}")
    if not stat.S_ISDIR(mode):
        raise NotADirectoryError(f"cohort root is not a directory: {root}")
    return root.resolve(strict=True)


def _safe_descendant(
    root: Path,
    relative_path: str | Path,
    *,
    expect_directory: bool,
    allow_missing: bool,
) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or any(part in {".", ".."} for part in relative.parts):
        raise UnsafeDashboardPathError(f"path must be a relative cohort descendant: {relative_path}")

    target = Path(os.path.abspath(os.fspath(root / relative)))
    if not _contains(root, target):
        raise UnsafeDashboardPathError(f"path escapes cohort root {root}: {relative_path}")

    current = root
    for index, component in enumerate(relative.parts):
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if allow_missing:
                return target
            raise FileNotFoundError(f"dashboard artifact does not exist: {target}") from None
        if stat.S_ISLNK(mode):
            raise UnsafeDashboardPathError(f"dashboard path contains a symlink: {current}")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(mode):
            raise NotADirectoryError(f"dashboard path component is not a directory: {current}")

    resolved = target.resolve(strict=False)
    if not _contains(root, resolved):
        raise UnsafeDashboardPathError(f"resolved path escapes cohort root {root}: {relative_path}")
    if target.exists():
        mode = target.lstat().st_mode
        expected = stat.S_ISDIR(mode) if expect_directory else stat.S_ISREG(mode)
        if not expected:
            expected_name = "directory" if expect_directory else "regular file"
            raise DashboardDataError(f"dashboard path is not a {expected_name}: {target}")
    elif not allow_missing:
        raise FileNotFoundError(f"dashboard artifact does not exist: {target}")
    return resolved


def _read_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise DashboardDataError(f"dashboard artifact is not a regular file: {path}")
        if file_stat.st_size > maximum_bytes:
            raise DashboardDataError(
                f"dashboard artifact exceeds {maximum_bytes} byte read limit: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(maximum_bytes + 1)
        if len(content) > maximum_bytes:
            raise DashboardDataError(
                f"dashboard artifact exceeds {maximum_bytes} byte read limit: {path}"
            )
        return content
    finally:
        os.close(descriptor)


def _reject_json_constant(value: str) -> None:
    raise DashboardDataError(f"non-finite JSON number is not permitted: {value}")


def _read_json_object(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    try:
        content = _read_bytes(path, maximum_bytes=maximum_bytes).decode("utf-8")
    except UnicodeDecodeError as error:
        raise DashboardDataError(f"dashboard JSON is not UTF-8: {path}") from error
    try:
        value = json.loads(content, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        raise DashboardDataError(f"dashboard JSON is invalid: {path}: {error.msg}") from error
    if not isinstance(value, dict):
        raise DashboardDataError(f"dashboard JSON must contain an object: {path}")
    return value


def _load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = _safe_descendant(
        root,
        MANIFEST_NAME,
        expect_directory=False,
        allow_missing=False,
    )
    manifest = _read_json_object(manifest_path, maximum_bytes=MANIFEST_MAX_BYTES)
    lineages = manifest.get("lineages")
    if not isinstance(lineages, list) or len(lineages) != LINEAGE_COUNT:
        raise DashboardDataError(f"cohort manifest must declare exactly {LINEAGE_COUNT} lineages")

    identifiers: set[str] = set()
    for order, lineage in enumerate(lineages, start=1):
        if not isinstance(lineage, dict):
            raise DashboardDataError(f"lineage {order} manifest entry must be an object")
        identifier = lineage.get("id")
        directory = lineage.get("directory")
        if not isinstance(identifier, str) or LINEAGE_ID_PATTERN.fullmatch(identifier) is None:
            raise DashboardDataError(f"lineage {order} has an invalid id")
        if identifier in identifiers:
            raise DashboardDataError(f"duplicate lineage id in cohort manifest: {identifier}")
        if not isinstance(directory, str) or not directory:
            raise DashboardDataError(f"lineage {identifier} has no directory")
        _safe_descendant(root, directory, expect_directory=True, allow_missing=True)
        identifiers.add(identifier)

    active = manifest.get("active_lineage_id", manifest.get("active_lineage"))
    if active is not None and active not in identifiers:
        raise DashboardDataError(f"active lineage is not declared by the cohort: {active}")
    return manifest


def _status_path(root: Path, lineage: dict[str, Any], *, allow_missing: bool) -> Path:
    relative = Path(str(lineage["directory"])) / "status.json"
    return _safe_descendant(
        root,
        relative,
        expect_directory=False,
        allow_missing=allow_missing,
    )


def _load_status(root: Path, lineage: dict[str, Any]) -> dict[str, Any]:
    path = _status_path(root, lineage, allow_missing=False)
    return _read_json_object(path, maximum_bytes=STATUS_MAX_BYTES)


def _public_status(status: dict[str, Any]) -> dict[str, Any]:
    return {field: status[field] for field in _PUBLIC_STATUS_FIELDS if field in status}


def _status_freshness(
    status: Mapping[str, Any],
    *,
    now: datetime,
) -> tuple[int | None, bool]:
    try:
        updated = datetime.fromisoformat(str(status["updated_at"]))
        if updated.tzinfo is None:
            raise ValueError("status timestamp has no timezone")
    except (KeyError, TypeError, ValueError):
        return None, str(status.get("phase")) == "training"
    age = max(0, int((now - updated).total_seconds()))
    return age, str(status.get("phase")) == "training" and age > 180


def _frame_path(
    root: Path,
    lineage: dict[str, Any],
    status: dict[str, Any],
    name: str,
    *,
    allow_missing: bool,
) -> Path:
    if name == "latest":
        relative_frame: str | Path = Path("frames") / "latest.png"
    elif name in LESSON_IDS:
        lesson_frames = status.get("lesson_frames")
        if not isinstance(lesson_frames, dict) or not isinstance(lesson_frames.get(name), str):
            raise FileNotFoundError(f"no frame declared for lesson: {name}")
        relative_frame = lesson_frames[name]
    else:
        raise FileNotFoundError(f"unknown frame name: {name}")

    frame_relative = Path(relative_frame)
    if frame_relative.suffix.lower() != ".png":
        raise UnsafeDashboardPathError(f"dashboard may serve PNG frames only: {relative_frame}")
    lineage_relative = Path(str(lineage["directory"]))
    lineage_root = _safe_descendant(
        root,
        lineage_relative,
        expect_directory=True,
        allow_missing=allow_missing,
    )
    frame = _safe_descendant(
        root,
        lineage_relative / frame_relative,
        expect_directory=False,
        allow_missing=allow_missing,
    )
    if not _contains(lineage_root, frame):
        raise UnsafeDashboardPathError(f"frame escapes its declared lineage: {relative_frame}")
    return frame


def _frame_urls(
    root: Path,
    lineage: dict[str, Any],
    status: dict[str, Any],
) -> dict[str, str]:
    names: list[str] = []
    if int(status.get("frame_revision", 0) or 0) > 0:
        names.append("latest")
    lesson_frames = status.get("lesson_frames")
    if isinstance(lesson_frames, dict):
        names.extend(lesson_id for lesson_id, _label in LESSONS if lesson_id in lesson_frames)

    revision = int(status.get("frame_revision", 0) or 0)
    urls: dict[str, str] = {}
    for name in names:
        path = _frame_path(root, lineage, status, name, allow_missing=True)
        if path.exists():
            urls[name] = "/api/frame?" + urlencode(
                {"lineage": lineage["id"], "name": name, "v": revision}
            )
    return urls


def load_cohort_snapshot(cohort_root: Path) -> dict[str, Any]:
    """Read a fresh, public cohort view from the manifest and three statuses."""

    root = _validate_cohort_root(cohort_root)
    manifest = _load_manifest(root)
    active = manifest.get("active_lineage_id", manifest.get("active_lineage"))
    now = datetime.now(UTC)
    public_manifest = {
        field: manifest[field]
        for field in (
            "protocol",
            "cohort_id",
            "title",
            "phase",
            "started_at",
            "updated_at",
            "elapsed_seconds",
            "action_cap_per_lineage",
            "cohort_action_cap",
            "source_commit",
            "media_directory",
            "storage",
        )
        if field in manifest
    }
    public_manifest["active_lineage_id"] = active
    if str(public_manifest.get("phase", "")) in {"ready", "training"}:
        try:
            started_at = datetime.fromisoformat(str(public_manifest["started_at"]))
            if started_at.tzinfo is None:
                raise ValueError("cohort start timestamp has no timezone")
        except (KeyError, TypeError, ValueError):
            pass
        else:
            public_manifest["elapsed_seconds"] = max(
                0,
                int((now - started_at).total_seconds()),
            )

    public_lineages = []
    for order, lineage in enumerate(manifest["lineages"], start=1):
        public_lineage: dict[str, Any] = {
            "id": lineage["id"],
            "order": order,
            "label": str(lineage.get("label") or f"Lineage {order}"),
            "seed": lineage.get("seed"),
            "parent_seed": lineage.get("parent_seed"),
            "state": str(lineage.get("state") or "pending"),
            "status_available": False,
            "status": {},
            "frame_urls": {},
        }
        try:
            status = _load_status(root, lineage)
        except FileNotFoundError:
            public_lineage["status_error"] = "status.json not written yet"
        except DashboardDataError as error:
            if isinstance(error, UnsafeDashboardPathError):
                raise
            public_lineage["state"] = "unavailable"
            public_lineage["status_error"] = str(error)
        else:
            public_lineage["status_available"] = True
            public_lineage["status"] = _public_status(status)
            public_lineage["state"] = str(status.get("phase") or public_lineage["state"])
            age, stale = _status_freshness(status, now=now)
            public_lineage["status_age_seconds"] = age
            public_lineage["status_stale"] = stale
            public_lineage["frame_urls"] = _frame_urls(root, lineage, status)
        public_lineages.append(public_lineage)

    measured_storage = [
        status["storage"]
        for lineage in public_lineages
        if isinstance((status := lineage.get("status")), dict)
        and isinstance(status.get("storage"), dict)
        and isinstance(status["storage"].get("combined_used_bytes"), int)
    ]
    if measured_storage:
        latest = max(
            measured_storage,
            key=lambda value: int(value["combined_used_bytes"]),
        )
        storage = dict(public_manifest.get("storage") or {})
        cohort = latest.get("cohort")
        media = latest.get("media")
        if isinstance(cohort, dict):
            storage["cohort_scientific_used_bytes"] = cohort.get("used_bytes")
            storage["cohort_scientific_cap_bytes"] = cohort.get("cap_bytes")
        if isinstance(media, dict):
            storage["media_used_bytes"] = media.get("used_bytes")
            storage["media_cap_bytes"] = media.get("cap_bytes")
        storage["combined_used_bytes"] = latest["combined_used_bytes"]
        storage["combined_cap_bytes"] = latest.get("combined_cap_bytes")
        public_manifest["storage"] = storage

    return {
        "manifest": public_manifest,
        "lessons": [
            {"id": lesson_id, "label": label} for lesson_id, label in LESSONS
        ],
        "lineages": public_lineages,
        "loaded_at": now.isoformat(timespec="seconds"),
    }


def resolve_frame(cohort_root: Path, *, lineage_id: str, name: str) -> Path:
    """Resolve one status-declared PNG without exposing arbitrary cohort files."""

    root = _validate_cohort_root(cohort_root)
    manifest = _load_manifest(root)
    lineage = next(
        (entry for entry in manifest["lineages"] if entry["id"] == lineage_id),
        None,
    )
    if lineage is None:
        raise FileNotFoundError(f"unknown lineage: {lineage_id}")
    status = _load_status(root, lineage)
    return _frame_path(root, lineage, status, name, allow_missing=False)


class CohortDashboardHandler(BaseHTTPRequestHandler):
    server: CohortDashboardServer

    def log_message(self, _format: str, *args: Any) -> None:
        del args

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
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
        self.end_headers()
        if send_body:
            self.wfile.write(content)

    def _send_error_json(
        self,
        status: HTTPStatus,
        message: str,
        *,
        send_body: bool,
    ) -> None:
        payload = json.dumps({"error": message}, sort_keys=True).encode("utf-8")
        self._send(
            payload,
            content_type="application/json; charset=utf-8",
            status=status,
            send_body=send_body,
        )

    def _handle(self, *, send_body: bool) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send(
                COHORT_DASHBOARD_HTML.encode("utf-8"),
                content_type="text/html; charset=utf-8",
                send_body=send_body,
            )
            return
        if parsed.path == "/favicon.ico":
            self._send(b"", content_type="image/x-icon", status=HTTPStatus.NO_CONTENT, send_body=False)
            return
        try:
            if parsed.path == "/api/cohort.json":
                payload = json.dumps(
                    load_cohort_snapshot(self.server.cohort_root),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                self._send(
                    payload,
                    content_type="application/json; charset=utf-8",
                    send_body=send_body,
                )
                return
            if parsed.path == "/api/frame":
                query = parse_qs(parsed.query, strict_parsing=False)
                lineage_values = query.get("lineage", [])
                name_values = query.get("name", [])
                if len(lineage_values) != 1 or len(name_values) != 1:
                    raise FileNotFoundError("frame requires one lineage and one name")
                frame = resolve_frame(
                    self.server.cohort_root,
                    lineage_id=lineage_values[0],
                    name=name_values[0],
                )
                self._send(
                    _read_bytes(frame, maximum_bytes=FRAME_MAX_BYTES),
                    content_type="image/png",
                    send_body=send_body,
                )
                return
        except FileNotFoundError as error:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(error), send_body=send_body)
            return
        except UnsafeDashboardPathError as error:
            self._send_error_json(HTTPStatus.FORBIDDEN, str(error), send_body=send_body)
            return
        except (DashboardDataError, OSError) as error:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(error), send_body=send_body)
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "not found", send_body=send_body)

    def do_GET(self) -> None:
        self._handle(send_body=True)

    def do_HEAD(self) -> None:
        self._handle(send_body=False)

    def do_POST(self) -> None:
        self._send_error_json(HTTPStatus.METHOD_NOT_ALLOWED, "read-only dashboard", send_body=True)

    do_DELETE = do_POST
    do_PATCH = do_POST
    do_PUT = do_POST


def start_u2_dashboard(
    cohort_root: Path,
    *,
    port: int = DEFAULT_PORT,
) -> CohortDashboardServer:
    """Start one independent read-only server; never fall back to another port."""

    if port <= 0 or port > 65_535:
        raise ValueError("dashboard port must be between 1 and 65535; port 0 fallback is forbidden")
    root = _validate_cohort_root(cohort_root)
    _load_manifest(root)
    server = CohortDashboardServer((DEFAULT_HOST, port), CohortDashboardHandler)
    server.cohort_root = root
    thread = threading.Thread(
        target=server.serve_forever,
        name="u2-cohort-dashboard",
        daemon=True,
    )
    thread.start()
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort_root", type=Path)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = start_u2_dashboard(args.cohort_root, port=args.port)
    print(f"U2 cohort dashboard: http://{DEFAULT_HOST}:{server.server_port}/", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
