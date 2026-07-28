# ruff: noqa: E501
"""Read-only dashboard that follows the append-only U2r segment chain."""

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
from urllib.parse import parse_qs, quote, urlparse

from dungeon_apprentice import v02_u2r

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8786
STATUS_MAX_BYTES = 4 * 1024 * 1024
FRAME_MAX_BYTES = 16 * 1024 * 1024
LESSONS = (
    ("navigate/full", "Navigate"),
    ("unlock/u0-visible", "Visible Unlock"),
    ("unlock/u1-local", "Local Unlock"),
    ("unlock/u2-separated", "Separated Unlock"),
)
LESSON_IDS = frozenset(identifier for identifier, _label in LESSONS)

U2R_DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dungeon Apprentice — U2r Stability Run</title>
<style>
:root{color-scheme:dark;--bg:#07100d;--panel:#101d17;--line:#294035;--ink:#edf9f1;
--muted:#9ab1a2;--green:#69e18c;--blue:#6fcbea;--amber:#f0ca70;--red:#ef7c7c}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 15% 0,
#173a29 0,var(--bg) 38%);font:15px/1.5 ui-rounded,system-ui,sans-serif}
main{max-width:1220px;margin:auto;padding:28px 20px 60px}header{display:flex;align-items:flex-end;
justify-content:space-between;gap:18px;margin-bottom:22px}.eyebrow{color:var(--green);font-size:12px;
font-weight:850;letter-spacing:.16em;text-transform:uppercase;margin-bottom:8px}h1{font-size:
clamp(30px,6vw,54px);letter-spacing:-.05em;line-height:1;margin:0}h2{margin:0 0 14px;font-size:19px}
.fresh{display:flex;align-items:center;gap:8px;color:var(--muted)}.dot{width:9px;height:9px;
border-radius:50%;background:var(--amber);box-shadow:0 0 14px var(--amber)}.grid{display:grid;
grid-template-columns:repeat(12,1fr);gap:13px}.card{background:color-mix(in srgb,var(--panel) 94%,
transparent);border:1px solid var(--line);border-radius:17px;padding:17px;box-shadow:0 18px 48px #0004}
.metric{grid-column:span 2}.metric span,.tiny{color:var(--muted);font-size:11px;text-transform:uppercase;
letter-spacing:.08em}.metric b{display:block;font-size:24px;letter-spacing:-.03em;margin-top:3px}
.wide{grid-column:span 8}.side{grid-column:span 4}.full{grid-column:1/-1}.bar{height:12px;
background:#06100a;border-radius:20px;overflow:hidden;margin:13px 0}.fill{height:100%;width:0;
background:linear-gradient(90deg,var(--blue),var(--green));transition:width .35s}.exam{display:grid;
grid-template-columns:minmax(130px,1fr) minmax(120px,2fr) 150px;gap:12px;align-items:center;
padding:11px 0;border-bottom:1px solid var(--line)}.exam:last-child{border:0}.track{height:10px;
background:#06100a;border-radius:20px;overflow:hidden}.score{text-align:right;font-variant-numeric:tabular-nums}
.pills{display:flex;flex-wrap:wrap;gap:7px}.pill{padding:6px 10px;border:1px solid var(--line);
border-radius:999px;color:var(--muted);font-size:12px}.pill.good{border-color:var(--green);color:var(--green)}
.pill.bad{border-color:var(--red);color:var(--red)}.frames{display:grid;
grid-template-columns:repeat(5,1fr);gap:10px}figure{margin:0}figure img{width:100%;aspect-ratio:1;
object-fit:contain;image-rendering:pixelated;border:1px solid var(--line);border-radius:10px;
background:#040806}figcaption{font-size:11px;color:var(--muted);margin-top:5px}.evidence{display:grid;
grid-template-columns:1fr 1fr;gap:8px}.evidence div{padding:9px;background:#08130e;border-radius:9px;
overflow-wrap:anywhere}.evidence b{display:block;font-size:12px}.segments{display:flex;gap:7px;
flex-wrap:wrap}.empty,.note{color:var(--muted)}@media(max-width:850px){.metric{grid-column:span 4}
.wide,.side{grid-column:1/-1}.frames{grid-template-columns:repeat(2,1fr)}}@media(max-width:560px){
main{padding:22px 13px 44px}header{align-items:flex-start;flex-direction:column}.metric{grid-column:span 6}
.exam{grid-template-columns:1fr}.score{text-align:left}.evidence{grid-template-columns:1fr}}
</style></head><body><main><header><div><div class="eyebrow">U2r · bounded stability remediation</div>
<h1>One apprentice, eleven windows</h1></div><div class="fresh"><i class="dot"></i>
<span id="freshness">Loading the run…</span></div></header><section class="grid">
<div class="card metric"><span>Run state</span><b id="phase">—</b></div>
<div class="card metric"><span>Remediation learned</span><b id="learned">—</b></div>
<div class="card metric"><span>Actions remaining</span><b id="remaining">—</b></div>
<div class="card metric"><span>Current rate</span><b id="rate">—</b></div>
<div class="card metric"><span>Optimizer updates</span><b id="updates">—</b></div>
<div class="card metric"><span>Active segment</span><b id="segment">—</b></div>
<div class="card metric"><span>Scientific storage</span><b id="storage">—</b></div>
<div class="card wide"><h2>Frozen four-lesson exam</h2><div id="exams" class="empty">
No remediation exam has completed yet.</div></div><div class="card side"><h2>Stability verdict</h2>
<div id="verdict" class="pills"></div><div class="bar"><div id="progress" class="fill"></div></div>
<div id="progressText" class="note"></div></div><div class="card side"><h2>Evidence boundary</h2>
<div id="provenance" class="evidence"></div></div><div class="card wide"><h2>Immutable segments</h2>
<div id="segments" class="segments"></div></div><div class="card full"><h2>What the policy sees</h2>
<div id="frames" class="frames"></div><div class="note">Read-only snapshots; the policy receives only
its pixels and private recurrent state.</div></div></section></main>
<script>
const LESSONS=[['navigate/full','Navigate'],['unlock/u0-visible','Visible Unlock'],
['unlock/u1-local','Local Unlock'],['unlock/u2-separated','Separated Unlock']];
const byId=id=>document.getElementById(id),dot=document.querySelector('.dot');
const fmt=n=>Number.isFinite(Number(n))?Number(n).toLocaleString():'—';
const bytes=n=>{let x=Number(n)||0,u=['B','KiB','MiB','GiB'];let i=0;
while(x>=1024&&i<3){x/=1024;i++}return `${i?x.toFixed(2):Math.round(x)} ${u[i]}`};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',
"'":'&#39;'}[c]));const phaseLabel=p=>({training:'Training',eligible:'Eligible',failed:'Stability failed',
interrupted:'Interrupted',crashed:'Stopped'}[p]||String(p||'Waiting').replaceAll('_',' '));
function render(d){let s=d.active_status||{},phase=s.phase||'waiting',trained=Number(
s.remediation_trained_actions||0),remaining=Number(s.remaining_remediation_actions??360448);
byId('phase').textContent=phaseLabel(phase);byId('learned').textContent=fmt(trained);
byId('remaining').textContent=fmt(remaining);byId('rate').textContent=`${Math.round(Number(
s.actions_per_second||0))}/s`;byId('updates').textContent=fmt(s.optimizer_updates);
byId('segment').textContent=`#${d.active_segment_index??0}`;byId('storage').textContent=bytes(
(s.storage||{}).combined_used_bytes);let pct=Math.max(0,Math.min(100,
100*trained/360448));byId('progress').style.width=`${pct}%`;byId('progressText').textContent=
`${fmt(trained)} of 360,448 fixed remediation actions · terminal child count ${fmt(
s.child_trained_actions)}/1,048,576`;let stale=Boolean(d.heartbeat_stale),color=phase==='training'&&!stale?
'var(--green)':phase==='crashed'||phase==='failed'?'var(--red)':'var(--amber)';dot.style.background=color;
dot.style.boxShadow=`0 0 14px ${color}`;byId('freshness').textContent=stale?'Heartbeat delayed':
`Updated ${fmt(Math.floor(Number(d.heartbeat_age_seconds||0)))}s ago`;let evaluations=Array.isArray(
s.latest_evaluations)?s.latest_evaluations:[];byId('exams').innerHTML=LESSONS.map(([id,label])=>{
let x=evaluations.find(v=>v.lesson_id===id);if(!x)return `<div class="exam"><b>${label}</b><div
class="note">Awaiting exam</div><div class="score">—</div></div>`;let rate=100*Number(x.successes||0)/
Math.max(1,Number(x.episodes||80)),panels=Array.isArray(x.panel_successes)?x.panel_successes:[];
return `<div class="exam"><b>${label}</b><div class="track"><div class="fill" style="width:${rate}%">
</div></div><div class="score">${fmt(x.successes)}/${fmt(x.episodes)} · panels ${panels.map(fmt).join(
' / ')}</div></div>`}).join('');let controller=s.controller||{},eligible=controller.terminal_eligible;
let reasons=Array.isArray(controller.terminal_reasons)?controller.terminal_reasons:[];
byId('verdict').innerHTML=`<span class="pill ${eligible===true?'good':eligible===false?'bad':''}">
${eligible===true?'Terminal eligible':eligible===false?'Terminal failed':'Terminal-only gate pending'}</span>`+
reasons.map(x=>`<span class="pill bad">${esc(x)}</span>`).join('');let p=d.provenance||{};
byId('provenance').innerHTML=[['Source',p.source_commit],['Parent',p.parent_checkpoint_sha256],
['Exclusions',p.exclusion_set_sha256],['Anchor',p.anchor_tag_object],['Mount',p.storage_mount_path]]
.map(([k,v])=>
`<div><span class="tiny">${k}</span><b>${esc(String(v||'—').slice(0,16))}</b></div>`).join('');
byId('segments').innerHTML=(d.segments||[]).map(x=>`<span class="pill ${x.index===
d.active_segment_index?'good':''}">#${x.index} · ${esc(phaseLabel(x.phase))}</span>`).join('');
let frames=s.frame_urls||{};byId('frames').innerHTML=Object.entries(frames).map(([name,url])=>
`<figure><img src="${esc(url)}&v=${Date.now()}" alt="${esc(name)}"><figcaption>${esc(name)}</figcaption>
</figure>`).join('')||'<div class="empty">No frame has been published yet.</div>'}
let failures=0;async function poll(){try{let r=await fetch(`/api/u2r.json?v=${Date.now()}`,{cache:'no-store'});
if(!r.ok)throw new Error(r.status);render(await r.json());failures=0}catch(e){if(++failures>1){
dot.style.background='var(--red)';byId('freshness').textContent='Dashboard cannot read the U2r run'}}
finally{setTimeout(poll,2000)}}poll();
</script></body></html>"""


class U2rDashboardError(RuntimeError):
    """Raised when read-only U2r dashboard evidence is unsafe or malformed."""


class U2rDashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    run_root: Path


def _absolute_regular_directory(path: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        metadata = root.lstat()
    except OSError as error:
        raise FileNotFoundError(f"U2r run root does not exist: {root}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise U2rDashboardError(f"U2r run root is not a regular directory: {root}")
    return root


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise U2rDashboardError(f"unsafe or missing dashboard evidence: {path}")
    if path.stat().st_size > STATUS_MAX_BYTES:
        raise U2rDashboardError(f"dashboard evidence is too large: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise U2rDashboardError(f"cannot read dashboard evidence: {path}") from error
    if not isinstance(value, dict):
        raise U2rDashboardError(f"dashboard evidence is not an object: {path}")
    return value


def discover_u2r_segments(run_root: Path) -> tuple[tuple[int, Path], ...]:
    """Return one contiguous, symlink-free initial/resume segment chain."""

    root = _absolute_regular_directory(run_root)
    base = v02_u2r.DEFAULT_RUN_NAME
    resume_pattern = re.compile(rf"^{re.escape(base)}-resume-([1-9][0-9]*)$")
    found: dict[int, Path] = {}
    for child in root.iterdir():
        if child.name == base:
            index = 0
        else:
            match = resume_pattern.fullmatch(child.name)
            if match is None:
                if child.name.startswith(base):
                    raise U2rDashboardError(f"unknown U2r segment name: {child.name}")
                continue
            index = int(match.group(1))
        if child.is_symlink() or not child.is_dir():
            raise U2rDashboardError(f"unsafe U2r segment: {child}")
        found[index] = child
    if not found or 0 not in found or set(found) != set(range(max(found) + 1)):
        raise U2rDashboardError("U2r segment chain is absent or non-contiguous")
    return tuple((index, found[index]) for index in range(max(found) + 1))


def _public_status(status: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    source = status.get("source") if isinstance(status.get("source"), Mapping) else {}
    parent = status.get("parent") if isinstance(status.get("parent"), Mapping) else {}
    controller = (
        status.get("controller") if isinstance(status.get("controller"), Mapping) else {}
    )
    curriculum = (
        status.get("curriculum") if isinstance(status.get("curriculum"), Mapping) else {}
    )
    storage = status.get("storage") if isinstance(status.get("storage"), Mapping) else {}
    storage_mount = (
        status.get("storage_mount")
        if isinstance(status.get("storage_mount"), Mapping)
        else {}
    )
    evaluations = status.get("latest_evaluations")
    if not isinstance(evaluations, list):
        evaluations = []
    public_evaluations = [
        {
            key: value
            for key, value in item.items()
            if key
            in {
                "lesson_id",
                "lesson_label",
                "successes",
                "episodes",
                "success_rate",
                "panel_successes",
                "mean_ineffective_interactions",
                "passed",
            }
        }
        for item in evaluations
        if isinstance(item, Mapping) and item.get("lesson_id") in LESSON_IDS
    ]
    return {
        "segment_index": index,
        "protocol": status.get("protocol"),
        "phase": status.get("phase"),
        "source": {"commit": source.get("commit"), "dirty": source.get("dirty")},
        "parent_checkpoint_sha256": parent.get("checkpoint_sha256"),
        "started_at": status.get("started_at"),
        "updated_at": status.get("updated_at"),
        "elapsed_seconds": status.get("elapsed_seconds"),
        "actions_per_second": status.get("actions_per_second", status.get("fps")),
        "child_trained_actions": status.get("child_trained_actions"),
        "remediation_trained_actions": status.get("remediation_trained_actions"),
        "remaining_remediation_actions": status.get("remaining_remediation_actions"),
        "lifetime_trained_actions": status.get("lifetime_trained_actions"),
        "optimizer_updates": status.get("optimizer_updates"),
        "latest_evaluations": public_evaluations,
        "controller": {
            "terminal_eligible": controller.get("terminal_eligible"),
            "terminal_reasons": controller.get("terminal_reasons", []),
            "terminal_pair": controller.get("terminal_pair"),
        },
        "curriculum": {
            "recovery": curriculum.get("recovery"),
            "recovery_cause": curriculum.get("recovery_cause"),
        },
        "storage": dict(storage),
        "storage_mount": {
            "path": storage_mount.get("path"),
            "is_mount": storage_mount.get("is_mount"),
            "distinct_from_system": storage_mount.get("distinct_from_system"),
        },
    }


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


def _frame_names(status: Mapping[str, Any], segment_index: int) -> dict[str, str]:
    names: dict[str, str] = {}
    frames = status.get("lesson_frames")
    if isinstance(frames, Mapping):
        for lesson in LESSON_IDS:
            if isinstance(frames.get(lesson), str):
                names[lesson] = (
                    f"/api/frame?segment={segment_index}&name={quote(lesson, safe='')}"
                )
    latest = status.get("frame_revision")
    if isinstance(latest, int) and latest > 0:
        names["live"] = f"/api/frame?segment={segment_index}&name=live"
    return names


def load_u2r_snapshot(run_root: Path) -> dict[str, Any]:
    """Load a bounded public view and automatically follow the newest segment."""

    segments = discover_u2r_segments(run_root)
    loaded: list[tuple[int, Path, dict[str, Any]]] = []
    for index, directory in segments:
        status_path = directory / "status.json"
        if not status_path.exists():
            status: dict[str, Any] = {
                "protocol": v02_u2r.PROTOCOL,
                "phase": "starting",
            }
        else:
            status = _read_json(status_path)
            if status.get("protocol") != v02_u2r.PROTOCOL:
                raise U2rDashboardError("segment status has the wrong protocol")
        loaded.append((index, directory, status))
    training = [item for item in loaded if item[2].get("phase") == "training"]
    if len(training) > 1:
        raise U2rDashboardError("more than one U2r segment claims to be training")
    active = training[0] if training else loaded[-1]
    active_index, _active_directory, active_raw = active
    public = _public_status(active_raw, index=active_index)
    public["frame_urls"] = _frame_names(active_raw, active_index)
    heartbeat_age = _timestamp_age(public.get("updated_at"))
    parent = (
        active_raw.get("parent")
        if isinstance(active_raw.get("parent"), Mapping)
        else {}
    )
    exclusions = (
        active_raw.get("exclusions")
        if isinstance(active_raw.get("exclusions"), Mapping)
        else {}
    )
    anchor = (
        active_raw.get("external_preregistration")
        if isinstance(active_raw.get("external_preregistration"), Mapping)
        else {}
    )
    return {
        "protocol": v02_u2r.PROTOCOL,
        "loaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "active_segment_index": active_index,
        "heartbeat_age_seconds": heartbeat_age,
        "heartbeat_stale": (
            active_raw.get("phase") == "training"
            and (heartbeat_age is None or heartbeat_age > 180)
        ),
        "segments": [
            {
                "index": index,
                "name": directory.name,
                "phase": status.get("phase", "unavailable"),
                "updated_at": status.get("updated_at"),
                "remediation_trained_actions": status.get(
                    "remediation_trained_actions"
                ),
            }
            for index, directory, status in loaded
        ],
        "active_status": public,
        "provenance": {
            "source_commit": public["source"].get("commit"),
            "parent_checkpoint_sha256": parent.get("checkpoint_sha256"),
            "exclusion_set_sha256": exclusions.get("exact_layout_set_sha256"),
            "anchor_tag_object": anchor.get("tag_object"),
            "storage_mount_path": public["storage_mount"].get("path"),
        },
    }


def resolve_u2r_frame(
    run_root: Path,
    *,
    segment_index: int,
    name: str,
) -> Path:
    """Resolve only a status-declared lesson frame or the fixed live frame."""

    segments = dict(discover_u2r_segments(run_root))
    directory = segments.get(int(segment_index))
    if directory is None:
        raise FileNotFoundError("unknown U2r segment")
    status = _read_json(directory / "status.json")
    if name == "live":
        relative_value: Any = "frames/latest.png"
    elif name in LESSON_IDS:
        frames = status.get("lesson_frames")
        relative_value = frames.get(name) if isinstance(frames, Mapping) else None
    else:
        raise FileNotFoundError("unknown U2r frame")
    if not isinstance(relative_value, str):
        raise FileNotFoundError("U2r frame has not been published")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise U2rDashboardError("U2r frame path escapes its segment")
    candidate = directory / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError("U2r frame is missing or unsafe")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(directory.resolve())
    except ValueError as error:
        raise U2rDashboardError("U2r frame path escapes its segment") from error
    if resolved.stat().st_size > FRAME_MAX_BYTES:
        raise U2rDashboardError("U2r frame is too large")
    return resolved


class U2rDashboardHandler(BaseHTTPRequestHandler):
    server_version = "DungeonApprenticeU2rDashboard/1"

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
            json.dumps({"error": message}, sort_keys=True).encode(),
            content_type="application/json; charset=utf-8",
            status=status,
            send_body=send_body,
        )

    def _handle(self, *, send_body: bool) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send(
                U2R_DASHBOARD_HTML.encode(),
                content_type="text/html; charset=utf-8",
                send_body=send_body,
            )
            return
        try:
            if parsed.path == "/api/u2r.json":
                payload = json.dumps(
                    load_u2r_snapshot(self.server.run_root),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
                self._send(
                    payload,
                    content_type="application/json; charset=utf-8",
                    send_body=send_body,
                )
                return
            if parsed.path == "/api/frame":
                query = parse_qs(parsed.query)
                segment_values = query.get("segment", [])
                name_values = query.get("name", [])
                if len(segment_values) != 1 or len(name_values) != 1:
                    raise FileNotFoundError("frame requires one segment and name")
                frame = resolve_u2r_frame(
                    self.server.run_root,
                    segment_index=int(segment_values[0]),
                    name=name_values[0],
                )
                self._send(
                    frame.read_bytes(),
                    content_type="image/png",
                    send_body=send_body,
                )
                return
        except (FileNotFoundError, ValueError) as error:
            self._error(HTTPStatus.NOT_FOUND, str(error), send_body=send_body)
            return
        except (OSError, U2rDashboardError) as error:
            self._error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                str(error),
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


def start_u2r_dashboard(
    run_root: Path,
    *,
    port: int = DEFAULT_PORT,
) -> U2rDashboardServer:
    """Start the fixed local read-only dashboard without port fallback."""

    if port <= 0 or port > 65_535:
        raise ValueError("dashboard port must be between 1 and 65535")
    root = _absolute_regular_directory(run_root)
    discover_u2r_segments(root)
    server = U2rDashboardServer((DEFAULT_HOST, port), U2rDashboardHandler)
    server.run_root = root
    thread = threading.Thread(
        target=server.serve_forever,
        name="u2r-dashboard",
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
    server = start_u2r_dashboard(args.run_root, port=args.port)
    print(
        f"U2r dashboard: http://{DEFAULT_HOST}:{server.server_port}/",
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
