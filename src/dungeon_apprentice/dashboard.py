# ruff: noqa: E501, RUF001
"""Dependency-free live dashboard for a Dungeon Apprentice run."""

from __future__ import annotations

import argparse
import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from dungeon_apprentice.artifacts import atomic_write_text

DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Dungeon Apprentice — Live Run</title><style>
:root{color-scheme:dark;--bg:#09100d;--panel:#111c17;--line:#26382f;--ink:#ecf7ef;
--muted:#94aa9d;--green:#67df8a;--amber:#f1c969;--blue:#6dc8e8;--red:#ec7777}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0,#163326 0,
var(--bg) 35%);color:var(--ink);font:15px/1.5 ui-rounded,system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:32px 22px 60px}header{display:flex;align-items:end;
justify-content:space-between;gap:20px;margin-bottom:24px}h1{font-size:clamp(28px,5vw,52px);line-height:1;
margin:0;letter-spacing:-.045em}.eyebrow{color:var(--green);font-size:12px;font-weight:800;
letter-spacing:.16em;text-transform:uppercase;margin-bottom:10px}.live{display:flex;align-items:center;gap:8px;
color:var(--muted)}.dot{width:9px;height:9px;border-radius:50%;background:var(--amber);box-shadow:0 0 15px
var(--amber)}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.card{background:color-mix(in srgb,
var(--panel) 92%,transparent);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:
0 18px 50px #0004}.metric{grid-column:span 3}.metric b{display:block;font-size:28px;letter-spacing:-.03em}
.metric span,.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}
.frame{grid-column:span 4}.progress{grid-column:span 8}.history{grid-column:span 12}.screen{width:100%;
aspect-ratio:1;object-fit:contain;image-rendering:pixelated;background:#050806;border-radius:10px;border:1px solid var(--line)}
h2{font-size:18px;margin:0 0 14px}.tier{display:grid;grid-template-columns:110px 1fr 65px;gap:12px;
align-items:center;margin:16px 0}.bar{height:12px;background:#07100b;border-radius:20px;overflow:hidden}.fill{height:100%;
background:linear-gradient(90deg,var(--blue),var(--green));border-radius:20px;transition:width .4s}.value{text-align:right;
font-variant-numeric:tabular-nums}.events{width:100%;border-collapse:collapse}.events td,.events th{padding:9px 8px;
border-bottom:1px solid var(--line);text-align:left}.events th{color:var(--muted);font-size:11px;text-transform:uppercase}
.empty{color:var(--muted);padding:25px 0}.foot{color:var(--muted);margin-top:18px;font-size:12px}@media(max-width:760px){
.metric{grid-column:span 6}.frame,.progress,.history{grid-column:span 12}header{align-items:start;flex-direction:column}}
</style></head><body><main><header><div><div class="eyebrow">Local experiential learning</div><h1>Dungeon Apprentice</h1></div>
<div class="live"><i class="dot"></i><span id="freshness">Waiting for trainer…</span></div></header>
<section class="grid"><div class="card metric"><span>Training steps</span><b id="steps">—</b></div>
<div class="card metric"><span>Speed</span><b id="fps">—</b></div><div class="card metric"><span>Current lesson</span><b id="lesson">—</b></div>
<div class="card metric"><span>Run time</span><b id="runtime">—</b></div><div class="card frame"><h2>Agent’s current view</h2>
<img class="screen" id="frame" alt="The latest pixels seen by the policy"><div class="foot">This is the policy’s entire visual world.</div></div>
<div class="card progress"><h2>Frozen unseen-level exams</h2><div id="tiers" class="empty">No exam completed yet.</div>
<div class="foot">90% passes the current lesson. Earlier lessons must remain at or above 80%.</div></div>
<div class="card history"><h2>Evaluation history</h2><div id="history" class="empty">The first exam has not run yet.</div></div></section></main>
<script>
const fmt=n=>Number(n||0).toLocaleString();const age=s=>{if(!s)return 'Waiting for trainer…';let d=(Date.now()-Date.parse(s))/1000;
return d<10?'Live — updated just now':`Updated ${Math.floor(d)} seconds ago`};const duration=s=>{let n=Math.max(0,Number(s||0));
let h=Math.floor(n/3600),m=Math.floor(n%3600/60);return h?`${h}h ${m}m`:`${m}m`};
function render(d){steps.textContent=fmt(d.total_timesteps);fps.textContent=`${Math.round(d.fps||0)}/s`;lesson.textContent=d.current_tier_label||'Navigate';
runtime.textContent=duration(d.elapsed_seconds);freshness.textContent=age(d.updated_at);document.querySelector('.dot').style.background=d.phase==='training'?'var(--green)':'var(--amber)';
if(d.frame_revision)frame.src=`frames/latest.png?v=${d.frame_revision}`;let latest=d.latest_evaluations||[];
tiers.innerHTML=latest.length?latest.map(x=>`<div class="tier"><strong>${x.tier_label}</strong><div class="bar"><div class="fill" style="width:${100*x.success_rate}%"></div></div><div class="value">${Math.round(100*x.success_rate)}%</div></div>`).join(''):'<div class="empty">No exam completed yet.</div>';
let hist=(d.evaluation_history||[]).slice().reverse();history.innerHTML=hist.length?`<table class="events"><thead><tr><th>Step</th><th>Lesson</th><th>Success</th><th>Decision</th></tr></thead><tbody>${hist.map(x=>`<tr><td>${fmt(x.timesteps)}</td><td>${x.tier_label}</td><td>${Math.round(100*x.success_rate)}%</td><td>${x.decision||'Measured'}</td></tr>`).join('')}</tbody></table>`:'<div class="empty">The first exam has not run yet.</div>'}
async function poll(){try{let r=await fetch(`status.json?v=${Date.now()}`,{cache:'no-store'});if(r.ok)render(await r.json())}catch(e){}finally{setTimeout(poll,2000)}}poll();
</script></body></html>"""


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def prepare_dashboard(run_directory: Path) -> None:
    atomic_write_text(run_directory / "index.html", DASHBOARD_HTML)


def start_dashboard(run_directory: Path, *, host: str, port: int) -> DashboardServer:
    prepare_dashboard(run_directory)
    handler: Any = functools.partial(SimpleHTTPRequestHandler, directory=str(run_directory))
    server = DashboardServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, name="dashboard", daemon=True)
    thread.start()
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = start_dashboard(args.run_directory.resolve(), host=args.host, port=args.port)
    print(f"Dashboard: http://{args.host}:{server.server_port}/")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
