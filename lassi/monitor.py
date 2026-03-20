"""
lassi/monitor.py — Live browser dashboard for monitoring the LASSI-Verify pipeline.

Usage (add to notebook imports cell, then run once):
    from lassi.monitor import start_monitor, log_figure, save_log
    start_monitor()   # redirects stdout/stderr and display(); shows clickable URL

Open the URL in a new browser tab and keep it visible while the pipeline runs.
The dashboard streams all print() output and IPython HTML/Markdown display output
in real-time, with timestamps and auto-scroll.

For matplotlib figures, plt.show() is automatically forwarded to the dashboard
after start_monitor() is called. You can also call log_figure() explicitly:
    log_figure()          # sends plt.gcf() to the dashboard
    log_figure(my_fig)    # sends a specific Figure

To save the log at the end of a run:
    save_log(path)        # writes all log text to a file; HTML blocks are tag-stripped

Works on Linux GUI, macOS, Windows, and over SSH tunnels (forward port 5007).
No external dependencies — uses only the Python standard library.
"""

import os
import sys
import re
import json
import base64
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime

# ── module-level state ──────────────────────────────────────────────────────────
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
_TAG_RE  = re.compile(r'<[^>]+>')
_lock = threading.Lock()
_subscribers: list = []    # list[queue.Queue] — one per open browser connection
_history: list = []        # all SSE messages (replayed to late-joining connections)
_original_stdout = None
_original_stderr = None
_started = False
_suppress_display = False  # True during start_monitor()'s own link display


# ── SSE event publisher ─────────────────────────────────────────────────────────

def _publish(event_type: str, payload: str) -> None:
    """Broadcast an SSE event to all subscribers and append to history."""
    msg = json.dumps({
        "type": event_type,
        "data": payload,
        "ts":   datetime.now().strftime("%H:%M:%S"),
    })
    with _lock:
        _history.append(msg)
        for q in _subscribers:
            q.put(msg)


def _strip_html(html: str) -> str:
    """Strip HTML tags and decode basic entities to plain text."""
    text = _TAG_RE.sub('', html)
    text = (text.replace('&amp;', '&').replace('&lt;', '<')
                .replace('&gt;', '>').replace('&nbsp;', ' ')
                .replace('&quot;', '"').replace('&#39;', "'"))
    text = re.sub(r'\n{3,}', '\n\n', text)   # collapse excessive blank lines
    return text.strip()


def _log_lines() -> list:
    """Return all log/display history entries as formatted '[HH:MM:SS] text\\n' strings.

    HTML display blocks are tag-stripped and prefixed with '[display]'.
    Figures are omitted.
    """
    lines = []
    with _lock:
        snapshot = list(_history)
    for msg in snapshot:
        try:
            ev = json.loads(msg)
            if ev.get("type") == "log":
                lines.append(f"[{ev['ts']}] {ev['data']}\n")
            elif ev.get("type") == "html":
                text = _strip_html(ev['data'])
                if text:
                    for line in text.splitlines():
                        if line.strip():
                            lines.append(f"[{ev['ts']}] [display] {line}\n")
        except (json.JSONDecodeError, KeyError):
            pass
    return lines


# ── stdout/stderr tee stream ────────────────────────────────────────────────────

class _TeeStream:
    """
    Wraps a stream (sys.stdout / sys.stderr) to duplicate output to the SSE
    dashboard.  Uses line-buffering so each dashboard entry is a complete line.
    ANSI color codes are stripped before sending to the browser.
    """
    def __init__(self, original):
        self._orig = original
        self._buf = ""

    def write(self, text: str) -> None:
        self._orig.write(text)
        clean = _ANSI_RE.sub("", text)
        self._buf += clean
        # Flush complete lines to the dashboard
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:   # skip blank lines produced by the split
                _publish("log", line)

    def flush(self) -> None:
        self._orig.flush()
        # Push any trailing partial line (e.g. print(..., end=""))
        if self._buf.strip():
            _publish("log", self._buf)
            self._buf = ""

    def __getattr__(self, name):
        # Delegate encoding, errors, isatty(), fileno(), etc. to the original
        return getattr(self._orig, name)


# ── IPython display publisher patch ────────────────────────────────────────────

def _patch_display_pub() -> None:
    """Hook into IPython's display publisher to capture all display() calls.

    Patches ip.display_pub.publish rather than IPython.display.display, so it
    works regardless of how pipeline modules imported the display function.
    HTML content is forwarded as an 'html' SSE event; plain-text-only output
    falls back to a 'log' event.
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return
        orig_publish = ip.display_pub.publish

        def _intercept(data, metadata=None, **kwargs):
            orig_publish(data, metadata, **kwargs)
            if _suppress_display:
                return
            if not isinstance(data, dict):
                return
            if 'text/html' in data:
                html = data['text/html']
                if html and html.strip():
                    _publish("html", html)
            elif 'text/markdown' in data:
                md = data['text/markdown']
                if md and md.strip():
                    try:
                        import markdown as _mdlib
                        html = _mdlib.markdown(md, extensions=['tables', 'fenced_code'])
                    except Exception:
                        # fallback: render as preformatted text
                        esc = md.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        html = f'<pre style="white-space:pre-wrap;margin:0">{esc}</pre>'
                    _publish("html", html)
            elif 'text/plain' in data:
                text = data['text/plain']
                if text and text.strip() and not text.strip().startswith('<IPython'):
                    _publish("log", text)

        ip.display_pub.publish = _intercept
    except Exception:
        pass   # Non-IPython environment — silently skip


# ── HTML dashboard page ─────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>LASSI Monitor</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #1e1e1e; color: #d4d4d4;
  font-family: 'Consolas', 'Courier New', monospace; font-size: 13px;
  display: flex; flex-direction: column; height: 100vh; overflow: hidden;
}
/* ── header ── */
#hdr {
  padding: 6px 14px; background: #252526;
  border-bottom: 1px solid #3c3c3c;
  display: flex; align-items: center; gap: 14px; flex-shrink: 0;
}
#hdr h1 { font-size: 14px; color: #9cdcfe; letter-spacing: .4px; }
#status { font-size: 11px; }
.ok  { color: #4ec9b0; }
.err { color: #f44747; }
.btn {
  cursor: pointer; font-size: 11px; padding: 3px 10px; border-radius: 3px;
  border: 1px solid #555; background: #3a3a3a; color: #ccc;
}
.btn.on { background: #0e639c; border-color: #1177bb; color: #fff; }
/* ── body ── */
#body { display: flex; flex: 1; overflow: hidden; }
/* log pane */
#log { flex: 1; overflow-y: auto; padding: 8px 14px; line-height: 1.55; }
/* plain text log lines */
.log-line {
  display: block;
  white-space: pre-wrap; word-break: break-word;
}
.ts { color: #6a9955; user-select: none; margin-right: 6px; font-size: 11px; }
/* HTML display blocks — from IPython display(HTML(...)) / display(Markdown(...)) */
details.html-collapsible { margin: 6px 0; }
details.html-collapsible > summary {
  padding: 3px 10px; cursor: pointer; list-style: none; user-select: none;
  color: #569cd6; font-style: italic;
  background: #252526; border-left: 4px solid #0e639c; border-radius: 2px;
}
details.html-collapsible > summary::-webkit-details-marker { display: none; }
details.html-collapsible > summary::before { content: '▶  '; font-size: 10px; }
details.html-collapsible[open] > summary::before { content: '▼  '; }
.html-block {
  background: #252526;
  border-left: 4px solid #0e639c;
  padding: 8px 12px;
  margin: 0;
  overflow-x: auto;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 13px;
  line-height: 1.5;
}
.html-block .ts { font-family: 'Consolas', monospace; display: block; margin-bottom: 4px; }
.html-block table { border-collapse: collapse; width: 100%; margin-top: 4px; }
.html-block td, .html-block th {
  border: 1px solid #3c3c3c; padding: 4px 10px; text-align: left;
}
.html-block th { background: #2d2d2d; color: #9cdcfe; }
.html-block tr:nth-child(even) { background: #2a2a2a; }
.html-block pre, .html-block code {
  background: #2d2d2d; padding: 2px 5px; border-radius: 3px;
  white-space: pre-wrap; word-break: break-word; font-family: 'Consolas', monospace;
}
/* ── line classification colours ── */
.lassi-start {
  color: #9cdcfe; font-weight: bold; font-size: 15px; letter-spacing: 0.4px;
  background: #1a2a3a; border-top: 3px solid #569cd6; border-bottom: 1px solid #3a5a8a;
  margin-top: 16px; padding: 7px 4px;
}
.stage-header {
  color: #569cd6; font-weight: bold;
  border-top: 1px solid #2a3a4a; margin-top: 10px; padding-top: 4px;
}
.run-info {
  color: #9cdcfe; font-size: 13px; background: #1a2a3a;
  padding: 3px 8px; border-bottom: 1px solid #3a5a8a; margin-bottom: 6px;
}
.tier-pass    { color: #4ec9b0; font-weight: bold; }
.tier-fail    { color: #f44747; font-weight: bold; }
.tier-skip    { color: #dcdcaa; }
.iteration-restart {
  color: #f4a247; font-weight: bold; font-size: 15px; letter-spacing: 0.4px;
  background: #2a1a08; border-top: 3px solid #a06020; border-bottom: 1px solid #7a4010;
  margin-top: 16px; padding: 7px 4px;
}
.self-correct { color: #f4a247; font-weight: bold; }
.loop-step    { color: #5a8aaa; }
.exec-success { color: #4ec9b0; font-weight: bold; }
.warn-line    { color: #ce9178; }
.debug-line   { color: #6a6a6a; font-size: 11px; }
.pipeline-done {
  color: #4ec9b0; font-weight: bold; font-size: 14px;
  border-top: 2px solid #2a5040; margin-top: 10px; padding-top: 6px;
}
.tier-override { color: #dcdcaa; font-style: italic; }
.verify-insert { color: #c586c0; font-weight: bold; }
/* collapsible prompt-review blocks */
/* shared collapsible block base — accent colour vars set per type */
details.collapsible-block {
  border-radius: 3px; margin: 5px 0;
}
details.collapsible-block > summary {
  padding: 3px 10px; cursor: pointer; list-style: none;
  font-style: italic; user-select: none;
}
details.collapsible-block > summary::-webkit-details-marker { display: none; }
details.collapsible-block > summary::before { content: '▶  '; font-size: 10px; }
details.collapsible-block[open] > summary::before { content: '▼  '; }
.collapsible-body { padding: 2px 14px; border-top: 1px solid #222; }
.collapsible-cap {
  margin-top: 4px; padding: 2px 4px; font-size: 10px;
  text-align: left; font-style: italic; user-select: none;
  cursor: pointer;
}
.collapsible-cap:hover { opacity: 0.7; }
/* prompt-review block — green accent */
details.prompt-block {
  border-left: 4px solid #6a9955; border-bottom: 2px solid #3a5a3a;
  border-top: 1px solid #3a5a3a; border-right: 1px solid #3a5a3a;
}
details.prompt-block > summary { color: #6a9955; }
.prompt-block .collapsible-cap { color: #4a7a4a; border-top: 1px solid #2e4e2e; }
/* code-execution output block — teal accent */
details.exec-block {
  border-left: 4px solid #3a8a8a; border-bottom: 2px solid #2a5a5a;
  border-top: 1px solid #2a5a5a; border-right: 1px solid #2a5a5a;
}
details.exec-block > summary { color: #4ec9b0; }
.exec-block .collapsible-cap { color: #3a7a7a; border-top: 1px solid #1e4a4a; }
/* refactoring plan block — steel-blue accent */
details.refplan-block {
  border-left: 4px solid #7a9abf; border-bottom: 2px solid #4a6a8f;
  border-top: 1px solid #4a6a8f; border-right: 1px solid #4a6a8f;
}
details.refplan-block > summary { color: #9cdcfe; }
.refplan-block .collapsible-cap { color: #5a7a9f; border-top: 1px solid #3a5a7f; }
/* LLM response section markers */
.llm-response  { color: #9cdcfe; font-style: italic; }
/* metrics / results section */
.metrics-header { color: #dcdcaa; font-weight: bold; border-top: 1px solid #3a3a20; margin-top: 8px; padding-top: 4px; }
.metrics-sub    { color: #7fb3c8; font-weight: bold; }
/* inline verdict badges */
.badge   { display: inline-block; padding: 0 5px; border-radius: 3px;
           font-weight: bold; font-size: 11px; vertical-align: baseline; margin: 0 2px; }
.b-pass  { background: #163020; color: #4ec9b0; border: 1px solid #2a5040; }
.b-fail  { background: #301616; color: #f44747; border: 1px solid #503030; }
.b-skip  { background: #302a10; color: #dcdcaa; border: 1px solid #504020; }
.b-valid { background: #163020; color: #4ec9b0; border: 1px solid #2a5040; }
.b-inv   { background: #301616; color: #f44747; border: 1px solid #503030; }
/* figures sidebar — hidden until first figure arrives */
#figs {
  display: none; flex-direction: column; gap: 8px;
  overflow-y: auto; padding: 8px; width: 440px;
  border-left: 1px solid #3c3c3c; flex-shrink: 0;
}
.fig-ts { font-size: 10px; color: #888; text-align: right; margin-bottom: 2px; }
#figs img { width: 100%; border: 1px solid #555; border-radius: 3px; }
</style>
</head><body>
<div id="hdr">
  <h1>&#9654; LASSI Monitor</h1>
  <span id="status" class="err">connecting…</span>
  <button class="btn on" id="asbtn" onclick="toggleAS()">Auto-scroll</button>
  <button class="btn" onclick="saveLog()">&#11015; Save log</button>
</div>
<div id="body">
  <div id="log"></div>
  <div id="figs"></div>
</div>
<script>
const log    = document.getElementById('log');
const figs   = document.getElementById('figs');
const status = document.getElementById('status');
let autoScroll = true;

/* ── auto-scroll: pause on manual scroll-up, resume when back at bottom ── */
log.addEventListener('scroll', () => {
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 60;
  if (!atBottom && autoScroll) {
    autoScroll = false;
    document.getElementById('asbtn').className = 'btn';
  } else if (atBottom && !autoScroll) {
    autoScroll = true;
    document.getElementById('asbtn').className = 'btn on';
  }
});

function toggleAS() {
  autoScroll = !autoScroll;
  document.getElementById('asbtn').className = autoScroll ? 'btn on' : 'btn';
  if (autoScroll) log.scrollTop = log.scrollHeight;
}
function saveLog() {
  fetch('/log')
    .then(r => r.text())
    .then(text => {
      const ts  = new Date().toISOString().slice(0,19).replace(/[T:]/g, '-');
      const a   = document.createElement('a');
      a.href     = 'data:text/plain;charset=utf-8,' + encodeURIComponent(text);
      a.download = 'monitor_log_' + ts + '.txt';
      a.click();
    });
}
function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
/* Classify a log line into a CSS class for whole-line colouring. */
function classify(text) {
  const t = text.trim();
  if (t.startsWith('*** DEBUG:'))                                       return 'debug-line';
  if (/\[WARNING\]/.test(t))                                            return 'warn-line';
  if (/-----> LASSI IS FINISHED|LASSI-EE CODE GENERATION COMPLETE/.test(t)) return 'pipeline-done';
  if (/LASSI Iteration Report/.test(t))                                  return 'iteration-restart';
  if (/SELF-CORRECTION:|SELF-IMPROVEMENT FOR ENERGY/.test(t))           return 'self-correct';
  if (/STARTING CODE GENERATION|POST-PROCESSING of GENERATED CODE/.test(t)) return 'loop-step';
  if (/Three-tier override:/.test(t))                                   return 'tier-override';
  if (/LASSI-Verify Tier [123]:.*\bPASS\b/.test(t) ||
      /Tier 3 \[.*\]: PASS/.test(t) ||
      /\[Base LLM Verification\].*\bPASS\b/.test(t) ||
      /Generated refactored code is better/.test(t))                    return 'tier-pass';
  if (/LASSI-Verify Tier [123]:.*\bFAIL\b/.test(t) ||
      /Tier 3 \[.*\]: FAIL/.test(t) ||
      /\[Base LLM Verification\].*\bFAIL\b/.test(t) ||
      /NOT better than source/.test(t) ||
      t.startsWith('\u274c'))                                            return 'tier-fail';
  if (/LASSI-Verify Tier [123]:.*\bSKIP\b/.test(t) ||
      /Tier 3 \[.*\]: SKIP/.test(t) ||
      /\[Base LLM Verification\].*\bSKIP\b/.test(t))                    return 'tier-skip';
  if (/Program executed successfully/.test(t))                                      return 'exec-success';
  if (/STARTING CODE INSERT/.test(t))                                               return 'verify-insert';
  if (/STARTING LASSI:/.test(t))                                                    return 'lassi-start';
  if (/STARTING (STAGE|LASSI)|WELCOME TO PIPELINE|SOURCE CODE: Test compile/.test(t)) return 'stage-header';
  if (t.startsWith('>>> App:'))                                                       return 'run-info';
  if (/LLM RESPONSE:|END LLM RESPONSE/.test(t))                                    return 'llm-response';
  if (/Power and Energy Feedback data/.test(t))                                     return 'metrics-header';
  if (/Energy consumed \(J\)|Average power \(W\)|Runtime \(s\)|Tracking metrics/.test(t)) return 'metrics-sub';
  return '';
}
/* Wrap verdict keywords in inline badge spans (applied after esc()). */
function badge(escaped) {
  return escaped
    .replace(/\bINVALID\b/g, '<span class="badge b-inv">INVALID</span>')
    .replace(/\bVALID\b/g,   '<span class="badge b-valid">VALID</span>')
    .replace(/\bPASS\b/g,    '<span class="badge b-pass">PASS</span>')
    .replace(/\bFAIL\b/g,    '<span class="badge b-fail">FAIL</span>')
    .replace(/\bSKIP\b/g,    '<span class="badge b-skip">SKIP</span>');
}
/* ── generic collapsible block state (prompt-review, code-execution, etc.) ── */
let _blockTarget = null;   // non-null while inside any collapsible block
let _blockBody   = null;   // kept so _closeBlock can append the cap line

function _openBlock(ts, label, cssClass, capText) {
  const details = document.createElement('details');
  details.className = 'collapsible-block ' + cssClass;
  const summary = document.createElement('summary');
  summary.innerHTML = '<span class="ts">' + ts + '</span>' + label;
  details.appendChild(summary);
  const body = document.createElement('div');
  body.className = 'collapsible-body';
  details.appendChild(body);
  log.appendChild(details);
  _blockTarget       = body;
  _blockBody         = body;
  _blockBody._capText = capText;   // store cap text on the element for _closeBlock
  if (autoScroll) log.scrollTop = log.scrollHeight;
}
function _closeBlock() {
  if (_blockBody) {
    const capText = _blockBody._capText;
    if (capText) {
      const cap = document.createElement('div');
      cap.className = 'collapsible-cap';
      cap.textContent = capText;
      cap.onclick = () => { const d = cap.closest('details'); if (d) d.open = false; };
      _blockBody.appendChild(cap);
    }
  }
  _blockTarget = null;
  _blockBody   = null;
}

function addLog(ts, text) {
  // Close-and-fall-through: line signals end of current block AND is rendered normally
  if (_blockTarget && /THE REFACTORING PLAN PROMPT/.test(text)) { _closeBlock(); }

  if (/=====> VERIFY THE PROMPTING <=====/.test(text)) {
    const tm = text.match(/\[tokens:\s*([\d,]+)\]/);
    const label = tm ? 'PROMPT REVIEW  ·  ' + tm[1] + ' tokens' : 'PROMPT REVIEW';
    _openBlock(ts, label, 'prompt-block', '▲  end of prompt review'); return;
  }
  if (/END PROMPT REVIEW/.test(text))                  { _closeBlock(); return; }
  if (/< BEGIN: STANDARD OUTPUT FROM CODE EXECUTION >/.test(text)) {
    _openBlock(ts, 'CODE EXECUTION OUTPUT', 'exec-block', '▲  end of execution output'); return;
  }
  if (/< END: Standard Output >/.test(text))           { _closeBlock(); return; }
  if (/LLM RESPONSE: refactoring plan/.test(text)) {
    _openBlock(ts, 'REFACTORING PLAN', 'refplan-block', '▲  end of refactoring plan'); return;
  }
  const cls    = classify(text);
  const node   = document.createElement('span');
  node.className = 'log-line' + (cls ? ' ' + cls : '');
  node.innerHTML = '<span class="ts">' + ts + '</span>' + badge(esc(text));
  (_blockTarget || log).appendChild(node);
  if (autoScroll) log.scrollTop = log.scrollHeight;
}
function addHtml(ts, html) {
  // Extract a short label from the first non-empty text in the HTML
  const _tmp = document.createElement('div');
  _tmp.innerHTML = html;
  const _label = (_tmp.textContent || '').trim().slice(0, 60).replace(/\s+/g, ' ') || 'display output';

  const details = document.createElement('details');
  details.className = 'html-collapsible';
  const summary = document.createElement('summary');
  summary.innerHTML = '<span class="ts">' + ts + '</span>' + esc(_label) + ((_label.length >= 60) ? '…' : '');
  details.appendChild(summary);

  const wrap = document.createElement('div');
  wrap.className = 'html-block';
  wrap.innerHTML = html;
  details.appendChild(wrap);

  (_blockTarget || log).appendChild(details);
  if (autoScroll) log.scrollTop = log.scrollHeight;
}
function addFig(ts, b64) {
  figs.style.display = 'flex';
  const wrap = document.createElement('div');
  wrap.innerHTML = '<div class="fig-ts">' + ts + '</div>'
                 + '<img src="data:image/png;base64,' + b64 + '">';
  figs.prepend(wrap);   // newest figure on top
}

const es = new EventSource('/events');
es.onopen  = () => { status.textContent = 'connected'; status.className = 'ok'; };
es.onerror = () => {
  status.textContent = 'disconnected — reload to reconnect';
  status.className = 'err';
};
es.onmessage = ev => {
  const e = JSON.parse(ev.data);
  if (e.type === 'clear')  { log.innerHTML = ''; }
  if (e.type === 'log')    addLog(e.ts, e.data);
  if (e.type === 'html')   addHtml(e.ts, e.data);
  if (e.type === 'figure') addFig(e.ts, e.data);
};
</script>
</body></html>"""


# ── HTTP / SSE handler ──────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body = _HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type",   "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type",  "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection",    "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            q = queue.Queue()
            with _lock:
                snapshot = list(_history)
                _subscribers.append(q)
            try:
                # Replay history so a late-joining browser sees everything
                for msg in snapshot:
                    self.wfile.write(f"data: {msg}\n\n".encode())
                self.wfile.flush()
                # Stream live events
                while True:
                    try:
                        msg = q.get(timeout=20)
                        self.wfile.write(f"data: {msg}\n\n".encode())
                        self.wfile.flush()
                    except queue.Empty:
                        # SSE keepalive comment — prevents browser timeout
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with _lock:
                    if q in _subscribers:
                        _subscribers.remove(q)

        elif self.path == "/log":
            body = "".join(_log_lines()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type",   "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_error(404)

    def log_message(self, *_):   # silence per-request HTTP logs
        pass


# ── public API ──────────────────────────────────────────────────────────────────

def clear_log() -> None:
    """Clear the in-memory log history and wipe the browser display.

    Call this at the start of each pipeline run so that the saved log only
    contains output from the current run.  Safe to call even if start_monitor()
    has not been called — no-op in that case.
    """
    global _history
    if not _started:
        return
    with _lock:
        _history.clear()
        for q in _subscribers:
            q.put(json.dumps({"type": "clear", "data": "", "ts": ""}))


def save_log(path: str = None) -> str:
    """Write the full monitor log to a text file.

    Each line is formatted as '[HH:MM:SS] text'.  HTML display blocks are
    tag-stripped and each non-blank line is prefixed with '[display]'.
    Figures are not included.
    Typically called at the end of the pipeline run cell to archive the log
    alongside the other experiment output files.

    Args:
        path: Destination file path.  If None, writes to monitor_log_<timestamp>.txt
              in the current working directory.

    Returns:
        Absolute path of the written file.
    """
    if not _started:
        return path or ""
    if path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"monitor_log_{ts}.txt"
    path = os.path.abspath(path)
    lines = _log_lines()
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[monitor] Log saved → {path}  ({len(lines)} lines)")
    return path


def log_figure(fig=None) -> None:
    """Send a matplotlib figure to the monitor dashboard as a PNG image.

    Call this instead of plt.show(), or pass a Figure object explicitly.
    If fig is None, uses plt.gcf().  Once start_monitor() has been called,
    plt.show() is automatically forwarded here — no manual calls needed.
    """
    try:
        import matplotlib.pyplot as plt
        from io import BytesIO
    except ImportError:
        print("[monitor] matplotlib not available — cannot send figure.")
        return
    if fig is None:
        fig = plt.gcf()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode()
    _publish("figure", b64)


def start_monitor(port: int = 5007) -> str:
    """Start the LASSI monitoring dashboard.

    Redirects all stdout / stderr output and IPython display(HTML/Markdown)
    calls to a live browser dashboard served at http://localhost:<port>.
    Also patches plt.show() so figures appear automatically in the sidebar.

    Safe to call multiple times — no-op after the first call.

    Args:
        port: TCP port for the dashboard server (default 5007).

    Returns:
        Dashboard URL string.
    """
    global _original_stdout, _original_stderr, _started, _suppress_display
    url = f"http://localhost:{port}"

    if _started:
        try:
            from IPython.display import display, HTML
            display(HTML(
                f"<b>[monitor]</b> Already running at "
                f"<a href='{url}' target='_blank'>{url}</a>"
            ))
        except ImportError:
            print(f"[monitor] Already running at {url}", file=_original_stdout)
        return url

    # ── background HTTP server ────────────────────────────────────────────────
    server = ThreadingHTTPServer(("", port), _Handler)
    t = threading.Thread(target=server.serve_forever,
                         name="lassi-monitor", daemon=True)
    t.start()

    # ── tee stdout / stderr ───────────────────────────────────────────────────
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    sys.stdout = _TeeStream(sys.stdout)
    sys.stderr = _TeeStream(sys.stderr)

    # ── patch plt.show() so figures go to the dashboard automatically ─────────
    try:
        import matplotlib.pyplot as plt
        _orig_show = plt.show
        def _patched_show(*args, **kwargs):
            log_figure(plt.gcf())
            _orig_show(*args, **kwargs)
        plt.show = _patched_show
    except ImportError:
        pass

    _started = True

    # ── display clickable link — suppressed so it doesn't capture itself ───────
    _suppress_display = True
    try:
        from IPython.display import display, HTML
        display(HTML(
            f'<div style="padding:8px 14px;background:#252526;border-radius:4px;'
            f'font-family:monospace;border:1px solid #3c3c3c;display:inline-block">'
            f'<b style="color:#9cdcfe">LASSI Monitor</b> started &rarr; '
            f'<a href="{url}" target="_blank" style="color:#4ec9b0">{url}</a>'
            f'&nbsp;<span style="color:#777;font-size:11px">'
            f'&mdash; open in a new tab and keep visible while the pipeline runs'
            f'</span></div>'
        ))
    except ImportError:
        print(f"[monitor] Dashboard at {url}", file=_original_stdout)
    finally:
        _suppress_display = False

    # ── hook IPython display publisher (after showing the link) ───────────────
    _patch_display_pub()

    return url
