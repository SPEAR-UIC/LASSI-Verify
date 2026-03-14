"""
lassi/monitor.py — Live browser dashboard for monitoring the LASSI-Verify pipeline.

Usage (add to notebook imports cell, then run once):
    from lassi.monitor import start_monitor, log_figure
    start_monitor()   # redirects stdout/stderr; displays a clickable URL

Open the URL in a new browser tab and keep it visible while the pipeline runs.
The dashboard streams all print() output in real-time with timestamps.

For matplotlib figures, plt.show() is automatically forwarded to the dashboard
after start_monitor() is called. You can also call log_figure() explicitly:
    log_figure()          # sends plt.gcf() to the dashboard
    log_figure(my_fig)    # sends a specific Figure

Works on Linux GUI, macOS, Windows, and over SSH tunnels (forward port 5007).
No external dependencies — uses only the Python standard library.
"""

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
_lock = threading.Lock()
_subscribers: list = []    # list[queue.Queue] — one per open browser connection
_history: list = []        # all SSE messages (replayed to late-joining connections)
_original_stdout = None
_original_stderr = None
_started = False


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
#log {
  flex: 1; overflow-y: auto; padding: 8px 14px;
  white-space: pre-wrap; word-break: break-word; line-height: 1.55;
}
.ts { color: #6a9955; user-select: none; margin-right: 6px; font-size: 11px; }
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

function toggleAS() {
  autoScroll = !autoScroll;
  const b = document.getElementById('asbtn');
  b.textContent = 'Auto-scroll';
  b.className   = autoScroll ? 'btn on' : 'btn';
}
function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function addLog(ts, text) {
  const node = document.createElement('span');
  node.innerHTML = '<span class="ts">' + ts + '</span>' + esc(text) + '\n';
  log.appendChild(node);
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
  if (e.type === 'log')    addLog(e.ts, e.data);
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

        else:
            self.send_error(404)

    def log_message(self, *_):   # silence per-request HTTP logs
        pass


# ── public API ──────────────────────────────────────────────────────────────────

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

    Redirects all stdout / stderr output to a live browser dashboard served at
    http://localhost:<port>.  Also patches plt.show() so figures appear
    automatically in the dashboard sidebar.

    Safe to call multiple times — no-op after the first call.

    Args:
        port: TCP port for the dashboard server (default 5007).

    Returns:
        Dashboard URL string.
    """
    global _original_stdout, _original_stderr, _started
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

    # ── display clickable link in notebook ────────────────────────────────────
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

    return url
