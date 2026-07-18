import tempfile
import os
import json
import subprocess
import html
import re
import time
from IPython.display import display, HTML

# curl exit codes that are transient network errors worth retrying.
# Includes DNS / connect / SSL / send-recv / HTTP-2 stream errors — anything
# where the request itself didn't reach a meaningful application response and
# a retry has a real chance of succeeding. Non-transient codes (e.g. 22 = HTTP
# 4xx/5xx with --fail, 47 = redirect loop, 60 = SSL cert problem, 67 = login
# denied) intentionally fall through to a raise so real bugs aren't masked.
#    6 = couldn't resolve host (DNS blip)
#    7 = failed to connect (network unreachable / refused)
#   18 = partial file (server closed connection mid-stream)
#   28 = operation timed out
#   35 = SSL connect error
#   52 = empty reply from server
#   55 = send failure
#   56 = recv failure (connection reset)
#   92 = HTTP/2 stream error
# A NEGATIVE returncode means the curl process was killed by a signal (e.g. -9
# from the OS OOM killer). Treat that as transient too — it can recover on retry
# once memory pressure subsides.
_CURL_RETRYABLE_CODES = {6, 7, 18, 28, 35, 52, 55, 56, 92}
_CURL_MAX_RETRIES = 5
_CURL_RETRY_BASE_DELAY_S = 5  # exponential backoff: 5, 10, 20, 40, 80 seconds
_CURL_RETRY_MAX_DELAY_S = 120

# Gateway/reverse-proxy default error-page bodies. curl exits 0 for these (the HTTP
# transport succeeded — only the upstream application failed), so they slip past the
# _CURL_RETRYABLE_CODES check above and would otherwise be treated as a real LLM
# response (fed to the coder as "generated code" or to the judge as a verdict).
# Matched by exact, case-insensitive equality on the stripped full body — not
# substring — so a legitimate short LLM response (e.g. "VALID") can never collide.
_API_ERROR_BODIES = {
    "internal server error",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "",
}

def _curl_backoff_seconds(attempt):
    """Exponential backoff with a cap. attempt is 1-based (1 = first retry)."""
    return min(_CURL_RETRY_BASE_DELAY_S * (2 ** (attempt - 1)),
               _CURL_RETRY_MAX_DELAY_S)

def argo_fetch_with_curl(data):

    local = True

    if local:
        host_url = "apps.inside.anl.gov"
    else:
        host_url = "localhost:5000"

    url = f"https://{host_url}/argoapi/api/v1/resource/streamchat/"

    headers = [
        "-H", "Content-Type: application/json",
        "-H", "Accept-Encoding: identity",
        "-H", "Accept: text/plain",
        "-H", "Cache-Control: no-cache",
        "-H", "Host: apps.inside.anl.gov"
    ]

    # Check if payload is large (> 50KB as a safety threshold)
    use_file = len(data) > 50000

    temp_file = None  # track temp file path for cleanup
    if use_file:
        # Write payload to temporary file for large requests
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(data)
            temp_file = f.name

        # Use @filename to reference the file
        data_arg = ["-d", f"@{temp_file}"]
        print(f"Large payload ({len(data)} bytes) - using temp file")
    else:
        # Small payload - pass directly
        data_arg = ["-d", data]

    # Live-streaming HTML widget (styled <pre> box with background).
    # Commented out to avoid duplicate display alongside Markdown output in the notebook.
    # Uncomment to re-enable real-time streaming visualization.
    # html_display = display(HTML(""), display_id=True)

    try:
        last_exception = None
        for _attempt in range(1, _CURL_MAX_RETRIES + 1):
            if _attempt > 1:
                _delay = _curl_backoff_seconds(_attempt - 1)
                print(f"\n[argo_fetch_with_curl] Retry {_attempt - 1}/{_CURL_MAX_RETRIES - 1} "
                      f"after {_delay}s (exponential backoff) ...")
                time.sleep(_delay)

            # Each attempt rebuilds response_text from scratch so a partial
            # accumulation from a failed attempt cannot compound memory pressure.
            response_text = ""
            process = None

            try:
                # Run curl as a subprocess, capturing output as it streams
                process = subprocess.Popen(
                    # ["curl", "-k", "-X", "POST", url, *headers, "-d", data],
                    ["curl", "-k", "-X", "POST", url, *headers, *data_arg],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1  # Line-buffered for real-time output
                )

                # Read output line by line to maintain real-time streaming
                for line in iter(process.stdout.readline, ''):
                    # Append new content and escape HTML characters for safety
                    response_text += line
                    # escaped_text = html.escape(response_text)

                    # # (Optional) If you want to add specific styling to tags, e.g. <think> tags:
                    # styled_text = re.sub(
                    #     r'&lt;think&gt;',
                    #     '<span style="color: #d32f2f; font-weight: bold;">&lt;think&gt;</span>',
                    #     escaped_text
                    # )
                    # styled_text = re.sub(
                    #     r'&lt;/think&gt;',
                    #     '<span style="color: #d32f2f; font-weight: bold;">&lt;/think&gt;</span>',
                    #     styled_text
                    # )

                    # # Create typewriter effect with a blinking cursor
                    # display_text = f"{styled_text}<span style='border-right:2px solid;animation:blink 1s infinite'> </span>"

                    # # Update the display output with styled text
                    # # Use dedent to remove any leading whitespace from the HTML template
                    # # Put everything on one line to avoid extra whitespace
                    # html_content = (
                    #     f"<pre class='lassi-out' style='padding:10px;border-radius:5px;"
                    #     f"white-space:pre-wrap;font-family:monospace;margin:0;'>{display_text}</pre>"
                    #     f"<style>"
                    #     f".lassi-out{{background:#f0f0f0;color:#1e1e1e;border:1px solid #ccc;}}"
                    #     f"@media(prefers-color-scheme:dark){{.lassi-out{{background:#1e1e1e;color:#d4d4d4;border:1px solid #444;}}}}"
                    #     f"@keyframes blink {{0%{{opacity:1}}50%{{opacity:0}}100%{{opacity:1}}}}"
                    #     f"</style>"
                    # )

                    # # Update the display output with styled text
                    # html_display.update(HTML(html_content))

                    # print(line, end="", flush=True)  # Print each chunk live
                    #output += line  # Capture full response

                process.stdout.close()
                process.wait()  # Ensure process exits cleanly

            except (MemoryError, OSError) as _io_exc:
                # MemoryError: Python ran out of memory accumulating response_text
                # (very large LLM response) or subprocess buffers.
                # OSError: subprocess pipe broken, file descriptor exhaustion,
                # interrupted system call, etc. — usually transient.
                # Free what we can before retrying so the next attempt has headroom.
                response_text = ""
                try:
                    if process is not None:
                        if process.stdout:
                            process.stdout.close()
                        process.kill()
                        process.wait(timeout=5)
                except Exception:
                    pass
                err_msg = (f"Python-side {type(_io_exc).__name__} during curl read: "
                           f"{_io_exc}")
                if _attempt < _CURL_MAX_RETRIES:
                    print(f"\n[argo_fetch_with_curl] {err_msg} "
                          f"on attempt {_attempt}/{_CURL_MAX_RETRIES} — will retry.")
                    last_exception = Exception(err_msg)
                    continue
                raise Exception(err_msg) from _io_exc

            # Check for errors. Treat negative returncode (signal kill, e.g. -9 from
            # the OS OOM killer) and the curated retryable code set the same way.
            if process.returncode != 0:
                try:
                    stderr_output = process.stderr.read()
                except Exception:
                    stderr_output = "(stderr unavailable)"
                err_msg = f"CURL Error (code {process.returncode}): {stderr_output}"
                _is_signal_kill = process.returncode < 0
                _is_retryable_code = process.returncode in _CURL_RETRYABLE_CODES
                if (_is_signal_kill or _is_retryable_code) and _attempt < _CURL_MAX_RETRIES:
                    _kind = ("signal kill" if _is_signal_kill
                             else f"transient curl code {process.returncode}")
                    print(f"\n[argo_fetch_with_curl] {_kind} on attempt "
                          f"{_attempt}/{_CURL_MAX_RETRIES} — will retry.")
                    last_exception = Exception(err_msg)
                    continue  # retry
                raise Exception(err_msg)

            # curl succeeded (transport layer OK) but the upstream application may
            # still have failed, returning a gateway/proxy error page as the body
            # instead of a real LLM response. Treat that the same as a transient
            # transport failure: retry with the same backoff schedule.
            if response_text.strip().lower() in _API_ERROR_BODIES:
                err_msg = f"Argo gateway returned an API-error body: '{response_text.strip()}'"
                if _attempt < _CURL_MAX_RETRIES:
                    print(f"\n[argo_fetch_with_curl] API-error body ('{response_text.strip()}') "
                          f"on attempt {_attempt}/{_CURL_MAX_RETRIES} — will retry.")
                    last_exception = Exception(err_msg)
                    continue  # retry
                raise Exception(err_msg)

            return response_text  # Store full response for later use

        # All retries exhausted
        raise last_exception

    finally:
        # Clean up temp file if we created one
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)
