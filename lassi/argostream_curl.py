import tempfile
import os
import json
import subprocess
import html
import re
import time
from IPython.display import display, HTML

# curl exit codes that are transient network errors worth retrying:
#   18 = partial file (server closed connection mid-stream)
#   28 = operation timed out
#   35 = SSL connect error
#   56 = recv failure (connection reset)
_CURL_RETRYABLE_CODES = {18, 28, 35, 56}
_CURL_MAX_RETRIES = 3
_CURL_RETRY_DELAY_S = 5  # seconds between attempts

def argo_fetch_with_curl(data):

    local = True

    if local:
        host_url = "apps-dev.inside.anl.gov"
    else:
        host_url = "localhost:5000"

    url = f"https://{host_url}/argoapi/api/v1/resource/streamchat/"

    headers = [
        "-H", "Content-Type: application/json",
        "-H", "Accept-Encoding: identity",
        "-H", "Accept: text/plain",
        "-H", "Cache-Control: no-cache",
        "-H", "Host: apps-dev.inside.anl.gov"
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
                print(f"\n[argo_fetch_with_curl] Retry {_attempt - 1}/{_CURL_MAX_RETRIES - 1} "
                      f"after {_CURL_RETRY_DELAY_S}s ...")
                time.sleep(_CURL_RETRY_DELAY_S)

            # Run curl as a subprocess, capturing output as it streams
            process = subprocess.Popen(
                # ["curl", "-k", "-X", "POST", url, *headers, "-d", data],
                ["curl", "-k", "-X", "POST", url, *headers, *data_arg],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line-buffered for real-time output
            )

            response_text = ""

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

            # Check for errors
            if process.returncode != 0:
                stderr_output = process.stderr.read()
                err_msg = f"CURL Error (code {process.returncode}): {stderr_output}"
                if process.returncode in _CURL_RETRYABLE_CODES and _attempt < _CURL_MAX_RETRIES:
                    print(f"\n[argo_fetch_with_curl] Transient curl error (code {process.returncode}) "
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
