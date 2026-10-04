"""
lassiapp.py — LASSI-Verify Experiment Launcher GUI

Provides a browser-based form to configure and launch LASSI-Verify experiments
without editing config.yaml or the notebook directly.

Usage:
    streamlit run lassiapp.py

Opens at http://localhost:8501
Monitor dashboard runs separately at http://localhost:5007
"""

import base64
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st
import yaml

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LASSI-Verify Launcher",
    page_icon="lassi/lassi-verify-logo.png",
    layout="wide",
)

# ── Load source_apps.json for dynamic dropdowns ───────────────────────────────
@st.cache_data
def load_source_apps():
    with open("source_apps.json", "r") as f:
        return json.load(f)

source_apps = load_source_apps()

HECBENCH_APPS = [k for k in source_apps["HeCBench"].keys() if k != "source_folder"]
MINIAPPS      = [k for k in source_apps["Miniapps"].keys()  if k != "source_folder"]

# LLM option lists
ARGO_MODELS     = ["gpt54", "gpt5", "gpt41", "gpt4olatest", "gpto4mini", "gpt4turbo", "gpt4o",
                   "claudesonnet46", "claudesonnet4"]
CEREBRAS_MODELS = ["gpt-oss-120b", "qwen-3-coder-480b"]
OLLAMA_MODELS   = ["devstral:24b", "qwq:latest", "deepseek-r1:70b", "qwen2.5-coder:32b",
                   "qwen3:latest", "qwen3-coder:30b", "codestral:22b", "llama3.3:latest",
                   "deepseek-coder-v2:16b", "phi4", "granite3.2:8b-instruct-q8_0"]
ALL_MODELS      = ARGO_MODELS + CEREBRAS_MODELS + OLLAMA_MODELS

# Load current config.yaml as defaults
def load_config(path="config.yaml"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}

cfg = load_config()

# ── Helper to get default values from config ─────────────────────────────────
def _get(keys, default=None):
    """Drill into nested cfg dict via dot-path list."""
    val = cfg
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val

# ── Header: logo + title ──────────────────────────────────────────────────────
_logo_b64 = base64.b64encode(Path("lassi/lassi-verify-logo.png").read_bytes()).decode()
st.markdown(f"""
<div style="display:flex; align-items:center; gap:18px; margin-bottom:4px;">
  <img src="data:image/png;base64,{_logo_b64}" width="100" style="flex-shrink:0; border-radius:6px;"/>
  <div>
    <h1 style="margin:0; padding:0; font-size:2.2rem;">LASSI-Verify Experiment Launcher</h1>
    <p style="margin:4px 0 0 0; color:gray; font-size:0.95rem;">
      Configure and launch the LASSI-Verify pipeline.
      Settings are saved to <code>config.yaml</code> then <code>run_lassi.py</code> is started.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Layout: two columns top, then verification + pipeline ─────────────────────
col_app, col_task, col_llm = st.columns([1, 1, 1.4])

# ── Column 1: App Selection ───────────────────────────────────────────────────
with col_app:
    st.subheader("App")
    collection = st.selectbox(
        "Collection",
        ["HeCBench", "Miniapps"],
        index=0 if _get(["app", "collection"], "HeCBench") == "HeCBench" else 1,
    )
    if collection == "HeCBench":
        app_list    = HECBENCH_APPS
        make_script = False
        build_script = False
    else:
        app_list     = MINIAPPS
        make_script  = True
        build_script = False

    default_app = _get(["app", "name"], "jacobi")
    app_name = st.selectbox(
        "Application",
        app_list,
        index=app_list.index(default_app) if default_app in app_list else 0,
    )

    # Show execution parameters for selected app (read-only info)
    app_params = source_apps[collection].get(app_name, {}).get("app_params", [""])
    code_lines = source_apps[collection].get(app_name, {}).get("code_lines", "?")
    st.caption(f"Lines of code: **{code_lines}**")
    st.caption(f"Exec params: `{app_params}`")

    st.subheader("Language")
    source_lang = st.selectbox("Source", ["CUDA", "OMP", "HIP", "SYCL"],
                               index=["CUDA", "OMP", "HIP", "SYCL"].index(
                                   _get(["language", "source"], "CUDA")))
    target_lang = st.selectbox("Target", ["CUDA", "OMP", "HIP", "SYCL"],
                               index=["CUDA", "OMP", "HIP", "SYCL"].index(
                                   _get(["language", "target"], "CUDA")))

# ── Column 2: Task Mode ───────────────────────────────────────────────────────
with col_task:
    st.subheader("Task Mode")
    _power  = _get(["task", "power_measure"],    False)
    _feedback = _get(["task", "yes_power_feedback"], True)
    if _power and _feedback:
        _default_mode = "Energy Efficiency (EE)"
    elif not _power and _feedback:
        _default_mode = "Runtime Performance"
    else:
        _default_mode = "Translation (one-shot)"

    task_mode = st.radio(
        "Mode",
        ["Energy Efficiency (EE)", "Runtime Performance", "Translation (one-shot)"],
        index=["Energy Efficiency (EE)", "Runtime Performance", "Translation (one-shot)"].index(_default_mode),
        help="EE: optimise energy. Runtime: optimise wall-clock time. Translation: one-shot, no feedback loop.",
    )
    if task_mode == "Energy Efficiency (EE)":
        power_measure     = True
        yes_power_feedback = True
    elif task_mode == "Runtime Performance":
        power_measure     = False
        yes_power_feedback = True
    else:
        power_measure     = False
        yes_power_feedback = False
        st.warning("⚠️ Translation mode has a known infinite-loop-on-success issue.")

    st.subheader("Experiment")
    trial_number = st.number_input("Starting trial #", min_value=1, value=int(_get(["experiment", "trial_number"], 1)))
    total_trials = st.number_input("Total trials",     min_value=1, value=int(_get(["experiment", "total_trials"], 1)))
    exp_label    = st.text_input("Experiment label (optional)",
                                 value=_get(["experiment", "label"], ""),
                                 help="Leave blank to auto-generate from model + app + date.")

# ── Column 3: LLM Selection ───────────────────────────────────────────────────
with col_llm:
    st.subheader("LLM")

    def model_selectbox(label, cfg_key, default):
        cur = _get(["llm", cfg_key], default)
        options = ALL_MODELS
        # If current value not in list (e.g. custom Ollama tag) prepend it
        if cur not in options:
            options = [cur] + options
        return st.selectbox(label, options, index=options.index(cur))

    model       = model_selectbox("Model (general)",  "model",       "gpt54")
    model_coder = model_selectbox("Model (coder)",     "model_coder", "gpt54")
    model_judge = model_selectbox("Model (judge)",     "model_judge", "gpt41")

    num_ctx       = st.number_input("Context tokens (model)",  min_value=4096, max_value=1000000,
                                    value=int(_get(["llm", "num_ctx"], 200000)), step=4096)
    num_ctx_coder = st.number_input("Context tokens (coder)", min_value=4096, max_value=1000000,
                                    value=int(_get(["llm", "num_ctx_coder"], 200000)), step=4096)

    col_a, col_b = st.columns(2)
    with col_a:
        argo_uri = st.selectbox("Argo endpoint", ["local", "remote"],
                                index=0 if _get(["llm", "argo_uri"], "local") == "local" else 1)
    with col_b:
        argo_username = st.text_input("Argo username", value=_get(["llm", "argo_username"], "enter-your-Argonne-domain-username"))

st.divider()

# ── Second row: Verification + Pipeline options ───────────────────────────────
col_ver, col_pipe = st.columns(2)

with col_ver:
    st.subheader("LASSI-Verify Tiers")
    run_tier2       = st.checkbox("Enable Tier 2 (empirical equivalence)",
                                  value=bool(_get(["verification", "run_tier2"], True)))
    tier2_tolerance = st.number_input("Tier 2 tolerance",
                                      value=float(_get(["verification", "tier2_tolerance"], 1e-6)),
                                      format="%.2e", step=1e-7, disabled=not run_tier2)
    run_tier3       = st.checkbox("Enable Tier 3 (metamorphic testing)",
                                  value=bool(_get(["verification", "run_tier3"], True)))
    tier3_tolerance = st.number_input("Tier 3 tolerance",
                                      value=float(_get(["verification", "tier3_tolerance"], 1e-5)),
                                      format="%.2e", step=1e-6, disabled=not run_tier3)
    run_llm_judge   = st.checkbox("Enable LLM judge",
                                  value=bool(_get(["verification", "run_llm_judge"], True)),
                                  help="Disable for pure three-tier mode (Tier 1+2+3 only).")
    run_code_analysis = st.checkbox("Enable code analysis (pre-Tier-1 rubric)",
                                    value=bool(_get(["verification", "run_code_analysis"], False)))

with col_pipe:
    st.subheader("Pipeline Options")
    vary_llm_temp = st.checkbox("Vary LLM temperature across iterations",
                                value=bool(_get(["pipeline", "vary_llm_temp"], True)))
    verifyprompt  = st.checkbox("Print LLM prompts to stdout",
                                value=bool(_get(["pipeline", "verifyprompt"], True)))
    no_monitor    = st.checkbox("Skip monitor dashboard (--no-monitor)",
                                value=False,
                                help="Monitor runs at http://localhost:5007 — uncheck to launch it.")

st.divider()

# ── Config preview ────────────────────────────────────────────────────────────
new_cfg = {
    "app": {
        "collection":   collection,
        "name":         app_name,
        "make_script":  make_script,
        "build_script": build_script,
    },
    "language": {"source": source_lang, "target": target_lang},
    "task": {
        "power_measure":     power_measure,
        "yes_power_feedback": yes_power_feedback,
    },
    "llm": {
        "model":          model,
        "model_coder":    model_coder,
        "model_judge":    model_judge,
        "num_ctx":        int(num_ctx),
        "num_ctx_coder":  int(num_ctx_coder),
        "argo_uri":       argo_uri,
        "argo_username":  argo_username,
    },
    "verification": {
        "run_tier2":          run_tier2,
        "tier2_tolerance":    float(tier2_tolerance),
        "run_tier3":          run_tier3,
        "tier3_tolerance":    float(tier3_tolerance),
        "run_llm_judge":      run_llm_judge,
        "run_code_analysis":  run_code_analysis,
    },
    "pipeline": {
        "vary_llm_temp": vary_llm_temp,
        "verifyprompt":  verifyprompt,
    },
    "experiment": {
        "trial_number": int(trial_number),
        "total_trials": int(total_trials),
        "label":        exp_label,
    },
}

with st.expander("Preview config.yaml that will be written", expanded=False):
    st.code(yaml.dump(new_cfg, default_flow_style=False, sort_keys=False), language="yaml")

# ── Launch row ────────────────────────────────────────────────────────────────
if "pipeline_pid" not in st.session_state:
    st.session_state.pipeline_pid = None

col_launch, col_monitor, col_stop, col_status = st.columns([1, 1, 0.6, 2])

with col_launch:
    launch = st.button("▶  Save Config & Launch", type="primary", use_container_width=True)

with col_monitor:
    if st.session_state.pipeline_pid is not None and not no_monitor:
        st.link_button("Open Monitor Dashboard ↗", "http://localhost:5007", use_container_width=True)

with col_stop:
    if st.session_state.pipeline_pid is not None:
        stop = st.button("⏹  Stop", use_container_width=True)
        if stop:
            try:
                os.kill(st.session_state.pipeline_pid, signal.SIGTERM)
                st.session_state.pipeline_pid = None
                st.rerun()
            except ProcessLookupError:
                # Process already finished on its own
                st.session_state.pipeline_pid = None
                st.rerun()
            except PermissionError:
                st.error("Permission denied — cannot stop that process.")

with col_status:
    if st.session_state.pipeline_pid is not None:
        st.success(f"**Experiment running** — PID {st.session_state.pipeline_pid}", icon="✅")

if launch:
    # Write config
    with open("config.yaml", "w") as f:
        f.write("# ── LASSI-Verify Experiment Configuration (written by lassiapp.py) ──\n\n")
        yaml.dump(new_cfg, f, default_flow_style=False, sort_keys=False)

    # Build command
    cmd = [sys.executable, "run_lassi.py", "--config", "config.yaml"]
    if no_monitor:
        cmd.append("--no-monitor")

    st.info(f"Launching: `{' '.join(cmd)}`")

    try:
        # Do NOT pipe stdout — the monitor's start_monitor() patches sys.stdout
        # internally to stream output to the web dashboard.  Piping would
        # intercept that stream and block the subprocess when the pipe buffer fills.
        # Raw output goes to the terminal where `streamlit run` was launched.
        proc = subprocess.Popen(cmd)
        # Give the monitor server a moment to bind its port
        time.sleep(2)
        if proc.poll() is None:
            st.session_state.pipeline_pid = proc.pid
            st.rerun()
        else:
            st.session_state.pipeline_pid = None
            st.error(f"Pipeline exited immediately (code {proc.returncode}). "
                     "Check the terminal where `streamlit run` was launched for errors.")
    except Exception as e:
        st.error(f"Failed to start pipeline: {e}")

st.divider()
st.caption("LASSI-Verify Experiment Launcher · Monitor: http://localhost:5007 · GUI: http://localhost:8501")
