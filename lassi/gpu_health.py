"""
GPU Health Module for LASSI-Verify

Lightweight pre-trial / per-version GPU health snapshots. Designed to catch
transient GPU/driver states that can silently corrupt empirical verification
results — the kind of failure we saw in KERNEL_TEST5_CHACHA_2 (06-20-2026)
where the same binary produced wrong output during the trial but reproduces
correctly afterward.

Two operating modes:
  - run_compute_test=True (used at pipeline start): full snapshot including
    a known-answer CUDA/HIP compute probe that verifies the GPU is currently
    producing arithmetically correct output. Compiled-once, cached binary in
    /tmp; subsequent calls just re-execute it.
  - run_compute_test=False (used at per-version snapshots): vendor smi query
    only — temperature, throttling reasons, ECC counters, recent kernel-fault
    events from dmesg. Sub-second cost; safe to call repeatedly.

Public API:
  snapshot(label, vendor=None, run_compute_test=False) -> dict

The returned dict has a 'health_verdict' field with one of:
  'OK'                  — no anomalies detected
  'WARN: <reason>'      — anomaly detected but trial can proceed
  'FAIL: <reason>'      — anomaly serious enough to abort (caller decides)
  'UNKNOWN: <reason>'   — could not determine health (smi missing, parse fail);
                          caller should not gate on this

The module is intentionally non-gating: snapshot() never raises and never
aborts. The caller (pipeline_v4.py) decides what to do with the verdict.
"""

import os
import re
import subprocess
import tempfile
import time
from datetime import datetime


# Cache the compiled smoke-test binary path between calls so we only pay
# the nvcc/hipcc cost on the first invocation per process. Cached under
# /tmp so it survives within a pipeline run but is wiped on reboot — which
# is correct, because a reboot may change driver/CUDA versions and we want
# a fresh compile if that happens.
_SMOKE_BIN_CACHE = {}  # vendor -> path

# nvidia-smi throttle reasons that warrant a WARN. Source: nvidia-smi --help-query-gpu
# Bits 0x1 (GPU_IDLE) and 0x2 (APPLICATIONS_CLOCKS_SETTING) are benign; everything
# else (HW_THERMAL, HW_POWER_BRAKE, SW_THERMAL_SLOWDOWN, SYNC_BOOST, etc.) is worth
# flagging. We mask off the benign bits before deciding whether to warn.
_THROTTLE_BENIGN_MASK = 0x3  # GPU_IDLE | APPLICATIONS_CLOCKS_SETTING

# Recent Xid events within this many seconds count as "fresh" for the
# recent_xid_count field. 5 minutes catches anything that hit during the
# current pipeline run or just before it.
_DMESG_RECENT_WINDOW_S = 300


def _which(cmd):
    """Locate executable; return None if not on PATH."""
    try:
        return subprocess.check_output(['which', cmd],
                                       stderr=subprocess.DEVNULL,
                                       text=True).strip() or None
    except subprocess.CalledProcessError:
        return None


def _run(cmd, timeout=10):
    """Run a subprocess; return (returncode, stdout, stderr). Never raises."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return -1, '', f'{type(e).__name__}: {e}'


def _detect_vendor():
    """Auto-detect GPU vendor; returns 'NVIDIA', 'AMD', or 'NONE'."""
    if _which('nvidia-smi'):
        rc, out, _ = _run(['nvidia-smi', '-L'], timeout=5)
        if rc == 0 and 'GPU' in out:
            return 'NVIDIA'
    if _which('rocm-smi'):
        rc, out, _ = _run(['rocm-smi', '--showid'], timeout=5)
        if rc == 0 and 'GPU' in out:
            return 'AMD'
    return 'NONE'


def _parse_int(s, default=None):
    """Parse leading int from a string like '42 C' or '0x0000000000000000'."""
    s = (s or '').strip()
    if not s or s.lower() in ('n/a', 'na', '[n/a]'):
        return default
    # Hex throttle bitmasks come through as '0x0000000000000000'
    if s.startswith('0x'):
        try:
            return int(s, 16)
        except ValueError:
            return default
    m = re.match(r'-?\d+', s)
    if m:
        try:
            return int(m.group(0))
        except ValueError:
            return default
    return default


def _query_nvidia():
    """Query NVIDIA GPU via nvidia-smi. Returns dict with vendor-agnostic fields."""
    # Query the smi-supported fields in one call for atomicity and speed.
    # If a field is not supported on this GPU/driver, nvidia-smi returns [N/A]
    # for that column — handled by _parse_int's default=None.
    query_fields = [
        'temperature.gpu',
        'utilization.gpu',
        'utilization.memory',
        'power.draw',
        'pstate',
        'clocks_throttle_reasons.active',
        'memory.used',
        'memory.free',
        'ecc.errors.uncorrected.aggregate.total',
        'driver_version',
    ]
    cmd = ['nvidia-smi',
           '--query-gpu=' + ','.join(query_fields),
           '--format=csv,noheader,nounits']
    rc, out, err = _run(cmd, timeout=10)
    if rc != 0:
        return {'vendor': 'NVIDIA', 'query_error': err.strip() or f'rc={rc}'}
    # Take just the first GPU's row (LASSI is single-GPU on this laptop;
    # multi-GPU support would aggregate across rows here).
    first_row = out.strip().split('\n')[0].split(',')
    values = [v.strip() for v in first_row]
    if len(values) < len(query_fields):
        return {'vendor': 'NVIDIA', 'query_error': f'unexpected column count: {len(values)}'}
    snap = {
        'vendor': 'NVIDIA',
        'temperature_c': _parse_int(values[0]),
        'util_gpu_pct': _parse_int(values[1]),
        'util_mem_pct': _parse_int(values[2]),
        'power_draw_w': _parse_int(values[3]),
        'perf_state': values[4] or None,
        'throttle_reasons_hex': values[5] or None,
        'mem_used_mb': _parse_int(values[6]),
        'mem_free_mb': _parse_int(values[7]),
        'ecc_uncorrected_total': _parse_int(values[8]),  # often [N/A] on consumer GPUs
        'driver_version': values[9] or None,
    }
    return snap


def _query_amd():
    """Query AMD GPU via rocm-smi. Returns dict with vendor-agnostic fields.

    Best-effort — rocm-smi output formatting varies across ROCm versions.
    Fields that don't parse are left as None; callers should treat None
    as "unknown" rather than zero.
    """
    rc, out, err = _run(['rocm-smi',
                         '--showtemp', '--showuse', '--showpower',
                         '--showperflevel', '--showmemuse',
                         '--csv'], timeout=10)
    if rc != 0:
        return {'vendor': 'AMD', 'query_error': err.strip() or f'rc={rc}'}
    snap = {
        'vendor': 'AMD',
        'temperature_c': None, 'util_gpu_pct': None, 'util_mem_pct': None,
        'power_draw_w': None, 'perf_state': None, 'throttle_reasons_hex': None,
        'mem_used_mb': None, 'mem_free_mb': None,
        'ecc_uncorrected_total': None, 'driver_version': None,
        'amd_raw_csv': out,  # opaque dump for manual inspection
    }
    # Best-effort regex extraction. ROCm CSV columns are named like
    # 'Temperature (Sensor edge) (C)' so we match by keyword.
    for line in out.splitlines():
        m = re.search(r'(\d+(?:\.\d+)?)\s*C', line)
        if m and snap['temperature_c'] is None and 'edge' in line.lower():
            try: snap['temperature_c'] = int(float(m.group(1)))
            except ValueError: pass
        m = re.search(r'GPU use\s*\(%\)\s*[:,]\s*(\d+)', line, re.IGNORECASE)
        if m: snap['util_gpu_pct'] = int(m.group(1))
        m = re.search(r'Average Graphics Package Power\s*\(W\)\s*[:,]\s*(\d+(?:\.\d+)?)',
                      line, re.IGNORECASE)
        if m:
            try: snap['power_draw_w'] = float(m.group(1))
            except ValueError: pass
    return snap


def _query_power_profile_nvidia():
    """Read the GPU's static power/clock policy + OS-level power state.

    Captures information that does NOT vary during a trial but DOES affect
    every measurement in the trial — the configured power limits, persistence
    mode, max clock, OS power profile, and AC adapter state. Intended to be
    called once at pipeline_start so the meta-file timeline shows the policy
    environment the trial ran under. Catches cases like
    KERNEL_TEST6_RandA_1 where sustained SW_THERMAL_SLOWDOWN + SW_POWER_CAP
    throttling at 44°C points at a vendor-set TGP cap or persistence-mode
    issue rather than actual thermal stress.

    Read-only — does NOT change any system state. Returns a dict; fields that
    cannot be read are None (treated as "unknown", not "default").
    """
    profile = {
        'power_limit_w': None,
        'power_default_limit_w': None,
        'power_min_limit_w': None,
        'power_max_limit_w': None,
        'persistence_mode': None,
        'max_sm_clock_mhz': None,
        'os_power_profile': None,
        'ac_adapter_online': None,
        'battery_capacity_pct': None,
    }

    # NVIDIA power/clock profile fields. power.management is deprecated in newer
    # smi releases and returns a deprecation notice rather than an error — we
    # just leave it None if unparseable.
    smi_fields = [
        'power.limit',
        'power.default_limit',
        'power.min_limit',
        'power.max_limit',
        'persistence_mode',
        'clocks.max.sm',
    ]
    rc, out, _ = _run(['nvidia-smi',
                       '--query-gpu=' + ','.join(smi_fields),
                       '--format=csv,noheader,nounits'], timeout=10)
    if rc == 0 and out.strip():
        row = [v.strip() for v in out.strip().split('\n')[0].split(',')]
        if len(row) >= 6:
            profile['power_limit_w']         = _parse_int(row[0])  # often [N/A] on consumer GPUs
            profile['power_default_limit_w'] = _parse_int(row[1])
            profile['power_min_limit_w']     = _parse_int(row[2])
            profile['power_max_limit_w']     = _parse_int(row[3])
            profile['persistence_mode']      = row[4] or None
            profile['max_sm_clock_mhz']      = _parse_int(row[5])

    # OS-level power profile (Linux power-profiles-daemon)
    if _which('powerprofilesctl'):
        rc, out, _ = _run(['powerprofilesctl', 'get'], timeout=5)
        if rc == 0:
            profile['os_power_profile'] = out.strip() or None

    # AC adapter state — checked via sysfs (Linux). Some systems name the
    # adapter directory ADP1 instead of AC; check both common names.
    for ac_path in ('/sys/class/power_supply/AC/online',
                    '/sys/class/power_supply/ADP1/online'):
        if os.path.exists(ac_path):
            try:
                with open(ac_path) as f:
                    profile['ac_adapter_online'] = int(f.read().strip())
                break
            except (OSError, ValueError):
                pass
    # Battery state (helps interpret AC=0 readings; full battery + AC=0 still
    # gives reasonable performance for short trials).
    for bat_path in ('/sys/class/power_supply/BAT0/capacity',
                     '/sys/class/power_supply/BAT1/capacity'):
        if os.path.exists(bat_path):
            try:
                with open(bat_path) as f:
                    profile['battery_capacity_pct'] = int(f.read().strip())
                break
            except (OSError, ValueError):
                pass

    return profile


def _dmesg_xid_check():
    """Count Xid events in dmesg within the recent window.

    Xid is the NVIDIA driver fault-event code. Any Xid in the last few minutes
    is a strong "GPU is sick" signal. Common offenders:
      Xid 13 — graphics engine exception (often invalid memory access)
      Xid 31 — MMU fault (often kernel bug or hardware issue)
      Xid 79 — GPU has fallen off the bus (driver lost contact)

    On systems where dmesg requires root, this returns count=-1 (unknown).
    Caller should treat -1 as "could not check" not "no errors".
    """
    rc, out, err = _run(['dmesg', '-T'], timeout=5)
    if rc != 0:
        # dmesg without root often fails on locked-down kernels (dmesg_restrict=1)
        return {'count': -1, 'note': f'dmesg unavailable: {err.strip()[:80] or "rc=" + str(rc)}'}
    now = time.time()
    recent_xids = []
    for line in out.splitlines():
        if 'Xid' not in line:
            continue
        # dmesg -T prefixes with bracketed timestamp like "[Mon Jun 21 12:34:56 2026]"
        m = re.match(r'\[([A-Z][a-z]{2}\s+[A-Z][a-z]{2}\s+\d+\s+\d{2}:\d{2}:\d{2}\s+\d{4})\]', line)
        if not m:
            continue
        try:
            ts = time.mktime(time.strptime(m.group(1), '%a %b %d %H:%M:%S %Y'))
        except ValueError:
            continue
        if (now - ts) <= _DMESG_RECENT_WINDOW_S:
            recent_xids.append(line.strip())
    return {'count': len(recent_xids), 'lines': recent_xids[:3]}  # cap to 3 sample lines


# Known-answer CUDA smoke test. Writes deterministic values, reads them back,
# verifies on host. Returns exit code 0 if all bytes match expected, else 1.
# Compiled once per process, cached binary path. The kernel is intentionally
# simple — just a fused multiply-add — so any wrong output is a strong signal
# the GPU is misbehaving (vs a complex kernel where wrong output could be a
# numerical-precision artifact).
_NVCC_SMOKE_SRC = r'''
#include <stdio.h>
#include <stdlib.h>
__global__ void _lassi_smoke_kernel(int* d, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) d[i] = i * 7 + 3;  // deterministic per-index value
}
int main() {
    const int N = 4096;
    int* d = nullptr;
    int* h = (int*)malloc(N * sizeof(int));
    if (!h) return 2;
    cudaError_t err = cudaMalloc(&d, N * sizeof(int));
    if (err != cudaSuccess) { free(h); return 3; }
    _lassi_smoke_kernel<<<(N + 255) / 256, 256>>>(d, N);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { cudaFree(d); free(h); return 4; }
    cudaMemcpy(h, d, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);
    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        if (h[i] != i * 7 + 3) mismatches++;
    }
    free(h);
    if (mismatches > 0) {
        fprintf(stderr, "LASSI_SMOKE: %d/%d mismatches\n", mismatches, N);
        return 1;
    }
    return 0;
}
'''


def _ensure_nvidia_smoke_bin():
    """Compile (once) and return path to the NVIDIA smoke-test binary.

    Returns (path, None) on success or (None, error_str) on compile failure.
    Cached across calls within a single Python process via _SMOKE_BIN_CACHE.
    """
    if 'NVIDIA' in _SMOKE_BIN_CACHE:
        return _SMOKE_BIN_CACHE['NVIDIA'], None
    if not _which('nvcc'):
        return None, 'nvcc not found on PATH'
    tmpdir = tempfile.mkdtemp(prefix='lassi_gpu_smoke_')
    src_path = os.path.join(tmpdir, 'smoke.cu')
    bin_path = os.path.join(tmpdir, 'smoke')
    with open(src_path, 'w') as f:
        f.write(_NVCC_SMOKE_SRC)
    # Use sm_89 to match LASSI's target arch; falls back to PTX JIT on
    # other Ada/Hopper GPUs. If you migrate to a different arch, update here.
    rc, out, err = _run(['nvcc', '-std=c++14', '-arch=sm_89', '-O3',
                         src_path, '-o', bin_path], timeout=30)
    if rc != 0:
        return None, f'nvcc rc={rc}: {(err or out).strip()[:200]}'
    _SMOKE_BIN_CACHE['NVIDIA'] = bin_path
    return bin_path, None


def _run_compute_test(vendor):
    """Run the known-answer compute test for the given vendor.

    Returns dict with 'verdict' ('PASS' / 'FAIL: <reason>' / 'SKIP: <reason>')
    and 'detail' (any stderr / exit code info).
    """
    if vendor == 'NVIDIA':
        bin_path, err = _ensure_nvidia_smoke_bin()
        if not bin_path:
            return {'verdict': f'SKIP: compile failed', 'detail': err}
        rc, out, stderr = _run([bin_path], timeout=15)
        if rc == 0:
            return {'verdict': 'PASS', 'detail': None}
        return {'verdict': f'FAIL: exit={rc}',
                'detail': (stderr or out).strip()[:200] or None}
    if vendor == 'AMD':
        return {'verdict': 'SKIP: AMD compute probe not implemented',
                'detail': 'add hipcc smoke test if needed'}
    return {'verdict': 'SKIP: no GPU vendor', 'detail': None}


def _derive_verdict(snap, xid_info, compute_result):
    """Combine snapshot fields into a single health_verdict string.

    Verdict ordering (most severe wins):
      FAIL  — ECC errors, recent Xid faults, failed compute probe
      WARN  — active throttling (non-benign bits), high temp (>90C), high mem util
      OK    — none of the above
      UNKNOWN — smi query failed entirely (no signal to gate on)
    """
    if snap.get('query_error'):
        return f"UNKNOWN: smi query failed ({snap['query_error'][:60]})"
    # FAIL conditions
    ecc = snap.get('ecc_uncorrected_total')
    if isinstance(ecc, int) and ecc > 0:
        return f"FAIL: {ecc} uncorrected ECC errors"
    if xid_info.get('count', 0) > 0:
        return f"FAIL: {xid_info['count']} recent Xid event(s)"
    if compute_result and compute_result['verdict'].startswith('FAIL'):
        return f"FAIL: compute probe {compute_result['verdict']}"
    # WARN conditions
    throttle = _parse_int(snap.get('throttle_reasons_hex'))
    if isinstance(throttle, int) and (throttle & ~_THROTTLE_BENIGN_MASK):
        return f"WARN: throttled (0x{throttle:x})"
    temp = snap.get('temperature_c')
    if isinstance(temp, int) and temp > 90:
        return f"WARN: high temperature ({temp}C)"
    return 'OK'


def _format_summary_line(label, snap, xid_info, compute_result):
    """One-line human-readable summary for the monitor log."""
    parts = [f"verdict={snap['health_verdict']}"]
    if snap.get('temperature_c') is not None:
        parts.append(f"temp={snap['temperature_c']}C")
    if snap.get('util_gpu_pct') is not None:
        parts.append(f"util={snap['util_gpu_pct']}%")
    if snap.get('perf_state'):
        parts.append(f"perf={snap['perf_state']}")
    if snap.get('throttle_reasons_hex'):
        parts.append(f"throttle={snap['throttle_reasons_hex']}")
    if snap.get('mem_used_mb') is not None and snap.get('mem_free_mb') is not None:
        total = snap['mem_used_mb'] + snap['mem_free_mb']
        parts.append(f"mem={snap['mem_used_mb']}/{total}MB")
    if xid_info.get('count', -1) >= 0:
        parts.append(f"xid_recent={xid_info['count']}")
    if compute_result:
        parts.append(f"compute={compute_result['verdict']}")
    return f"[GPU HEALTH | {label}] " + " ".join(parts)


def snapshot(label, vendor=None, run_compute_test=False):
    """Capture a GPU health snapshot.

    Args:
        label: short context tag like 'pipeline_start' or 'pre_lassi_v3';
               appears in the monitor log line and the timeline.
        vendor: 'NVIDIA' / 'AMD' / None (auto-detect).
        run_compute_test: True only at pipeline start (compile-or-run cost is
            ~1-3 sec amortized). Subsequent per-version calls should leave
            this False — they get the cheap smi snapshot only.

    Returns:
        Dict with structured health data. Key fields:
            label, timestamp, vendor,
            temperature_c, util_gpu_pct, util_mem_pct, power_draw_w,
            perf_state, throttle_reasons_hex,
            mem_used_mb, mem_free_mb,
            ecc_uncorrected_total, driver_version,
            xid_recent_count, xid_recent_lines,
            compute_test_verdict (or None if not run),
            health_verdict, summary_line.

        Caller can append the dict to verification_log['gpu_health_timeline']
        and inspect health_verdict to decide whether to abort.

    Prints summary_line to stdout (so it lands in the monitor log via the
    existing print-capture wrapper). Never raises.
    """
    if vendor is None:
        vendor = _detect_vendor()
    timestamp = datetime.now().isoformat(timespec='seconds')

    if vendor == 'NVIDIA':
        snap = _query_nvidia()
    elif vendor == 'AMD':
        snap = _query_amd()
    else:
        snap = {'vendor': 'NONE', 'query_error': 'no GPU vendor detected'}

    xid_info = _dmesg_xid_check() if vendor == 'NVIDIA' else {'count': -1, 'note': 'non-NVIDIA: dmesg Xid scan skipped'}

    compute_result = _run_compute_test(vendor) if run_compute_test else None

    # Power-profile snapshot — captured at the same call sites as the compute
    # probe (pipeline_start and post_source_pre_lassi). These fields rarely
    # change during a trial but materially affect every measurement, so a
    # one-shot read at trial start is enough; per-iteration snapshots skip it.
    # Read-only — never modifies system state. Currently NVIDIA-only; AMD path
    # returns None so the field is present in the schema for both vendors.
    power_profile = None
    if run_compute_test and vendor == 'NVIDIA':
        power_profile = _query_power_profile_nvidia()

    snap['label'] = label
    snap['timestamp'] = timestamp
    snap['xid_recent_count'] = xid_info.get('count', -1)
    snap['xid_recent_lines'] = xid_info.get('lines', [])
    snap['xid_note'] = xid_info.get('note')
    snap['compute_test_verdict'] = compute_result['verdict'] if compute_result else None
    snap['compute_test_detail'] = compute_result['detail'] if compute_result else None
    snap['power_profile'] = power_profile
    snap['health_verdict'] = _derive_verdict(snap, xid_info, compute_result)
    snap['summary_line'] = _format_summary_line(label, snap, xid_info, compute_result)

    print(snap['summary_line'])
    return snap


def format_timeline(timeline):
    """Format a list of snapshot dicts as a multi-line block for the meta data file."""
    if not timeline:
        return "----- GPU HEALTH TIMELINE -----\n(no snapshots captured)\n"
    lines = ["----- GPU HEALTH TIMELINE -----"]
    for s in timeline:
        lines.append(f"  [{s.get('timestamp', '?')}] [{s.get('label', '?')}] {s.get('summary_line', '?')}")
        if s.get('xid_recent_count', 0) > 0:
            for xid_line in s.get('xid_recent_lines', [])[:2]:
                lines.append(f"      xid: {xid_line[:140]}")
        if s.get('compute_test_detail'):
            lines.append(f"      compute detail: {s['compute_test_detail'][:140]}")
        pp = s.get('power_profile')
        if pp:
            # One-line summary: shows configured limits + persistence + OS profile + AC.
            # Helps spot RandA_1-style throttling causes (e.g., power.limit < max, or
            # persistence_mode=Disabled, or os_power_profile=power-saver) post-hoc.
            lines.append(
                f"      power profile: limit={pp.get('power_limit_w')}W "
                f"default={pp.get('power_default_limit_w')}W "
                f"max={pp.get('power_max_limit_w')}W "
                f"persistence={pp.get('persistence_mode')} "
                f"max_sm_clock={pp.get('max_sm_clock_mhz')}MHz "
                f"os_profile={pp.get('os_power_profile')} "
                f"ac_online={pp.get('ac_adapter_online')} "
                f"battery={pp.get('battery_capacity_pct')}%"
            )
    return "\n".join(lines) + "\n"


if __name__ == '__main__':
    # Manual smoke test of the module itself
    print("--- Pipeline-start snapshot (with compute probe) ---")
    s1 = snapshot('manual_test_start', run_compute_test=True)
    print()
    print("--- Per-version snapshot (smi only) ---")
    s2 = snapshot('manual_test_v3')
    print()
    print(format_timeline([s1, s2]))
