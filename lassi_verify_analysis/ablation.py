"""
Oracle-anchored per-tier ablation (plan §8, "T-F").

Reconstructs, from data already on disk, what each verification-layer configuration
would have concluded -- no tiers were ever actually disabled (see the trajectory-
dependence caveat below). Two complementary views:

  View A  trajectory-level detection (all attempts, LASSI population, NOT oracle-anchored).
          Uses every compile+execute-success attempt, including self-corrected-away
          intermediate attempts whose code never survives to be oracle-graded. Answers
          "how much independent detection work did each layer do during self-correction?"

  View B  oracle-anchored final-attempt attribution. Joins `oracle_results.csv` onto the
          final-attempt rows (same join as `oracle_confusion.py`) and, for every row the
          full framework marks INVALID, attributes it to whichever layer(s) failed, split
          by the oracle: oracle FAIL -> true (unique) catch; oracle PASS -> (unique) false
          reject. This is the fix over the two prior (never-implemented) ablation designs
          in `planning/LASSI_Verify_Metrics_Specification.md` / `_Data_Collection_Schema.md`,
          which used tier-vs-tier as their own ground truth and so could not tell a real
          catch from a false reject.

Layer decomposition: Tier 2's verdict conflates output-correctness and the witness-chain
tripwire (both come from the same `compare_passthrough_outputs()` pass over PASSTHROUGH
labels). This module splits them via the fixed `witness_chain_state` label
(`lassi/verify.py:1544`) into two independent layers: `tier2_output` and `witness`.

Caveat (state in paper): this is a data-only reconstruction over a single full-pipeline
run where all tiers were active throughout. Self-correction from earlier layers means
later layers see code that earlier layers already fixed -- a production "Tier 2 only"
system would see different (potentially worse) code, since Tier 1 self-correction would
never have run. This measures detection capability on the trajectory the full pipeline
actually produced, not the behavior of a reduced-layer pipeline.

Usage:
    python -m lassi_verify_analysis.ablation --exp-dir DIR1 [DIR2 ...]
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

from .oracle_confusion import TESTABLE, _framework_verdict

WITNESS_LABEL_RE = re.compile(r"^witness_chain_state(#\d+)?$")
KERNEL_NAME_RE = re.compile(r"-cuda_main.*")

LAYERS = ["tier1", "tier2_output", "witness", "tier3"]

# Sub-configurations reported alongside the full framework (tier1+tier2_output+witness+tier3,
# computed via the canonical `_framework_verdict` so it matches oracle_confusion.py exactly).
CONFIGS = [
    ("T1", ["tier1"]),
    ("T2out", ["tier2_output"]),
    ("Witness", ["witness"]),
    ("T3", ["tier3"]),
    ("T1+T2out", ["tier1", "tier2_output"]),
    ("T1+T2out+Witness", ["tier1", "tier2_output", "witness"]),
]


def _kernel_name(app_name: str) -> str:
    """Strip the `-cuda_main...` experiment suffix, matching data_loader's `kernel_name`."""
    return KERNEL_NAME_RE.sub("", str(app_name)).strip()


def _split_labels(val) -> list[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _layer_verdicts(row) -> dict:
    """Split one attempt-log row's tier verdicts into the 4 independent layers."""
    t1 = str(row.get("tier1_verdict", "")).strip().upper() or "NA"
    t2 = str(row.get("tier2_verdict", "")).strip().upper() or "NA"
    t3 = str(row.get("tier3_verdict", "")).strip().upper() or "NA"

    if t2 == "FAIL":
        failed = _split_labels(row.get("tier2_failed_labels"))
        witness_failed = any(WITNESS_LABEL_RE.match(l) for l in failed)
        # A FAIL with no parseable labels is a data gap, not a passing witness/output --
        # default it onto tier2_output so the Framework config still matches tier2_verdict.
        output_failed = (not failed) or any(not WITNESS_LABEL_RE.match(l) for l in failed)
        tier2_output = "FAIL" if output_failed else "PASS"
        witness = "FAIL" if witness_failed else "PASS"
    else:
        # PASS / SKIP / NA carries through identically to both sub-layers.
        tier2_output = t2
        witness = t2

    return {"tier1": t1, "tier2_output": tier2_output, "witness": witness, "tier3": t3}


def _config_invalid(layers: dict, names: list[str]) -> bool:
    return any(layers[n] == "FAIL" for n in names)


def _unique_catcher(layers: dict) -> str | None:
    """Name of the sole failing layer, or None if 0 or >1 layers failed (shared catch)."""
    failing = [n for n in LAYERS if layers[n] == "FAIL"]
    return failing[0] if len(failing) == 1 else None


def _add_layer_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    layer_rows = df.apply(_layer_verdicts, axis=1)
    for name in LAYERS:
        df[f"layer_{name}"] = [lv[name] for lv in layer_rows]
    return df


# ---------------------------------------------------------------------------
# View A -- trajectory-level detection (all LASSI attempts, not oracle-anchored)
# ---------------------------------------------------------------------------

def _load_attempts(exp_dir: str) -> pd.DataFrame | None:
    al_g = glob.glob(os.path.join(exp_dir, "*attempt_log.csv"))
    if not al_g:
        return None
    return pd.read_csv(al_g[0], index_col=False)


def _view_a(df: pd.DataFrame) -> dict:
    is_zs = df["generation_type"].astype(str).str.strip().str.lower() == "zeroshot"
    ok = (df["compile_status"].astype(str).str.strip().str.lower() == "success") & (
        df["execute_status"].astype(str).str.strip().str.lower() == "success"
    )
    sub = df[~is_zs & ok]
    if sub.empty:
        return {"n": 0, "layers": {}, "unique_catch": {}, "configs": {}}

    sub = _add_layer_columns(sub)

    layers_out = {}
    for name in LAYERS:
        col = sub[f"layer_{name}"]
        reachable = col.isin(["PASS", "FAIL"])
        n_reach = int(reachable.sum())
        n_fail = int((col == "FAIL").sum())
        layers_out[name] = {
            "reachable": n_reach,
            "fail": n_fail,
            "detection_rate": (n_fail / n_reach) if n_reach else None,
        }

    unique = sub.apply(lambda r: _unique_catcher({n: r[f"layer_{n}"] for n in LAYERS}), axis=1)
    unique_catch = unique.value_counts(dropna=True).to_dict()

    configs_out = {}
    for cname, names in CONFIGS:
        inv = sub.apply(lambda r: _config_invalid({n: r[f"layer_{n}"] for n in LAYERS}, names), axis=1)
        configs_out[cname] = int(inv.sum())
    configs_out["Framework"] = int((sub.apply(_framework_verdict, axis=1) == "INVALID").sum())

    return {"n": len(sub), "layers": layers_out, "unique_catch": unique_catch, "configs": configs_out}


# ---------------------------------------------------------------------------
# View B -- oracle-anchored final-attempt attribution
# ---------------------------------------------------------------------------

def _load_view_b(exp_dir: str) -> pd.DataFrame | None:
    """Final-attempt LASSI rows joined to the independent oracle.

    Same join as `oracle_confusion._load_one` (trial_number, code_block_index==code_version)
    but keeps the full attempt-log row (tier verdicts + failed labels) instead of collapsing
    to just the framework/judge summary, since layer attribution needs the detail.
    """
    orc_p = os.path.join(exp_dir, "oracle_results.csv")
    al_g = glob.glob(os.path.join(exp_dir, "*attempt_log.csv"))
    if not os.path.exists(orc_p) or not al_g:
        return None
    orc = pd.read_csv(orc_p, index_col=False)
    al = pd.read_csv(al_g[0], index_col=False)
    is_zs = al["generation_type"].astype(str).str.strip().str.lower() == "zeroshot"
    al = al[~is_zs & (al["is_final_attempt"] == True)].copy()  # noqa: E712
    al = al.rename(columns={"code_block_index": "code_version"})
    return orc.merge(al, on=["trial_number", "code_version"], how="left", suffixes=("", "_al"))


def _view_b(df: pd.DataFrame) -> dict:
    sub = df[df["oracle_verdict"].isin(TESTABLE)]
    if sub.empty:
        return {"n": 0, "n_invalid": 0, "true_unique_catch": {}, "unique_false_reject": {}}

    sub = _add_layer_columns(sub)
    framework = sub.apply(_framework_verdict, axis=1)
    invalid = sub[framework == "INVALID"]

    true_catch: dict[str, int] = {}
    false_reject: dict[str, int] = {}
    for _, r in invalid.iterrows():
        failing = [n for n in LAYERS if r[f"layer_{n}"] == "FAIL"]
        key = "+".join(failing) if failing else "unknown"
        bucket = true_catch if r["oracle_verdict"] == "FAIL" else false_reject
        bucket[key] = bucket.get(key, 0) + 1

    return {
        "n": len(sub),
        "n_invalid": len(invalid),
        "true_unique_catch": true_catch,
        "unique_false_reject": false_reject,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_view_a(va: dict) -> None:
    print(f"  View A (trajectory, n={va['n']} compile+execute-success LASSI attempts):")
    if va["n"] == 0:
        print("    (no data)")
        return
    for name in LAYERS:
        l = va["layers"][name]
        rate = f"{l['detection_rate']:.3f}" if l["detection_rate"] is not None else "n/a"
        print(f"    {name:14s} tested={l['reachable']:4d}  fail={l['fail']:4d}  activity_rate={rate}")
    print(f"    unique-catch (exactly one layer failed): {va['unique_catch']}")
    print(f"    cumulative-config INVALID counts: {va['configs']}")


def _print_view_b(vb: dict) -> None:
    print(f"  View B (oracle-anchored, n={vb['n']} graded final attempts, {vb['n_invalid']} framework-INVALID):")
    if vb["n"] == 0:
        print("    (no data)")
        return
    print(f"    true unique/shared catch (oracle FAIL):    {vb['true_unique_catch'] or '{} (none yet -- see project_true_reject_mechanism)'}")
    print(f"    unique/shared false-reject (oracle PASS):  {vb['unique_false_reject'] or '{}'}")


def _merge_counts(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def _csv_rows(kernel: str, va: dict, vb: dict) -> list[dict]:
    rows = []
    for name in LAYERS:
        l = va["layers"].get(name, {})
        rows.append({
            "kernel": kernel, "view": "A", "name": name,
            "metric": "reachable", "value": l.get("reachable"),
        })
        rows.append({
            "kernel": kernel, "view": "A", "name": name,
            "metric": "fail", "value": l.get("fail"),
        })
    for name, count in va.get("unique_catch", {}).items():
        rows.append({"kernel": kernel, "view": "A", "name": f"unique_catch:{name}", "metric": "count", "value": count})
    for cname, count in va.get("configs", {}).items():
        rows.append({"kernel": kernel, "view": "A", "name": f"config:{cname}", "metric": "invalid_count", "value": count})
    for name, count in vb.get("true_unique_catch", {}).items():
        rows.append({"kernel": kernel, "view": "B", "name": name, "metric": "true_unique_catch", "value": count})
    for name, count in vb.get("unique_false_reject", {}).items():
        rows.append({"kernel": kernel, "view": "B", "name": name, "metric": "unique_false_reject", "value": count})
    return rows


# ---------------------------------------------------------------------------
# Plotting
#
# Color choice (2026-07-27, revised): the prior blue/dark-orange/purple/dark-red
# set put Tier 2 and Tier 3 both in dark warm brownish tones -- too close at a
# glance. Replaced with the dataviz skill's reference categorical palette
# (`references/palette.md`), slots 1-4 in their documented fixed order:
# blue / orange / aqua / yellow. That doc explicitly validates this exact
# contiguous prefix against the "adjacent pairlist (stacks, bars, lines)" --
# our chart type -- clearing every CVD/normal-vision gate in both light and
# dark modes (worst adjacent CVD dE 9.1 light / 8.4 dark; worst adjacent
# normal-vision dE 19.6 light / 19.3 dark). Not independently re-run through
# validate_palette.js here (no Node runtime in this sandbox) -- resting on the
# skill's own published numbers for this exact slot subset/order, not a fresh
# measurement. If Node is available elsewhere, confirm with:
#   node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100" --mode light
# ---------------------------------------------------------------------------

LAYER_COLORS = {
    "tier1": "#2a78d6",
    "tier2_output": "#eb6834",
    "witness": "#1baf7a",
    "tier3": "#eda100",
}
LAYER_LABELS = {
    "tier1": "Tier 1",
    "tier2_output": "Tier 2",
    "witness": "Witness",
    "tier3": "Tier 3",
}
MULTI_LAYER_COLOR = "#8a8a8a"
MULTI_LAYER_LABEL = "Multiple layers"


def _attribution_color(key: str) -> str:
    return LAYER_COLORS.get(key, MULTI_LAYER_COLOR)


def _attribution_label(key: str) -> str:
    if key in LAYER_LABELS:
        return LAYER_LABELS[key]
    if "+" in key:
        return " + ".join(LAYER_LABELS.get(k, k) for k in key.split("+"))
    return key


def _lighten(hex_color: str, amount: float) -> str:
    """Blend hex_color toward white by `amount` in [0, 1] (0 = unchanged, 1 = white).

    Used to encode the true-catch/false-reject OUTCOME as a light/dark shade of each
    tier's own hue, instead of hatching (2026-07-27, user feedback: hatch patterns read
    as too busy) -- keeps categorical hue = tier, lightness = outcome, no second texture
    channel.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _merge_witness_key(key: str) -> str:
    """Fold 'witness' into 'tier2_output' for View B's 3-tier attribution story.

    Witness is an add-on cheat-detection gate bolted onto Tier 2's output-comparison
    pass (both come from the same `compare_passthrough_outputs()` call) -- it was added
    after observing LLM loop-deletion cheating, not part of the original three-tier
    design (user framing, 2026-07-27). View A still reports it as its own layer since
    View A's job is self-correction *workload*, where the witness add-on's activity is
    itself part of the story; View B is the "do the three core tiers pull their weight"
    story, so witness's catches/false-rejects are folded into Tier 2 here.
    """
    parts = {("tier2_output" if p == "witness" else p) for p in key.split("+")}
    return "+".join(sorted(parts, key=lambda p: LAYERS.index(p) if p in LAYERS else len(LAYERS)))


def _merge_witness_into_tier2(counts: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for k, v in counts.items():
        mk = _merge_witness_key(k)
        merged[mk] = merged.get(mk, 0) + v
    return merged


def _plot_view_a_chart(per_kernel_va: dict[str, dict], ax) -> None:
    """Horizontal grouped bar chart: per-kernel detection rate for each of the 4 layers.

    One row band per kernel (oriented to match View B, so the two figures read the same
    way), each holding 4 thin horizontal bars -- one per layer. Bar length = FAIL /
    reachable among all compile+execute-success LASSI attempts (trajectory-level,
    includes self-corrected-away intermediate attempts). Annotated with the raw fail
    count, since rate alone hides small-n kernels. Vertical stacking (2026-07-27, user
    request) replaces an earlier wide grouped-column layout that grew linearly with
    kernel count; legend moved below the axes for the same reason (avoids widening the
    figure to make room for a corner legend).
    """
    import matplotlib.ticker as mticker

    kernels = [k for k, va in per_kernel_va.items() if va.get("n", 0) > 0]
    n_kernels = len(kernels)
    n_layers = len(LAYERS)
    y_positions = list(range(n_kernels - 1, -1, -1))  # first kernel at top, matches View B
    height = 0.8 / n_layers

    # Shaded backdrop per kernel row, with a thin white gap between rows, drawn first
    # (low zorder) so the bars/labels render on top of it.
    for y in y_positions:
        ax.axhspan(y - 0.45, y + 0.45, color="#ececec", zorder=0)

    max_rate = 0.0
    for i, layer in enumerate(LAYERS):
        widths, fails, testeds = [], [], []
        for k in kernels:
            l = per_kernel_va[k]["layers"].get(layer, {})
            rate = l.get("detection_rate")
            widths.append(rate if rate is not None else 0.0)
            fails.append(l.get("fail", 0))
            testeds.append(l.get("reachable", 0))
        max_rate = max(max_rate, max(widths, default=0.0))
        offsets = [y + (i - (n_layers - 1) / 2) * height for y in y_positions]
        bars = ax.barh(offsets, widths, height=height * 0.92, color=LAYER_COLORS[layer],
                        alpha=0.88, label=LAYER_LABELS[layer])
        for bar, f, n in zip(bars, fails, testeds):
            if n == 0:
                continue  # layer never ran at all for this kernel -- nothing to label
            # Show fail/tested for every bar, including 0% ones: "0/26" says 26 attempts
            # ran and none needed correction, which is itself meaningful (not "no data").
            # Numerator bolded via mathtext so the count that drives the bar length reads
            # clearly.
            ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,
                    rf"$\mathbf{{{f}}}$/{n}", ha="left", va="center", fontsize=7,
                    color="#333333")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(kernels, fontsize=9)
    # X-axis is a RATE (fail/tested), not a count -- 100% is a fixed, meaningful ceiling
    # ("every tested attempt on this layer/kernel failed") regardless of each bar's own
    # denominator. No bar in this dataset gets close to 100%, so a fixed 0-100% scale
    # wastes most of the plot; tighten to just past the observed max instead (2026-07-27,
    # user request), leaving headroom for the fail/tested text labels past each bar tip.
    xmax = max(max_rate * 1.3, 0.1)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, min_n_ticks=4))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("Self-correction activity rate (FAIL / tested attempts)", fontsize=9)
    ax.set_title("Self-Correction Activity by Layer\n"
                  "(measures self-correction workload only)",
                  fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=8,
              ncol=n_layers, frameon=False)
    ax.text(0.0, -0.16,
            "Labels = fail / tested, per bar. \"Tested\" = attempts where this layer actually ran on\n"
            "that trajectory; it shrinks for later layers because the pipeline is fail-fast (Tier 2 is\n"
            "skipped on an attempt where Tier 1 already failed, etc.), so bars are not all over the same n.",
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color="#555555", style="italic")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_view_b_chart(per_kernel_vb: dict[str, dict], ax) -> None:
    """Horizontal stacked bar: per-kernel oracle-anchored INVALID attribution, by the
    three CORE verification tiers.

    Dark, full-saturation segments = TRUE CATCH (oracle FAIL): the framework correctly
    rejected genuinely broken code. This is the GOOD outcome and is drawn at full
    strength to read as the headline result. Light/pale segments (same hue, lightened
    toward white via `_lighten`) = unique/shared FALSE REJECT (framework INVALID, oracle
    PASS: the framework threw out code that was actually correct) -- labeled bluntly as
    FALSE REJECT rather than the softer "over-strict." These are a conservatism cost we
    intend to reduce (NOT the dangerous error -- false ACCEPTs, i.e. accepting broken
    code, are 0 for the framework and by construction cannot appear here), drawn pale to
    read as "flagged, still needs work" (2026-07-27, user request: visually separate the
    good result from the one that needs future work; a first pass used hatching for this
    but the user found it too busy, so lightness now carries the outcome instead of a
    second texture channel -- hue stays exclusively the tier encoding).
    Known contributors and identified fixes are stated in the footnote / paper text.

    Witness is folded into Tier 2 here via `_merge_witness_into_tier2` (2026-07-27, user
    framing): the three tiers are LASSI-Verify's core design, and witness is an add-on
    cheat-detection gate bolted onto Tier 2's comparison pass, added later once LLM
    loop-deletion cheating was observed -- not a fourth core tier. This keeps View B's
    story focused on "do the three core tiers pull their weight." The disaggregated
    Tier-2-output-vs-Witness split is still available in View A and the raw CSV.
    """
    import matplotlib.patches as mpatches

    kernels = [k for k, vb in per_kernel_vb.items() if vb.get("n", 0) > 0]
    y_positions = list(range(len(kernels) - 1, -1, -1))  # first kernel at top

    seen_keys: list[str] = []

    for kernel, y in zip(kernels, y_positions):
        vb = per_kernel_vb[kernel]
        true_catch = _merge_witness_into_tier2(vb["true_unique_catch"])
        false_reject = _merge_witness_into_tier2(vb["unique_false_reject"])
        left = 0
        # Order keys by total count desc for readability; stable across the true/false split.
        keys = sorted(
            set(true_catch) | set(false_reject),
            key=lambda k: true_catch.get(k, 0) + false_reject.get(k, 0),
            reverse=True,
        )
        if not keys:
            ax.text(0.3, y, "0 false-reject / 0 true-reject", va="center", fontsize=7.5, color="#888888")
            continue
        for key in keys:
            color = _attribution_color(key)
            light_color = _lighten(color, 0.6)
            tc = true_catch.get(key, 0)
            fr = false_reject.get(key, 0)
            if key not in seen_keys:
                seen_keys.append(key)
            # Good result first (dark, full-strength): a real bug, correctly caught.
            if tc:
                ax.barh(y, tc, left=left, height=0.6, color=color, alpha=0.95,
                        edgecolor="white", linewidth=0.6)
                ax.text(left + tc / 2, y, str(tc), ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
                left += tc
            # Needs-more-work result second (pale, same hue): correct code rejected.
            if fr:
                ax.barh(y, fr, left=left, height=0.6, color=light_color, alpha=0.95,
                        edgecolor="white", linewidth=0.6)
                ax.text(left + fr / 2, y, str(fr), ha="center", va="center",
                        fontsize=7.5, color="#333333", fontweight="bold")
                left += fr
        ax.text(left + 0.3, y, f"n={left}", va="center", fontsize=8, color="#333333")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(kernels, fontsize=9)
    ax.set_xlabel("Framework-INVALID final attempts, attributed to layer(s)", fontsize=9)
    ax.set_title("LASSI-Verify Code Rejections by Tier\n"
                  "(dark = real bug correctly caught; pale = false reject, needs more work)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if seen_keys:
        # Order entries to match View A's canonical layer order/colors (tier1, tier2_output,
        # tier3 -- witness is folded into tier2_output here, see _merge_witness_into_tier2),
        # then any remaining multi-tier combos in first-encountered order. Legend swatches
        # are built explicitly (not taken from drawn bars) so every tier reads as a plain
        # dark color here -- the dark/pale distinction is an OUTCOME, not a per-tier
        # property, and is called out separately via the two patches appended below.
        ordered_keys = [l for l in LAYERS if l in seen_keys] + \
                       [k for k in seen_keys if k not in LAYERS]
        tier_patches = [mpatches.Patch(facecolor=_attribution_color(k), alpha=0.95,
                                        label=_attribution_label(k)) for k in ordered_keys]
        outcome_patches = [
            mpatches.Patch(facecolor="#555555", alpha=0.95, edgecolor="white",
                            label="Real bug caught (good)"),
            mpatches.Patch(facecolor=_lighten("#555555", 0.6), alpha=0.95, edgecolor="white",
                            label="False reject (needs work)"),
        ]
        # Below the axes, not in a corner -- with 9+ kernel rows a solid bar can span the
        # full width and collide with any in-plot legend position. Two rows: tier color,
        # then outcome meaning, so the two encodings don't run together in one line.
        leg1 = ax.legend(handles=tier_patches, loc="upper center", bbox_to_anchor=(0.5, -0.14),
                          fontsize=8, frameon=False, ncol=len(tier_patches), title="Tier")
        ax.add_artist(leg1)
        ax.legend(handles=outcome_patches, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                   fontsize=8, frameon=False, ncol=2, title="Outcome")
    # Footnote: define FALSE REJECT plainly, separate it from the (zero) soundness error,
    # and record the known contributors honestly -- fixes so far are PARTIAL, not a claim
    # of resolution (user feedback 2026-07-27: don't let this read as whitewashed). Placed
    # below the legends.
    ax.text(0.0, -0.32,
            "Pale bars = FALSE REJECTS: framework marked INVALID but the independent oracle PASSed (correct code\n"
            "was rejected). These are a conservatism cost, NOT a soundness failure -- false ACCEPTs (accepting broken\n"
            "code) are 0 for the framework and cannot appear here. Tier 2 here includes its witness-chain add-on gate\n"
            "(a cheat-detection tripwire bolted on after observing LLM loop-deletion cheating, not a 4th core tier --\n"
            "see View A for the disaggregated Tier-2/Witness split). Known contributors (fixes below are partial / in\n"
            "progress, not fully resolved): Tier 2 = full-field comparison stricter than the kernel's own tolerance in\n"
            "some cases, plus witness re-insertion/binary-fold artifacts -- several fixed in commit 30d1c69, but false\n"
            "rejects still occur (root cause for the pure-comparison share still under investigation, no fix yet);\n"
            "Tier 3 = weak or source/variant-mismatched metamorphic stimulus -- partially addressed by SENSITIVITY\n"
            "hardening 7aaff6c, source/variant parameter-matching identified as a further fix but NOT YET implemented.",
            transform=ax.transAxes, ha="left", va="top", fontsize=7, color="#555555", style="italic")


def _write_ablation_figures(per_kernel_va: dict[str, dict], per_kernel_vb: dict[str, dict],
                             output_dir: str) -> None:
    """Write View A and View B as two SEPARATE standalone figures.

    Split out (2026-07-27, user request) from the earlier single stacked panel so each view
    can be refined (labels/colors/layout) independently. File names:
        ablation_view_a_activity.{png,pdf}      -- View A, self-correction activity by layer
        ablation_view_b_attribution.{png,pdf}   -- View B, oracle-anchored INVALID attribution
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # --- View A (horizontal grouped bars, one row band per kernel -- matches View B) ---
    n_kernels_a = max(len([k for k, va in per_kernel_va.items() if va.get("n", 0) > 0]), 1)
    fig_a, ax_a = plt.subplots(figsize=(11, 3.4 + 0.45 * n_kernels_a))
    _plot_view_a_chart(per_kernel_va, ax_a)
    fig_a.tight_layout()
    for ext in ("png", "pdf"):
        fig_a.savefig(os.path.join(output_dir, f"ablation_view_a_activity.{ext}"),
                      dpi=150, bbox_inches="tight")
    plt.close(fig_a)

    # --- View B (horizontal stacked bars, one row per kernel) ---
    n_kernels = max(len(per_kernel_vb), 1)
    fig_b, ax_b = plt.subplots(figsize=(11, 3.0 + 0.42 * n_kernels))
    _plot_view_b_chart(per_kernel_vb, ax_b)
    fig_b.tight_layout()
    for ext in ("png", "pdf"):
        fig_b.savefig(os.path.join(output_dir, f"ablation_view_b_attribution.{ext}"),
                      dpi=150, bbox_inches="tight")
    plt.close(fig_b)

    print(f"\nWrote {os.path.join(output_dir, 'ablation_view_a_activity.png')}")
    print(f"Wrote {os.path.join(output_dir, 'ablation_view_b_attribution.png')}")


def run(exp_dirs: list[str], out_csv: str | None = None, plot_dir: str | None = None) -> None:
    all_rows: list[dict] = []
    per_kernel_va: dict[str, dict] = {}
    per_kernel_vb: dict[str, dict] = {}
    agg_va_layers = {name: {"reachable": 0, "fail": 0} for name in LAYERS}
    agg_va_unique: dict[str, int] = {}
    agg_va_configs: dict[str, int] = {}
    agg_vb_true: dict[str, int] = {}
    agg_vb_false: dict[str, int] = {}
    agg_vb_n = 0
    agg_vb_n_invalid = 0

    for d in exp_dirs:
        attempts = _load_attempts(d)
        vb_df = _load_view_b(d)
        if attempts is None:
            print(f"[skip] no attempt_log in {d}")
            continue
        kernel = os.path.basename(d)
        if "app_name" in attempts.columns and not attempts.empty:
            kernel = _kernel_name(attempts["app_name"].iloc[0])

        va = _view_a(attempts)
        vb = _view_b(vb_df) if vb_df is not None else {"n": 0, "n_invalid": 0, "true_unique_catch": {}, "unique_false_reject": {}}

        print(f"\n===== {kernel} =====")
        _print_view_a(va)
        _print_view_b(vb)

        all_rows.extend(_csv_rows(kernel, va, vb))
        per_kernel_va[kernel] = va
        per_kernel_vb[kernel] = vb

        for name in LAYERS:
            agg_va_layers[name]["reachable"] += va["layers"].get(name, {}).get("reachable", 0)
            agg_va_layers[name]["fail"] += va["layers"].get(name, {}).get("fail", 0)
        agg_va_unique = _merge_counts(agg_va_unique, va.get("unique_catch", {}))
        agg_va_configs = _merge_counts(agg_va_configs, va.get("configs", {}))
        agg_vb_true = _merge_counts(agg_vb_true, vb.get("true_unique_catch", {}))
        agg_vb_false = _merge_counts(agg_vb_false, vb.get("unique_false_reject", {}))
        agg_vb_n += vb.get("n", 0)
        agg_vb_n_invalid += vb.get("n_invalid", 0)

    if not all_rows:
        print("no data")
        return

    print("\n" + "=" * 60)
    print("AGGREGATE (all kernels)")
    print("=" * 60)
    agg_va = {
        "n": sum(v["reachable"] for v in agg_va_layers.values()),  # informational only
        "layers": {
            name: {
                "reachable": v["reachable"],
                "fail": v["fail"],
                "detection_rate": (v["fail"] / v["reachable"]) if v["reachable"] else None,
            }
            for name, v in agg_va_layers.items()
        },
        "unique_catch": agg_va_unique,
        "configs": agg_va_configs,
    }
    agg_vb = {"n": agg_vb_n, "n_invalid": agg_vb_n_invalid, "true_unique_catch": agg_vb_true, "unique_false_reject": agg_vb_false}
    _print_view_a(agg_va)
    _print_view_b(agg_vb)

    if out_csv:
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        print(f"\nWrote {out_csv}")

    if plot_dir:
        _write_ablation_figures(per_kernel_va, per_kernel_vb, plot_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Oracle-anchored per-tier ablation (View A + View B, plan §8 T-F)")
    ap.add_argument("--exp-dir", nargs="+", required=True)
    default_csv = os.path.join(os.path.dirname(__file__), "ablation_results_cumulative.csv")
    ap.add_argument("--out-csv", default=default_csv)
    ap.add_argument("--plot-dir", default=None,
                     help="if given, write ablation_summary.png/pdf (View A + View B charts) here")
    a = ap.parse_args(argv)
    run(a.exp_dir, out_csv=a.out_csv, plot_dir=a.plot_dir)


if __name__ == "__main__":
    main()
