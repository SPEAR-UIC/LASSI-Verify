#!/usr/bin/env python3
"""
paper_oracle_results.py — turn the oracle-anchored confusion matrix into the
SC26 paper's results tables + figures.

Reads the machine-readable cumulative rows
(`lassi_verify_analysis/oracle_confusion_matrix_cumulative.csv`, produced by
`oracle_confusion.py`) and aggregates the four reported axes separately
(framework/judge x LASSI/zero-shot) into:

  - headline soundness/completeness table (markdown + CSV)
  - per-kernel framework/judge breakdown table (markdown)
  - Figure A: soundness & completeness across the 4 axes
  - Figure B: per-kernel framework outcomes on the LASSI population (stacked)
  - Figure C: framework-vs-judge false-accepts on the LASSI population
             (the "tiers beat the judge baseline" discriminator)

Kernel set is parameterized via --exclude so deferred kernels (jacobi,
matrix-rotate) stay out and an extrema top-up folds in automatically once
`oracle_confusion.py` refreshes the CSV. Default excludes jacobi (SC26 decision
2026-07-27: jacobi -> future work; matrix-rotate was never persisted to the CSV).

Report rule (hard project constraint): the four axes are NEVER pooled.
"""
import argparse
import csv
import os
from collections import defaultdict

# Fixed kernel display order (generation-time order of the campaign).
KERNEL_ORDER = [
    "entropy", "extrema", "all-pairs-distance", "atomicCost", "threadfence",
    "layout", "bsearch", "randomaccess", "colorwheel", "chacha20",
    "floydwarshall", "murmurhash3", "jaccard", "lid-driven-cavity",
    "segment-reduce", "pathfinder", "jacobi",
]

# Status-semantic, CVD-safe outcome palette (green/blue/amber/red).
# TA & TR are both correct outcomes (cool/green); OS is a safe-direction warning;
# FA is the only unsound outcome (critical). Direct labels on every mark satisfy
# the dataviz relief rule (identity never rests on color alone).
COLORS = {
    "TA": "#2a9d5c",  # true-accept  (correct accept)
    "TR": "#2a78d6",  # true-reject  (correct reject)
    "OS": "#eda100",  # over-strict  (safe-direction false-reject)
    "FA": "#d1382c",  # false-accept (unsound — the one that matters)
}
GRID = "#d9d9d6"
INK = "#0b0b0b"
INK2 = "#52514e"

AXES = [
    ("framework", "lassi"),
    ("judge", "lassi"),
    ("framework", "zeroshot"),
    ("judge", "zeroshot"),
]
AXIS_LABEL = {
    ("framework", "lassi"): "Verify x LASSI",
    ("judge", "lassi"): "LLM Judge x LASSI",
    ("framework", "zeroshot"): "Verify x zero-shot",
    ("judge", "zeroshot"): "LLM Judge x zero-shot",
}


def load(csv_path, exclude):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["kernel"] in exclude:
                continue
            for k in ("true_accept", "false_accept", "over_strict", "true_reject"):
                r[k] = int(r[k])
            rows.append(r)
    return rows


def aggregate(rows):
    """Return {(axis,gen): dict(TA,FA,OS,TR,n,soundness,completeness)}."""
    agg = defaultdict(lambda: dict(TA=0, FA=0, OS=0, TR=0))
    for r in rows:
        a = agg[(r["axis"], r["generation"])]
        a["TA"] += r["true_accept"]
        a["FA"] += r["false_accept"]
        a["OS"] += r["over_strict"]
        a["TR"] += r["true_reject"]
    for a in agg.values():
        a["n"] = a["TA"] + a["FA"] + a["OS"] + a["TR"]
        denom_s = a["FA"] + a["TR"]
        a["soundness"] = (1 - a["FA"] / denom_s) if denom_s else float("nan")
        denom_c = a["TA"] + a["OS"]
        a["completeness"] = (a["TA"] / denom_c) if denom_c else float("nan")
    return agg


def per_kernel(rows):
    d = defaultdict(dict)
    for r in rows:
        d[r["kernel"]][(r["axis"], r["generation"])] = r
    return d


def fmt(v):
    return "—" if v != v else f"{v:.3f}"  # NaN check


def write_tables(agg, pk, kernels, out_md):
    lines = []
    lines.append("## Headline — soundness & completeness (four axes, never pooled)\n")
    lines.append("| Axis | n | TA | FA | OS | TR | Soundness | Completeness |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for ax in AXES:
        a = agg[ax]
        lines.append(
            f"| {AXIS_LABEL[ax]} | {a['n']} | {a['TA']} | **{a['FA']}** | {a['OS']} | "
            f"{a['TR']} | {fmt(a['soundness'])} | {fmt(a['completeness'])} |"
        )
    lines.append("")
    lines.append("Soundness = 1 - FA/(FA+TR); Completeness = TA/(TA+OS).\n")

    lines.append("## Per-kernel — framework & judge on the LASSI population\n")
    lines.append("| Kernel | n | FW TA | FW FA | FW OS | FW TR | JD TA | JD FA | JD OS | JD TR |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in kernels:
        fw = pk[k].get(("framework", "lassi"))
        jd = pk[k].get(("judge", "lassi"))
        if not fw:
            continue
        n = fw["true_accept"] + fw["false_accept"] + fw["over_strict"] + fw["true_reject"]
        lines.append(
            f"| {k} | {n} | {fw['true_accept']} | {fw['false_accept']} | {fw['over_strict']} | "
            f"{fw['true_reject']} | {jd['true_accept']} | {jd['false_accept']} | "
            f"{jd['over_strict']} | {jd['true_reject']} |"
        )
    lines.append("")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)


def make_figures(agg, pk, kernels, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": INK2, "axes.linewidth": 0.8,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "figure.dpi": 200, "svg.fonttype": "none",
    })

    def save(fig, name):
        fig.savefig(os.path.join(out_dir, name + ".png"), bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    import numpy as np

    # ---- Figure A: soundness & completeness across the 4 axes ----
    # Two conceptually different measures -> two distinct hues (CVD-safe blue/orange
    # adjacent pair from the reference palette). Soundness (the property that matters
    # for a correctness tool) gets the primary blue; completeness gets orange.
    C_SOUND = "#2a78d6"   # blue  — soundness
    C_COMPL = "#eb6834"   # orange — completeness
    HEADLINE_ROW = 0      # Verify x LASSI: the paper's headline axis

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels = [AXIS_LABEL[a] for a in AXES]
    snd = [agg[a]["soundness"] for a in AXES]
    cmp = [agg[a]["completeness"] for a in AXES]
    y = np.arange(len(AXES))
    h = 0.38

    # Faint highlight band behind the headline axis (Framework x LASSI).
    ax.axhspan(y[HEADLINE_ROW] - 0.5, y[HEADLINE_ROW] + 0.5, color="#eef4fb", zorder=0)

    b1 = ax.barh(y + h / 2, snd, height=h, color=C_SOUND, label="Soundness  (1 - FA/(FA+TR))",
                 zorder=3)
    b2 = ax.barh(y - h / 2, cmp, height=h, color=C_COMPL, label="Completeness  (TA/(TA+FR))",
                 zorder=3)
    for bars, vals in ((b1, snd), (b2, cmp)):
        for i, (rect, v) in enumerate(zip(bars, vals)):
            if v != v:  # NaN
                continue
            bold = (i == HEADLINE_ROW and v >= 0.9999)  # emphasize the perfect-soundness headline
            # Values at/near 1.0 clear the dashed reference line with visible white
            # space instead of sitting flush against it (padding grows near the cap).
            pad = 0.03 if v >= 0.999 else 0.015
            ax.text(v + pad, rect.get_y() + rect.get_height() / 2,
                    f"{v:.3f}", va="center", ha="left", fontsize=8.5, color=INK,
                    fontweight=("bold" if bold else "normal"))

    # In-bar callout on the headline soundness bar (white text, left-aligned) — the
    # blue bar fills 0..1.0 so there is room here, and it can't overflow the axes.
    fa_h = agg[AXES[HEADLINE_ROW]]["FA"]
    n_h = agg[AXES[HEADLINE_ROW]]["n"]
    ax.text(0.02, y[HEADLINE_ROW] + h / 2,
            f"perfect soundness — {fa_h} false-accepts in {n_h} variants",
            va="center", ha="left", fontsize=8, color="white", fontweight="bold", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for i, tl in enumerate(ax.get_yticklabels()):
        if i == HEADLINE_ROW:
            tl.set_fontweight("bold")  # headline axis label bold
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.invert_yaxis()
    # Faint reference line at the ideal (1.0).
    ax.axvline(1.0, color="#b6b6b2", lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_xlabel("Rate")
    ax.set_title("Verification quality: Verify vs. LLM Judge, LASSI vs. zero-shot",
                 fontsize=11, loc="left", pad=8)
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.30), ncol=2, frameon=False, fontsize=8.5)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "figA_soundness_completeness")

    # ---- Figure B: per-kernel framework outcomes, LASSI (stacked) ----
    ks = [k for k in kernels if pk[k].get(("framework", "lassi"))]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    yk = np.arange(len(ks))
    order = [("TA", "true_accept"), ("TR", "true_reject"), ("OS", "over_strict"), ("FA", "false_accept")]
    left = np.zeros(len(ks))
    for key, col in order:
        vals = np.array([pk[k][("framework", "lassi")][col] for k in ks], dtype=float)
        ax.barh(yk, vals, left=left, height=0.7, color=COLORS[key],
                edgecolor="white", linewidth=1.2, zorder=3)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(left[i] + v / 2, yk[i], f"{int(v)}", va="center", ha="center",
                        fontsize=7.5, color="white", fontweight="bold")
        left += vals
    ax.set_yticks(yk)
    ax.set_yticklabels(ks)
    ax.invert_yaxis()
    ax.set_xlabel("Oracle-gradable variants")
    ax.set_title("Verification outcomes by kernel (LASSI population)",
                 fontsize=11, loc="left", pad=40)
    legend = [Patch(facecolor=COLORS["TA"], label="True accept"),
              Patch(facecolor=COLORS["TR"], label="True reject"),
              Patch(facecolor=COLORS["OS"], label="False reject"),
              Patch(facecolor=COLORS["FA"], label="False accept (n = 0)")]
    ax.legend(handles=legend, loc="lower right", bbox_to_anchor=(1.0, 1.008),
              ncol=2, frameon=False, fontsize=8.5)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "figB_per_kernel_framework")

    # ---- Figure C: framework vs judge false-accepts, LASSI (the discriminator) ----
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    fw = agg[("framework", "lassi")]
    jd = agg[("judge", "lassi")]
    cats = ["False-accept\n(reduces soundness)", "True-reject\n(caught real bug)"]
    fw_v = [fw["FA"], fw["TR"]]
    jd_v = [jd["FA"], jd["TR"]]
    x = np.arange(len(cats))
    w = 0.36
    bF = ax.bar(x - w / 2, fw_v, width=w, color="#2a78d6", label="Verification (three tiers)", zorder=3)
    bJ = ax.bar(x + w / 2, jd_v, width=w, color="#d1382c", label="LLM Judge (baseline)", zorder=3)
    for bars in (bF, bJ):
        for rect in bars:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.15,
                    f"{int(rect.get_height())}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Variants (LASSI, n=%d)" % fw["n"])
    ax.set_ylim(0, max(max(fw_v), max(jd_v)) + 3)
    ax.set_title("Soundness formula inputs: Verification vs. LLM Judge",
                 fontsize=11, loc="left", pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "figC_framework_vs_judge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="lassi_verify_analysis/oracle_confusion_matrix_cumulative.csv")
    ap.add_argument("--exclude", nargs="*", default=["jacobi"],
                    help="kernels to omit (SC26 default: jacobi -> future work)")
    ap.add_argument("--out-dir", default="lassi_verify_analysis/oracle_figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load(args.csv, set(args.exclude))
    agg = aggregate(rows)
    pk = per_kernel(rows)
    kernels = [k for k in KERNEL_ORDER if k not in set(args.exclude) and k in pk]

    md = write_tables(agg, pk, kernels, os.path.join(args.out_dir, "results_tables.md"))
    print(md)
    print(f"\nExcluded: {sorted(set(args.exclude))} | kernels in set: {len(kernels)}")

    make_figures(agg, pk, kernels, args.out_dir)
    print(f"Figures + tables written to {args.out_dir}/")


if __name__ == "__main__":
    main()
