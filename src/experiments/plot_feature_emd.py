"""Per-feature EMD analysis: which columns degrade first as lambda_priv rises?

Every run report stores `utility.emd_per_feature`. This script averages it
across seeds for each lambda_priv configuration, expresses each feature's EMD
as a ratio to the plain baseline (all lambdas = 0), and draws a heatmap with
features on the y-axis sorted by cardinality and lambda_priv on the x-axis.

If the density account is right, the high-cardinality columns — the ones that
mode-specific normalization expands into the widest one-hot spans — should be
the first to degrade on Diabetes.

Usage:
    uv run python -m src.experiments.plot_feature_emd --dataset heart --kwarg batch_size=32 --kwarg epochs=300
    uv run python -m src.experiments.plot_feature_emd --dataset diabetes --kwarg batch_size=500
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.experiments.plot_tradeoff import LOSS_KEYS, _label, _parse_kwarg_filters, collect_runs


def feature_emd_table(runs: dict, only_margin: float | None) -> tuple[pd.DataFrame, dict]:
    """Return (DataFrame features x configs of mean EMD, {config_label: lambda_priv})."""
    cols = {}
    lam = {}
    for key, per_seed in runs.items():
        lm, lc, lp, m = key
        # keep the plain baseline, and the 'both' configs at the chosen margin
        is_baseline = (lm == 0 and lc == 0 and lp == 0)
        is_both = (lm > 0 and lc > 0)
        if not (is_baseline or is_both):
            continue
        if is_both and only_margin is not None and not np.isclose(m, only_margin):
            continue
        frames = []
        for r in per_seed.values():
            pf = r.get("utility", {}).get("emd_per_feature")
            if pf:
                frames.append(pd.Series(pf, dtype=float))
        if not frames:
            continue
        label = "baseline" if is_baseline else f"λ={lp:g}"
        cols[label] = pd.concat(frames, axis=1).mean(axis=1)
        lam[label] = lp
    df = pd.DataFrame(cols)
    ordered = sorted(df.columns, key=lambda c: lam[c])
    return df[ordered], lam


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--generator", default="tvae_loss_aware")
    p.add_argument("--kwarg", action="append", metavar="KEY=VALUE")
    p.add_argument("--margin", type=float, default=1.5, help="dcr_margin of the 'both' runs to include")
    p.add_argument("--results-dir", default="experiments/results")
    p.add_argument("--out-dir", default="experiments/figures")
    p.add_argument("--top", type=int, default=8, help="how many most-degraded features to print")
    args = p.parse_args()

    filters = _parse_kwarg_filters(args.kwarg)
    runs = collect_runs(args.dataset, args.generator, filters, args.results_dir)
    if not runs:
        raise SystemExit("no runs found")

    df, lam = feature_emd_table(runs, args.margin)
    if "baseline" not in df.columns:
        raise SystemExit("no plain-baseline run (all lambdas = 0) found to normalize against")
    if df.shape[1] < 2:
        raise SystemExit("need at least one 'both' configuration besides the baseline")

    # cardinality from the real data, to sort features and test the density account
    bundle = load_dataset(args.dataset)
    nunique = bundle.real.nunique()
    real_std = bundle.real.std(ddof=0)
    df = df.loc[[c for c in df.index if c in nunique.index]]
    df["nunique"] = nunique.loc[df.index].values
    df["real_std"] = real_std.loc[df.index].values
    df = df.sort_values("nunique")

    ratio = df.drop(columns=["nunique", "real_std"]).div(df["baseline"].replace(0, np.nan), axis=0)
    log2ratio = np.log2(ratio.replace(0, np.nan)).fillna(0)
    # drop constant columns (baseline EMD exactly 0) from the plot; nothing to say about them
    keep = df["baseline"] > 0
    df, ratio, log2ratio = df[keep], ratio[keep], log2ratio[keep]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.dataset + ("_" + "_".join(f"{k}{v}" for k, v in filters.items()) if filters else "")

    table = df.copy()
    for c in ratio.columns:
        table[f"{c}_ratio"] = ratio[c]
    csv_path = out_dir / f"feature_emd_{tag}.csv"
    table.to_csv(csv_path)

    # --- heatmap -------------------------------------------------------------
    plot = log2ratio.drop(columns="baseline")
    ylabels = [f"{f}  (k={int(k)})" for f, k in zip(plot.index, df["nunique"])]
    vmax = float(np.nanmax(np.abs(plot.values))) or 1.0

    # Fixed, print-friendly geometry: width does not depend on the number of
    # lambda columns (a single column would otherwise become a tall strip), and
    # row height is capped so 40-row datasets stay within one page.
    n_rows, n_cols = plot.shape
    fig_w = 6.5
    row_h = 0.26 if n_rows <= 20 else 0.21
    fig_h = min(max(3.0, row_h * n_rows + 1.4), 11.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(plot.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(plot.columns, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(ylabels, fontsize=8 if n_rows <= 20 else 7)
    ax.set_xlabel("privacy weight", fontsize=9)
    ax.set_title(f"{args.dataset}: per-feature EMD vs baseline", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6 if n_rows > 20 else 0.8, pad=0.02)
    cbar.set_label("log2(EMD / baseline EMD)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    png_path = out_dir / f"feature_emd_{tag}.png"
    fig.savefig(png_path, dpi=200)

    # --- console summary -----------------------------------------------------
    # Three rankings, each unit-confounded in its own way:
    #   ratio      explodes when baseline EMD ~ 0 (binaries, near-constant columns)
    #   absolute   is dominated by wide-range raw units (dollars, code indices)
    #   std-scaled divides the change by the real column's std, putting a
    #              binary's proportion shift and a continuous shift on one scale.
    # The std-scaled one is the primary ranking. For a binary label-encoded
    # column, EMD is |p_real - p_synth| and reads as percentage points.
    top_cfg = plot.columns[-1]
    change = df[top_cfg] - df["baseline"]
    scaled = (change / df["real_std"].replace(0, np.nan)).dropna().sort_values(ascending=False)

    print(f"\nLargest std-scaled EMD increase at {top_cfg}  ((EMD - baseline) / real std):")
    for f, s in scaled.head(args.top).items():
        k = int(df.loc[f, "nunique"])
        pp = f"   = {100 * change[f]:+.1f} pp shift in proportion" if k == 2 else ""
        print(f"  {f:<24} {s:6.2f} sd   {df.loc[f, 'baseline']:9.4f} -> {df.loc[f, top_cfg]:9.4f}   (k={k}){pp}")

    print(f"\nLargest absolute EMD increase at {top_cfg} (raw units):")
    for f, d in change.sort_values(ascending=False).head(args.top).items():
        print(f"  {f:<24} {df.loc[f, 'baseline']:9.4f} -> {df.loc[f, top_cfg]:9.4f}   (k={int(df.loc[f, 'nunique'])})")

    improved = ratio[top_cfg][ratio[top_cfg] < 0.8].sort_values()
    if len(improved):
        print(f"\nFeatures that IMPROVED at {top_cfg} (ratio < 0.8):")
        for f, v in improved.items():
            print(f"  {f:<24} x{v:5.2f}   (k={int(df.loc[f, 'nunique'])})")

    # cardinality vs degradation, on finite values only
    finite = ratio[top_cfg].replace([np.inf, -np.inf], np.nan).dropna()
    finite = finite[finite > 0]
    if len(finite) > 3:
        k = np.log(df.loc[finite.index, "nunique"].astype(float))
        r = np.corrcoef(k, np.log(finite))[0, 1]
        print(f"\ncorr(log cardinality, log EMD ratio) at {top_cfg} = {r:+.2f}  (n={len(finite)} finite columns)")

    print(f"\nwrote {csv_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()
