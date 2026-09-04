"""Aggregate loss-aware runs across seeds and plot the privacy–utility trade-off.

Panel A: F1 discrepancy (utility cost) vs mean DCR (privacy gain), one point
         per lambda/margin configuration, error bars = std across seeds.
Panel B: loss components over epochs for one representative run, showing the
         terms trading off during training.

Also writes the aggregated table to CSV so the numbers in the docs/report are
reproducible from disk rather than transcribed by hand.

Usage:
    uv run python -m src.experiments.plot_tradeoff --dataset heart --kwarg batch_size=32 --kwarg epochs=300
    uv run python -m src.experiments.plot_tradeoff --dataset adult --kwarg batch_size=500

Runs are grouped by (lambda_mmd, lambda_corr, lambda_priv, dcr_margin). When
the same (group, seed) appears more than once, only the most recent run is
kept, so re-running a configuration after a code fix supersedes the old
result automatically.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.persistence import load_report
from src.experiments.query import query_index

LOSS_KEYS = ("lambda_mmd", "lambda_corr", "lambda_priv", "dcr_margin")
LOSS_DEFAULTS = {"lambda_mmd": 0.0, "lambda_corr": 0.0, "lambda_priv": 0.0, "dcr_margin": 1.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_value(s: str):
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _parse_kwarg_filters(items: list[str] | None) -> dict:
    out = {}
    for item in items or []:
        k, _, v = item.partition("=")
        out[k] = _parse_value(v)
    return out


def _matches(row_kwargs: dict, filters: dict) -> bool:
    for k, v in filters.items():
        rv = row_kwargs.get(k)
        if isinstance(v, (int, float)) and isinstance(rv, (int, float)):
            if not np.isclose(float(rv), float(v)):
                return False
        elif rv != v:
            return False
    return True


def _group_key(kwargs: dict) -> tuple:
    return tuple(float(kwargs.get(k, LOSS_DEFAULTS[k])) for k in LOSS_KEYS)


def _label(key: tuple) -> str:
    lm, lc, lp, m = key
    if lm == 0 and lc == 0 and lp == 0:
        return "baseline"
    if lp == 0:
        return "utility only"
    if lm == 0 and lc == 0:
        return f"priv only  λ={lp:g}  μ={m:g}"
    return f"λ={lp:g}  μ={m:g}"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def collect_runs(dataset: str, generator: str, filters: dict, output_dir: str) -> dict[tuple, dict[int, dict]]:
    """Return {group_key: {seed: report}} keeping the most recent run per (group, seed)."""
    rows = query_index(output_dir=output_dir, dataset_name=dataset, generator_name=generator)
    latest: dict[tuple, dict[int, tuple[str, str]]] = defaultdict(dict)  # (created_at, run_name)
    for row in rows:
        kw = row.get("generator_kwargs") or {}
        if not _matches(kw, filters):
            continue
        key = _group_key(kw)
        seed = row["seed"]
        created = row.get("created_at") or ""
        if seed not in latest[key] or created > latest[key][seed][0]:
            latest[key][seed] = (created, row["run_name"])

    runs: dict[tuple, dict[int, dict]] = {}
    for key, per_seed in latest.items():
        runs[key] = {}
        for seed, (_, run_name) in per_seed.items():
            try:
                runs[key][seed] = load_report(run_name, output_dir)
            except FileNotFoundError:
                continue
    return runs


METRICS = [
    ("utility", "f1_discrepancy"),
    ("utility", "mmd"),
    ("utility", "mean_emd"),
    ("utility", "correlation_distance"),
    ("privacy", "dcr_mean"),
    ("privacy", "dcr_5th_percentile"),
    ("privacy", "nndr_mean"),
    ("privacy", "disclosure_rate"),
]


def aggregate(runs: dict[tuple, dict[int, dict]]) -> pd.DataFrame:
    records = []
    for key, per_seed in runs.items():
        rec = {k: v for k, v in zip(LOSS_KEYS, key)}
        rec["label"] = _label(key)
        rec["n_seeds"] = len(per_seed)
        for section, metric in METRICS:
            vals = np.array([r[section][metric] for r in per_seed.values()], dtype=float)
            rec[f"{metric}_mean"] = vals.mean()
            rec[f"{metric}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        records.append(rec)
    df = pd.DataFrame(records)
    return df.sort_values(["lambda_priv", "dcr_margin", "lambda_mmd"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tradeoff(ax, agg: pd.DataFrame, dataset: str) -> None:
    for _, r in agg.iterrows():
        ax.errorbar(
            r["dcr_mean_mean"], r["f1_discrepancy_mean"],
            xerr=r["dcr_mean_std"], yerr=r["f1_discrepancy_std"],
            fmt="o", capsize=3, markersize=6,
        )
        ax.annotate(
            r["label"], (r["dcr_mean_mean"], r["f1_discrepancy_mean"]),
            textcoords="offset points", xytext=(6, 6), fontsize=8,
        )
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("mean DCR  (higher = more private)")
    ax.set_ylabel("F1 discrepancy  (higher = more utility lost)")
    ax.set_title(f"Privacy–utility trade-off — {dataset}")
    ax.grid(alpha=0.3)


def plot_loss_curves(ax, report: dict, title: str) -> None:
    lv = pd.DataFrame(report["artifacts"]["loss_values"])
    per_epoch = lv.groupby("Epoch")[["Recon", "KL", "MMD", "Corr", "Priv"]].mean()

    ax.plot(per_epoch.index, per_epoch["Recon"], label="Recon", lw=1.5)
    ax.plot(per_epoch.index, per_epoch["KL"], label="KL", lw=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("Recon / KL")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    for col, ls in (("MMD", "-"), ("Corr", "--"), ("Priv", ":")):
        if (per_epoch[col] != 0).any():
            ax2.plot(per_epoch.index, per_epoch[col], label=col, lw=1.5, ls=ls, color="black", alpha=0.7)
    ax2.set_ylabel("penalty terms")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax.set_title(title)


def pick_loss_run(runs: dict[tuple, dict[int, dict]], lambda_priv: float, dcr_margin: float) -> tuple[str, dict] | None:
    for key, per_seed in runs.items():
        _, _, lp, m = key
        if np.isclose(lp, lambda_priv) and np.isclose(m, dcr_margin) and per_seed:
            seed = min(per_seed)
            return f"{_label(key)}  seed={seed}", per_seed[seed]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--generator", default="tvae_loss_aware")
    p.add_argument("--kwarg", action="append", metavar="KEY=VALUE",
                   help="only include runs whose generator_kwargs match (repeatable), e.g. batch_size=32")
    p.add_argument("--results-dir", default="experiments/results")
    p.add_argument("--out-dir", default="experiments/figures")
    p.add_argument("--loss-lambda-priv", type=float, default=10.0,
                   help="lambda_priv of the run to use for the loss-curve panel")
    p.add_argument("--loss-margin", type=float, default=1.5,
                   help="dcr_margin of the run to use for the loss-curve panel")
    p.add_argument("--exclude-collapse", type=float, default=None, metavar="DCR",
                   help="drop configurations whose mean DCR exceeds this (keeps the plot readable)")
    args = p.parse_args()

    filters = _parse_kwarg_filters(args.kwarg)
    runs = collect_runs(args.dataset, args.generator, filters, args.results_dir)
    if not runs:
        raise SystemExit(f"no runs found for dataset={args.dataset} generator={args.generator} filters={filters}")

    agg = aggregate(runs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.dataset + ("_" + "_".join(f"{k}{v}" for k, v in filters.items()) if filters else "")
    csv_path = out_dir / f"tradeoff_{tag}.csv"
    agg.to_csv(csv_path, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(agg[["label", "n_seeds",
                   "f1_discrepancy_mean", "f1_discrepancy_std",
                   "dcr_mean_mean", "dcr_mean_std",
                   "dcr_5th_percentile_mean", "mean_emd_mean",
                   "correlation_distance_mean", "disclosure_rate_mean"]])

    plot_agg = agg if args.exclude_collapse is None else agg[agg["dcr_mean_mean"] <= args.exclude_collapse]

    loss_run = pick_loss_run(runs, args.loss_lambda_priv, args.loss_margin)
    ncols = 2 if loss_run else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    axes = np.atleast_1d(axes)

    plot_tradeoff(axes[0], plot_agg, args.dataset)
    if loss_run:
        title, report = loss_run
        plot_loss_curves(axes[1], report, f"Loss components — {title}")

    fig.tight_layout()
    png_path = out_dir / f"tradeoff_{tag}.png"
    fig.savefig(png_path, dpi=150)
    print(f"\nwrote {csv_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()
