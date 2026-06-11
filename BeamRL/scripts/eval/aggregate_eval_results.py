#!/usr/bin/env python3
"""
Aggregate evaluation results across multiple random seeds.

Reads results.json files from the directory structure produced by eval_model_beamrl.sh:
    {output_dir}/beamrl_eval/{seed}/{model_ckpt_id}/results.json

Computes mean ± std per metric per checkpoint across all seeds found.

Outputs:
    - Console summary table
    - aggregated_results.csv  (one row per checkpoint × metric)
    - aggregated_results.png  (line plot of mean ± std over checkpoints)

Usage:
    python aggregate_eval_results.py --output_dir /path/to/output_dir
    python aggregate_eval_results.py --output_dir /path/to/output_dir --seeds 42 123 456
    python aggregate_eval_results.py --output_dir /path/to/output_dir --pattern "*beamrl_260101*"
"""

import argparse
import csv
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Metrics read directly from results.json (precomputed by run_dataset_eval.py).
PRECOMPUTED_METRICS = ["accuracy_at_1", "accuracy_avg", "accuracy_majority", "format_score"]

# Metrics derived from the per-sample `accuracy_rewards` array in results.json.
# Pass@k: "at least one of k generations is correct." Run-time-derived because
# run_dataset_eval.py doesn't write it as a top-level scalar.
DERIVED_METRICS = ["accuracy_pass_at_k"]

# All metrics reported by this aggregator (precomputed + derived).
METRICS = PRECOMPUTED_METRICS + DERIVED_METRICS

METRIC_LABELS = {
    "accuracy_at_1":      "Pass@1",
    "accuracy_pass_at_k": "Pass@k",
    "accuracy_avg":       "Avg. Accuracy",
    "accuracy_majority":  "Majority@k",
    "format_score":       "Format Score",
}

# Pass@k threshold for the "at least one correct" metric.
# Mirrors num_generations: 7 in the evaluation recipes.
PASS_K_DEFAULT = 7


def extract_checkpoint_number(ckpt_dir_name: str) -> int:
    """Extract the numeric step from a checkpoint directory name for sorting."""
    match = re.search(r"checkpoint-(\d+)", ckpt_dir_name)
    return int(match.group(1)) if match else 0


def find_result_files(output_dir: Path, seeds: list[int], pattern: str) -> dict:
    """
    Walk the output directory and collect results.json paths.

    Returns:
        dict mapping checkpoint_id -> {seed -> Path}
    """
    data = defaultdict(dict)  # ckpt_id -> {seed -> path}

    for seed in seeds:
        seed_dir = output_dir / "beamrl_eval" / str(seed)
        if not seed_dir.exists():
            logger.warning(f"Seed directory not found: {seed_dir}")
            continue

        for ckpt_dir in sorted(seed_dir.iterdir()):
            if not ckpt_dir.is_dir():
                continue
            if pattern and not _match_pattern(ckpt_dir.name, pattern):
                continue
            results_file = ckpt_dir / "results.json"
            if results_file.exists():
                data[ckpt_dir.name][seed] = results_file
            else:
                logger.warning(f"Missing results.json in: {ckpt_dir}")

    return data


def _match_pattern(name: str, pattern: str) -> bool:
    """Simple glob-style matching (supports * wildcard)."""
    regex = re.escape(pattern).replace(r"\*", ".*")
    return bool(re.fullmatch(regex, name))


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _pass_at_k_from_rewards(acc_rewards: list) -> float | None:
    """Per-seed Pass@k: fraction of samples for which at least one of the k
    generations received a positive accuracy reward. Returns None if input is
    not the expected per-sample-per-generation list-of-lists."""
    if not isinstance(acc_rewards, list) or not acc_rewards:
        return None
    n_passed = 0
    n_samples = 0
    for row in acc_rewards:
        if not isinstance(row, list) or not row:
            continue
        n_samples += 1
        if any(float(v) > 0.0 for v in row):
            n_passed += 1
    if n_samples == 0:
        return None
    return n_passed / n_samples


def aggregate(data: dict) -> dict:
    """
    Compute mean ± std for each metric per checkpoint.

    Precomputed metrics (Pass@1, Avg, Majority, Format) are read directly from
    results.json. Derived metrics (Pass@k) are computed here from the per-sample
    `accuracy_rewards` array — Pass@k is not written as a top-level scalar by
    run_dataset_eval.py, so it must be derived if we want it at the overall
    level (the per-category aggregator already derives it; this brings the
    overall aggregator into parity).

    Returns:
        dict mapping ckpt_id -> {metric -> {"mean": float, "std": float, "n": int, "values": list}}
    """
    agg = {}
    for ckpt_id, seed_paths in data.items():
        agg[ckpt_id] = {}

        # Read each seed's results.json once, extract both precomputed and derived metrics.
        per_seed_values: dict[str, list[float]] = {m: [] for m in METRICS}
        for seed, path in seed_paths.items():
            try:
                result = load_results(path)
            except Exception as e:
                logger.warning(f"Could not read {path}: {e}")
                continue

            for metric in PRECOMPUTED_METRICS:
                if metric in result:
                    per_seed_values[metric].append(float(result[metric]))
                else:
                    logger.warning(f"Metric '{metric}' not found in {path}")

            # Derived: Pass@k from per-sample accuracy_rewards
            pass_at_k = _pass_at_k_from_rewards(result.get("accuracy_rewards"))
            if pass_at_k is not None:
                per_seed_values["accuracy_pass_at_k"].append(pass_at_k)
            else:
                logger.warning(
                    f"Could not derive accuracy_pass_at_k from {path} "
                    "(missing or malformed accuracy_rewards)"
                )

        for metric, values in per_seed_values.items():
            if values:
                agg[ckpt_id][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n": len(values),
                    "values": values,
                }
            else:
                logger.warning(f"No values for metric '{metric}' in checkpoint '{ckpt_id}'")
    return agg


def print_summary(agg: dict) -> None:
    """Print a formatted table to the console."""
    ckpt_ids = sorted(agg.keys(), key=extract_checkpoint_number)

    col_w = 22
    header = f"{'Checkpoint':<40}" + "".join(f"{METRIC_LABELS[m]:>{col_w}}" for m in METRICS)
    print("\n" + "=" * (40 + col_w * len(METRICS)))
    print("AGGREGATED EVALUATION RESULTS (mean ± std)")
    print("=" * (40 + col_w * len(METRICS)))
    print(header)
    print("-" * (40 + col_w * len(METRICS)))

    for ckpt_id in ckpt_ids:
        row = f"{ckpt_id:<40}"
        for metric in METRICS:
            if metric in agg[ckpt_id]:
                m = agg[ckpt_id][metric]
                cell = f"{m['mean']:.4f} ± {m['std']:.4f} (n={m['n']})"
            else:
                cell = "N/A"
            row += f"{cell:>{col_w}}"
        print(row)

    print("=" * (40 + col_w * len(METRICS)) + "\n")


def write_csv(agg: dict, output_path: Path) -> None:
    """Write aggregated results to a CSV file."""
    ckpt_ids = sorted(agg.keys(), key=extract_checkpoint_number)

    fieldnames = ["checkpoint", "checkpoint_step", "metric", "mean", "std", "n_seeds"] + \
                 [f"seed_{i}_value" for i in range(10)]  # up to 10 seeds

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for ckpt_id in ckpt_ids:
            step = extract_checkpoint_number(ckpt_id)
            for metric in METRICS:
                if metric not in agg[ckpt_id]:
                    continue
                m = agg[ckpt_id][metric]
                row = {
                    "checkpoint": ckpt_id,
                    "checkpoint_step": step,
                    "metric": metric,
                    "mean": m["mean"],
                    "std": m["std"],
                    "n_seeds": m["n"],
                }
                for i, v in enumerate(m["values"]):
                    row[f"seed_{i}_value"] = v
                writer.writerow(row)

    logger.info(f"CSV written to: {output_path}")


def load_eval_categories(eval_dataset_name: str, eval_split: str) -> list[str]:
    """
    Load the eval dataset from HuggingFace Hub and return its `category` column
    as a Python list, in dataset order.

    The order must match the order in which run_dataset_eval.py wrote per-sample
    arrays into results.json (which iterates the dataset in order via index ranges,
    so this assumption holds).
    """
    from datasets import load_dataset

    logger.info(f"Loading eval dataset {eval_dataset_name} (split={eval_split}) for per-category labels ...")
    ds = load_dataset(eval_dataset_name, split=eval_split)
    if "category" not in ds.column_names:
        logger.warning(
            f"Dataset {eval_dataset_name} has no 'category' column "
            f"(columns: {ds.column_names}); falling back to 'uncategorized'."
        )
        return ["uncategorized"] * len(ds)
    return list(ds["category"])


def load_eval_categories_from_json(path: Path) -> list[str]:
    """
    Load per-sample category labels from a local JSON file, in dataset order.

    Offline alternative to load_eval_categories(): some environments cannot
    `load_dataset("tphage/BeamRL-EvalData-v2")` because pyarrow-19 raises
    "Repetition level histogram size mismatch" on the v2 parquet. In that case
    extract the `category` column once (e.g. with duckdb) into a local JSON file.

    Accepts either {"categories": [...]} or a bare JSON list.
    """
    with open(path) as f:
        obj = json.load(f)
    cats = obj["categories"] if isinstance(obj, dict) else obj
    logger.info(f"Loaded {len(cats)} category labels from local JSON {path}.")
    return list(cats)


def aggregate_per_category(
    data: dict,
    categories: list[str],
    pass_k: int = PASS_K_DEFAULT,
) -> dict:
    """
    Compute per-category mean ± std for each metric per checkpoint.

    Each results.json file is expected to contain per-sample arrays
    (`accuracy_rewards` of shape [n_samples][n_generations], `format_rewards` similar).
    For each sample we derive:
        - accuracy_at_1   = first generation's accuracy reward (1.0 or 0.0)
        - accuracy_avg    = mean of accuracy rewards across all generations
        - accuracy_majority = 1.0 if >=ceil(k/2) generations correct, else 0.0
        - accuracy_pass_at_k = 1.0 if any generation correct, else 0.0  [new metric]
        - format_score    = mean format reward across generations
    These are then grouped by `categories[i]` and aggregated across seeds.

    Returns:
        dict mapping (ckpt_id, category) -> {metric -> {"mean", "std", "n", "values"}}
    """
    # METRICS already includes accuracy_pass_at_k (derived)
    metrics_with_pass_k = METRICS
    majority_threshold = (pass_k + 1) // 2

    # per_ckpt_cat[ckpt][category][metric] -> list of per-seed means
    per_ckpt_cat: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for ckpt_id, seed_paths in data.items():
        for seed, path in seed_paths.items():
            try:
                result = load_results(path)
            except Exception as e:
                logger.warning(f"Could not read {path}: {e}")
                continue

            acc_rewards = result.get("accuracy_rewards")  # [n_samples][n_gens]
            fmt_rewards = result.get("format_rewards")    # [n_samples][n_gens]
            if not isinstance(acc_rewards, list) or not isinstance(fmt_rewards, list):
                logger.warning(f"{path} missing per-sample reward arrays; skipping per-category aggregation for this file")
                continue

            n_samples = len(acc_rewards)
            if n_samples != len(categories):
                logger.warning(
                    f"{path}: n_samples ({n_samples}) != len(categories) ({len(categories)}); "
                    "category mapping may be misaligned. Skipping this file."
                )
                continue

            # Group per-sample metric values by category
            cat_buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for i in range(n_samples):
                cat = categories[i]
                a_row = acc_rewards[i] or []
                f_row = fmt_rewards[i] or []
                if not a_row:
                    continue
                cat_buckets[cat]["accuracy_at_1"].append(float(a_row[0]))
                cat_buckets[cat]["accuracy_avg"].append(float(np.mean(a_row)))
                n_correct = sum(1 for v in a_row if float(v) > 0.0)
                cat_buckets[cat]["accuracy_majority"].append(1.0 if n_correct >= majority_threshold else 0.0)
                cat_buckets[cat]["accuracy_pass_at_k"].append(1.0 if n_correct >= 1 else 0.0)
                if f_row:
                    cat_buckets[cat]["format_score"].append(float(np.mean(f_row)))

            # For each (category, metric), the per-seed value is the mean over samples in that category
            for cat, mvals in cat_buckets.items():
                for metric in metrics_with_pass_k:
                    if metric in mvals and mvals[metric]:
                        per_ckpt_cat[(ckpt_id, cat)][metric].append(float(np.mean(mvals[metric])))

    # Final aggregation: mean ± std over seeds for each (ckpt, category, metric)
    agg: dict[tuple[str, str], dict] = {}
    for key, mvals in per_ckpt_cat.items():
        agg[key] = {}
        for metric, values in mvals.items():
            if not values:
                continue
            agg[key][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values),
                "values": values,
            }
    return agg


def write_per_category_csv(agg: dict, output_path: Path) -> None:
    """Write per-category aggregated results to CSV."""
    keys = sorted(agg.keys(), key=lambda k: (extract_checkpoint_number(k[0]), k[1]))

    fieldnames = ["checkpoint", "checkpoint_step", "category", "metric", "mean", "std", "n_seeds"] + \
                 [f"seed_{i}_value" for i in range(10)]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for (ckpt_id, cat) in keys:
            step = extract_checkpoint_number(ckpt_id)
            for metric, m in agg[(ckpt_id, cat)].items():
                row = {
                    "checkpoint": ckpt_id,
                    "checkpoint_step": step,
                    "category": cat,
                    "metric": metric,
                    "mean": m["mean"],
                    "std": m["std"],
                    "n_seeds": m["n"],
                }
                for i, v in enumerate(m["values"]):
                    row[f"seed_{i}_value"] = v
                writer.writerow(row)

    logger.info(f"Per-category CSV written to: {output_path}")


def write_figures(agg: dict, output_path: Path) -> None:
    """
    Plot mean ± std for each metric over training checkpoints.

    Produces one figure with subplots for each metric.
    """
    ckpt_ids = sorted(agg.keys(), key=extract_checkpoint_number)
    steps = [extract_checkpoint_number(c) for c in ckpt_ids]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
    if len(METRICS) == 1:
        axes = [axes]

    for ax, metric in zip(axes, METRICS):
        means, stds = [], []
        valid_steps = []
        for ckpt_id, step in zip(ckpt_ids, steps):
            if metric in agg[ckpt_id]:
                means.append(agg[ckpt_id][metric]["mean"])
                stds.append(agg[ckpt_id][metric]["std"])
                valid_steps.append(step)

        if not means:
            ax.set_title(f"{METRIC_LABELS[metric]}\n(no data)")
            continue

        means_arr = np.array(means)
        stds_arr = np.array(stds)

        ax.plot(valid_steps, means_arr, marker="o", linewidth=1.5, markersize=4)
        ax.fill_between(
            valid_steps,
            means_arr - stds_arr,
            means_arr + stds_arr,
            alpha=0.25,
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Training step (checkpoint)")
        ax.set_ylabel("Score")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate BeamRL evaluation results across seeds")
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Root output directory (parent of beamrl_eval/)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
        help="Seeds to aggregate (default: 42 123 456)"
    )
    parser.add_argument(
        "--pattern", type=str, default="",
        help="Optional glob pattern to filter checkpoint dirs (e.g. '*beamrl_260101*')"
    )
    parser.add_argument(
        "--out_csv", type=str, default=None,
        help="Path for CSV output (default: {output_dir}/aggregated_results.csv)"
    )
    parser.add_argument(
        "--out_fig", type=str, default=None,
        help="Path for figure output (default: {output_dir}/aggregated_results.png)"
    )
    parser.add_argument(
        "--per_category", action="store_true",
        help="Also compute per-category mean ± std and write aggregated_per_category.csv. "
             "Requires --eval_dataset (HuggingFace dataset name) to look up the `category` column."
    )
    parser.add_argument(
        "--eval_dataset", type=str, default="tphage/BeamRL-EvalData-v2",
        help="HuggingFace dataset name used during evaluation, for per-category labels "
             "(default: tphage/BeamRL-EvalData-v2). Only used when --per_category is set."
    )
    parser.add_argument(
        "--eval_split", type=str, default="train",
        help="Dataset split for per-category labels (default: train)."
    )
    parser.add_argument(
        "--categories_json", type=str, default=None,
        help="Local JSON file with per-sample category labels (offline alternative to "
             "downloading the eval dataset). Accepts {\"categories\": [...]} or a bare list. "
             "Use this to bypass the pyarrow-19 parquet bug on the v2 dataset."
    )
    parser.add_argument(
        "--pass_k", type=int, default=PASS_K_DEFAULT,
        help=f"k for Pass@k and Majority@k (default: {PASS_K_DEFAULT})."
    )
    parser.add_argument(
        "--out_per_category_csv", type=str, default=None,
        help="Path for per-category CSV (default: {output_dir}/aggregated_per_category.csv)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    out_csv = Path(args.out_csv) if args.out_csv else output_dir / "aggregated_results.csv"
    out_fig = Path(args.out_fig) if args.out_fig else output_dir / "aggregated_results.png"
    out_per_cat = Path(args.out_per_category_csv) if args.out_per_category_csv \
        else output_dir / "aggregated_per_category.csv"

    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Pattern filter: '{args.pattern}' (empty = no filter)")

    data = find_result_files(output_dir, args.seeds, args.pattern)
    if not data:
        logger.error("No result files found. Check --output_dir and --seeds.")
        return

    logger.info(f"Found {len(data)} checkpoint(s) with result files.")

    agg = aggregate(data)
    print_summary(agg)
    write_csv(agg, out_csv)
    write_figures(agg, out_fig)

    if args.per_category:
        if args.categories_json:
            categories = load_eval_categories_from_json(Path(args.categories_json))
        else:
            categories = load_eval_categories(args.eval_dataset, args.eval_split)
            logger.info(f"Loaded {len(categories)} category labels from {args.eval_dataset}.")
        per_cat_agg = aggregate_per_category(data, categories, pass_k=args.pass_k)
        if per_cat_agg:
            write_per_category_csv(per_cat_agg, out_per_cat)
        else:
            logger.warning("Per-category aggregation produced no rows; check inputs.")


if __name__ == "__main__":
    main()
