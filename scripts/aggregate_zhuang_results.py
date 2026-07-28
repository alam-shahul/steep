#!/usr/bin/env python

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize_config_dir
from loguru import logger

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

CONFIG_DIR = Path(__file__).resolve().parents[1] / "steep" / "config"
DEFAULT_OUTPUT_CSV = Path("results") / "zhuang_summary.csv"


def default_eval_root() -> Path:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config")
    return Path(cfg.cache_dir) / "evaluations"


EDGE_METHOD_NAME_TO_TYPE = {
    "mog": "steep.sketcher.MoGSketcher",
    "mog_edge": "steep.sketcher.MoGSketcher",
    "edge_random": "steep.sketcher.RandomEdgeSketcher",
    "edge_spatialshort": "steep.sketcher.SpatialShortEdgeSketcher",
    "edge_spatiallong": "steep.sketcher.SpatialLongEdgeSketcher",
    "edge_exprsim": "steep.sketcher.ExpressionSimilarityEdgeSketcher",
    "edge_hybrid": "steep.sketcher.HybridSpatialExpressionEdgeSketcher",
}


NODE_METHOD_NAME_TO_TYPE = {
    "mog_node": "steep.sketcher.MoGSketcher",
    "geosketch": "steep.sketcher.GeoSketcher",
    "hopper": "steep.sketcher.HopperSketcher",
    "jointhopper": "steep.sketcher.JointHopperSketcher",
    "leverage_score_sampling": "steep.sketcher.LeverageScoreSketcher",
    "random_subsample": "steep.sketcher.RandomSubsampleSketcher",
    "spatialhopper": "steep.sketcher.SpatialHopperSketcher",
}


METHOD_NAME_TO_TYPE = {
    **EDGE_METHOD_NAME_TO_TYPE,
    **NODE_METHOD_NAME_TO_TYPE,
}


METHOD_GROUP = {
    **{name: "edge" for name in EDGE_METHOD_NAME_TO_TYPE},
    **{name: "node" for name in NODE_METHOD_NAME_TO_TYPE},
}


METHOD_NAME_DISPLAY = {
    # edge-based
    "mog": "mog",
    "mog_edge": "mog_edge",
    "edge_random": "edge_random",
    "edge_spatialshort": "edge_spatialshort",
    "edge_spatiallong": "edge_spatiallong",
    "edge_exprsim": "edge_exprsim",
    "edge_hybrid": "edge_hybrid",
    # node-based
    "mog_node": "mog_node",
    "geosketch": "geosketch",
    "hopper": "hopper",
    "jointhopper": "jointhopper",
    "leverage_score_sampling": "leverage",
    "random_subsample": "random_subsample",
    "spatialhopper": "spatialhopper",
}


EDGE_METHOD_ORDER = [
    "mog_edge",
    "mog",
    "edge_random",
    "edge_spatialshort",
    "edge_spatiallong",
    "edge_exprsim",
    "edge_hybrid",
]


NODE_METHOD_ORDER = [
    "mog_node",
    "random_subsample",
    "geosketch",
    "hopper",
    "spatialhopper",
    "jointhopper",
    "leverage",
]


EDGE_METHOD_PALETTE = {
    "mog": "#D55E00",
    "mog_edge": "#D55E00",
    "edge_random": "#7A7A7A",
    "edge_spatialshort": "#0072B2",
    "edge_spatiallong": "#56B4E9",
    "edge_exprsim": "#009E73",
    "edge_hybrid": "#CC79A7",
}


NODE_METHOD_PALETTE = {
    "mog_node": "#D55E00",
    "random_subsample": "#7A7A7A",
    "geosketch": "#E69F00",
    "hopper": "#0072B2",
    "spatialhopper": "#56B4E9",
    "jointhopper": "#009E73",
    "leverage": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build benchmark summary and separate edge/node plots from evaluation config/metric files.",
    )

    parser.add_argument(
        "--eval-root",
        type=Path,
        default=default_eval_root(),
        help="Root directory containing evaluation folders with config.json and evaluation.json.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output all-method summary CSV path. Default: {DEFAULT_OUTPUT_CSV}",
    )

    parser.add_argument(
        "--plot-prefix",
        type=Path,
        default=None,
        help=("Output prefix for plots and split CSV files. " "Default: output CSV path without suffix."),
    )

    parser.add_argument(
        "--label-key",
        type=str,
        default="original_leiden",
        help="Label key for the default NMI plot. Ignored when --metric-key is provided.",
    )

    parser.add_argument(
        "--metric-key",
        type=str,
        default=None,
        help=(
            "Flattened evaluation metric key for the first plot panel, for example "
            "evaluation.classification_metrics_by_label_key.cell_type_coarse.accuracy. "
            "Default: evaluation.metrics_by_label_key.<label-key>.nmi_mean."
        ),
    )

    parser.add_argument(
        "--metric-label",
        type=str,
        default=None,
        help="Y-axis label for --metric-key. Default: the metric key suffix.",
    )

    parser.add_argument(
        "--latest-only",
        action="store_true",
        help=(
            "If multiple evaluations exist for the same method/ratio/seed, keep the one whose "
            "evaluation.json has the newest modification time."
        ),
    )

    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["edge", "node", "all"],
        default=["edge", "node"],
        help="Which plots to write.",
    )

    return parser.parse_args()


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat = {}

    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            flat.update(flatten_dict(value, prefix=full_key))
        else:
            flat[full_key] = value

    return flat


def method_name_to_type(method_name: str) -> str | None:
    return METHOD_NAME_TO_TYPE.get(method_name)


def method_name_to_group(method_name: str) -> str | None:
    return METHOD_GROUP.get(method_name)


def short_method_name_from_folder(method_name: str) -> str:
    return METHOD_NAME_DISPLAY.get(method_name, method_name)


def group_method_order(group: str) -> list[str]:
    if group == "edge":
        return EDGE_METHOD_ORDER
    if group == "node":
        return NODE_METHOD_ORDER
    return EDGE_METHOD_ORDER + NODE_METHOD_ORDER


def group_method_palette(group: str) -> dict[str, str]:
    if group == "edge":
        return EDGE_METHOD_PALETTE
    if group == "node":
        return NODE_METHOD_PALETTE
    return {**EDGE_METHOD_PALETTE, **NODE_METHOD_PALETTE}


def method_rank(method_name: str, group: str) -> int:
    order = group_method_order(group)
    return order.index(method_name) if method_name in order else len(order)


def method_color(method_name: str, group: str) -> str:
    palette = group_method_palette(group)
    return palette.get(method_name, "#333333")


def method_name_from_sketcher_config(sketcher_config: dict[str, Any]) -> str | None:
    sketcher_type = sketcher_config.get("type")
    args = sketcher_config.get("args") or {}

    if sketcher_type == "steep.sketcher.MoGSketcher":
        sketch_mode = args.get("sketch_mode")
        if sketch_mode == "edge":
            return "mog_edge"
        if sketch_mode == "node":
            return "mog_node"
        return "mog"

    for method_name, method_type in METHOD_NAME_TO_TYPE.items():
        if method_type == sketcher_type:
            return method_name

    return None


def retention_ratio_from_sketcher_config(sketcher_config: dict[str, Any]) -> float | None:
    args = sketcher_config.get("args") or {}
    value = args.get("retention_ratio")
    if value is None:
        value = (args.get("mog_args") or {}).get("retention_ratio")
    return None if value is None else float(value)


def load_evaluation_json(eval_json: Path) -> dict[str, Any] | None:
    if not eval_json.exists():
        return None

    try:
        with open(eval_json) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None
    except OSError:
        return None


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None
    except OSError:
        return None


def build_row_from_config(config_path: Path) -> dict[str, Any] | None:
    config = load_json(config_path)
    if config is None or config.get("condition") != "sketch":
        return None

    input_config = config.get("input_config") or {}
    sketcher_config = input_config.get("sketcher") or {}
    method_name = method_name_from_sketcher_config(sketcher_config)
    if method_name is None:
        return None

    method_type = method_name_to_type(method_name)
    method_group = method_name_to_group(method_name)
    retention_ratio = retention_ratio_from_sketcher_config(sketcher_config)
    random_seed = input_config.get("seed")
    random_seed = None if random_seed is None else int(random_seed)

    if method_type is None or method_group is None or retention_ratio is None or random_seed is None:
        return None

    eval_json = config_path.with_name("evaluation.json")
    evaluation = load_evaluation_json(eval_json)
    if evaluation is None:
        logger.warning("Skipping {}: evaluation.json not found or invalid at {}", config_path, eval_json)
        return None

    short_name = short_method_name_from_folder(method_name)

    return {
        "method_folder": method_name,
        "method_group": method_group,
        "config_json": str(config_path),
        "evaluation_json": str(eval_json),
        "evaluation_directory": str(eval_json.parent),
        "condition": "sketch",
        "sketcher.type": method_type,
        "sketcher.short_name": short_name,
        "sketcher.args.retention_ratio": retention_ratio,
        "seed": random_seed,
        **flatten_dict(evaluation, prefix="evaluation"),
    }


def collect_rows(
    eval_root: Path,
    latest_only: bool,
) -> list[dict[str, Any]]:
    rows = []

    for config_path in sorted(eval_root.glob("*/config.json")):
        row = build_row_from_config(config_path)
        if row is not None:
            rows.append(row)

    if latest_only:
        rows = deduplicate_rows_by_latest_evaluation(rows)

    return rows


def deduplicate_rows_by_latest_evaluation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, float, int], dict[str, Any]] = {}

    for row in rows:
        key = (
            row["method_group"],
            row["sketcher.short_name"],
            float(row["sketcher.args.retention_ratio"]),
            int(row["seed"]),
        )

        eval_json = Path(row["evaluation_json"])

        try:
            mtime = eval_json.stat().st_mtime
        except OSError:
            mtime = 0.0

        old = best.get(key)

        if old is None:
            best[key] = row
            continue

        old_eval_json = Path(old["evaluation_json"])

        try:
            old_mtime = old_eval_json.stat().st_mtime
        except OSError:
            old_mtime = 0.0

        if mtime > old_mtime:
            best[key] = row

    return sorted(
        best.values(),
        key=lambda r: (
            r["method_group"],
            method_rank(r["sketcher.short_name"], r["method_group"]),
            float(r["sketcher.args.retention_ratio"]),
            int(r["seed"]),
        ),
    )


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def get_retention_ratio(row: dict[str, Any]) -> float | None:
    value = row.get("sketcher.args.retention_ratio")

    if value not in (None, ""):
        return float(value)

    value = row.get("sketcher.args.mog_args.retention_ratio")

    if value not in (None, ""):
        return float(value)

    return None


def group_metric_by_method_and_x(
    rows: list[dict[str, Any]],
    metric_key: str,
) -> dict[str, dict[float, list[float]]]:
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        method = row.get("sketcher.short_name")
        x = get_retention_ratio(row)
        y = row.get(metric_key)

        if method in (None, "") or x is None or y in (None, ""):
            continue

        try:
            grouped[str(method)][float(x)].append(float(y))
        except (TypeError, ValueError):
            continue

    return grouped


def plot_panel(
    ax,
    scores: dict[str, dict[float, list[float]]],
    title: str,
    group: str,
    xlabel: str = "Retention Ratio",
) -> None:
    for method in sorted(scores.keys(), key=lambda m: method_rank(m, group)):
        xs = sorted(scores[method].keys())
        means = [float(np.mean(scores[method][x])) for x in xs]
        stds = [float(np.std(scores[method][x])) for x in xs]

        color = method_color(method, group)
        emphasize = method in {"mog", "mog_edge", "mog_node"}

        ax.errorbar(
            xs,
            means,
            yerr=stds,
            label=method,
            color=color,
            marker="o",
            markersize=4.5 if emphasize else 3.5,
            linewidth=3.0 if emphasize else 1.8,
            capsize=3,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=14)
    ax.grid(alpha=0.2)


def write_plot(
    rows: list[dict[str, Any]],
    output_path: Path,
    label_key: str,
    metric_key: str | None,
    metric_label: str | None,
    group: str,
) -> None:
    score_key = metric_key or f"evaluation.metrics_by_label_key.{label_key}.nmi_mean"
    score_label = metric_label or ("NMI" if metric_key is None else score_key.split(".")[-1])

    score_values = group_metric_by_method_and_x(rows, score_key)

    gpu_memory_scores_raw = group_metric_by_method_and_x(
        rows,
        "evaluation.peak_gpu_memory_bytes",
    )

    gpu_memory_scores = {
        method: {x: [value / (1024**3) for value in values] for x, values in scores_by_x.items()}
        for method, scores_by_x in gpu_memory_scores_raw.items()
    }

    runtime_scores = group_metric_by_method_and_x(
        rows,
        "evaluation.epoch_time_mean_seconds",
    )

    if not score_values:
        available_metric_cols = sorted(
            {
                key
                for row in rows
                for key in row
                if key.startswith("evaluation.") and isinstance(row.get(key), int | float | str)
            },
        )

        msg = f"No valid rows found for `{score_key}` in group `{group}`.\n" f"Available metric columns include:\n"

        if available_metric_cols:
            msg += "\n".join(available_metric_cols[:75])
        else:
            msg += "None"

        raise ValueError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4))

    plot_panel(
        axes[0],
        score_values,
        title=score_label,
        group=group,
    )

    plot_panel(
        axes[1],
        gpu_memory_scores,
        title="GPU Memory (GiB)",
        group=group,
    )

    plot_panel(
        axes[2],
        runtime_scores,
        title="Runtime (s)",
        group=group,
    )

    handles, labels = axes[0].get_legend_handles_labels()

    order = group_method_order(group)
    idx_order = sorted(
        range(len(labels)),
        key=lambda i: order.index(labels[i]) if labels[i] in order else len(order),
    )

    handles = [handles[i] for i in idx_order]
    labels = [labels[i] for i in idx_order]

    fig.legend(
        handles,
        labels,
        title="Sketcher",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )

    fig.subplots_adjust(wspace=0.25, right=0.82)

    pdf_output_path = output_path.with_suffix(".pdf")

    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.savefig(pdf_output_path, bbox_inches="tight")
    plt.close(fig)


def print_summary(rows: list[dict[str, Any]]) -> None:
    counter: dict[str, dict[str, dict[float, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)),
    )

    for row in rows:
        group = row["method_group"]
        method = row["sketcher.short_name"]
        ratio = float(row["sketcher.args.retention_ratio"])
        counter[group][method][ratio] += 1

    logger.info("Collected runs:")

    for group in ["edge", "node"]:
        if group not in counter:
            continue

        logger.info("[{}]", group)
        order = group_method_order(group)

        for method in sorted(
            counter[group].keys(),
            key=lambda x: order.index(x) if x in order else 999,
        ):
            parts = [f"r={ratio:g}: n={counter[group][method][ratio]}" for ratio in sorted(counter[group][method])]
            logger.info("{}: {}", method, ", ".join(parts))


def main() -> None:
    args = parse_args()

    rows = collect_rows(
        eval_root=args.eval_root,
        latest_only=args.latest_only,
    )

    if not rows:
        raise FileNotFoundError(
            f"No valid rows found under {args.eval_root}.",
        )

    write_csv(rows, args.output_csv)
    logger.info("Wrote all rows to {}", args.output_csv)

    plot_prefix = args.plot_prefix or args.output_csv.with_suffix("")

    requested_groups = ["edge", "node"] if "all" in args.groups else args.groups

    for group in requested_groups:
        group_rows = [row for row in rows if row.get("method_group") == group]

        if not group_rows:
            logger.warning("Skipping group `{}`: no rows", group)
            continue

        group_csv = plot_prefix.with_name(f"{plot_prefix.name}_{group}.csv")
        group_plot = plot_prefix.with_name(f"{plot_prefix.name}_{group}.png")

        write_csv(group_rows, group_csv)
        logger.info("Wrote {} rows to {}", group, group_csv)

        write_plot(
            group_rows,
            group_plot,
            label_key=args.label_key,
            metric_key=args.metric_key,
            metric_label=args.metric_label,
            group=group,
        )
        logger.info("Wrote {} plot to {} and {}", group, group_plot, group_plot.with_suffix(".pdf"))

    print_summary(rows)


if __name__ == "__main__":
    main()
