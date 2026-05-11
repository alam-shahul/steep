#!/usr/bin/env python

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RECORDS_DIR = Path("/work/magroup/xinyuelu/steep/experiments/zhuang_subset/records")
DEFAULT_EVAL_ROOT = Path("/work/magroup/shared/steep/evaluations")


EVALUATION_JSON_RE = re.compile(
    r'"evaluation_json"\s*:\s*"([^"]+evaluation\.json)"',
)

CHECKPOINT_RE = re.compile(
    r"Checkpoint directory initialized at\s+(.*/checkpoints/([A-Za-z0-9\-]+))",
)


LOG_NAME_RE = re.compile(
    r"run_r(?P<ratio>[0-9]+(?:[.p][0-9]+)?)_s(?P<seed>[0-9]+)\.log$",
)


EDGE_METHOD_NAME_TO_TYPE = {
    "mog": "steep.sketcher.MoGSketcher",
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
        description="Build benchmark summary and separate edge/node plots from log files.",
    )

    parser.add_argument(
        "--records-dir",
        type=Path,
        default=DEFAULT_RECORDS_DIR,
        help="Directory containing method folders with run_r*_s*.log files.",
    )

    parser.add_argument(
        "--eval-root",
        type=Path,
        default=DEFAULT_EVAL_ROOT,
        help=(
            "Fallback root directory containing evaluation folders. "
            "Used only when evaluation_json is not found in the log."
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output all-method summary CSV path.",
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
        help="Label key for NMI plot.",
    )

    parser.add_argument(
        "--latest-only",
        action="store_true",
        help=(
            "If multiple logs exist for the same method/ratio/seed, keep the one whose "
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


def parse_ratio_seed_from_log_name(log_path: Path) -> tuple[float, int] | None:
    match = LOG_NAME_RE.search(log_path.name)
    if match is None:
        return None

    ratio_text = match.group("ratio").replace("p", ".")
    ratio = float(ratio_text)
    seed = int(match.group("seed"))

    return ratio, seed


def read_log_text(log_path: Path) -> str | None:
    try:
        return log_path.read_text(errors="ignore")
    except OSError:
        return None


def parse_evaluation_json_from_log(log_path: Path) -> Path | None:
    text = read_log_text(log_path)
    if text is None:
        return None

    matches = EVALUATION_JSON_RE.findall(text)
    if not matches:
        return None

    return Path(matches[-1])


def parse_checkpoint_id_from_log(log_path: Path) -> str | None:
    text = read_log_text(log_path)
    if text is None:
        return None

    matches = CHECKPOINT_RE.findall(text)
    if not matches:
        return None

    return matches[-1][1]


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


def fallback_eval_json_from_checkpoint(
    log_path: Path,
    eval_root: Path,
) -> tuple[Path | None, str | None]:
    checkpoint_id = parse_checkpoint_id_from_log(log_path)

    if checkpoint_id is None:
        return None, None

    return eval_root / checkpoint_id / "evaluation.json", checkpoint_id


def build_row_from_log(
    log_path: Path,
    eval_root: Path,
) -> dict[str, Any] | None:
    method_name = log_path.parent.name

    method_type = method_name_to_type(method_name)
    method_group = method_name_to_group(method_name)

    if method_type is None or method_group is None:
        return None

    ratio_seed = parse_ratio_seed_from_log_name(log_path)
    if ratio_seed is None:
        print(f"Skipping {log_path}: cannot parse retention ratio / seed from log name.")
        return None

    retention_ratio, random_seed = ratio_seed

    eval_json = parse_evaluation_json_from_log(log_path)
    checkpoint_id = parse_checkpoint_id_from_log(log_path)

    if eval_json is None:
        eval_json, fallback_checkpoint_id = fallback_eval_json_from_checkpoint(log_path, eval_root)
        if checkpoint_id is None:
            checkpoint_id = fallback_checkpoint_id

    if eval_json is None:
        print(f"Skipping {log_path}: neither evaluation_json nor checkpoint id found in log.")
        return None

    evaluation = load_evaluation_json(eval_json)
    if evaluation is None:
        print(f"Skipping {log_path}: evaluation.json not found or invalid at {eval_json}.")
        return None

    short_name = short_method_name_from_folder(method_name)

    row = {
        "method_folder": method_name,
        "method_group": method_group,
        "log_path": str(log_path),
        "checkpoint_id": checkpoint_id,
        "evaluation_json": str(eval_json),
        "evaluation_directory": str(eval_json.parent),
        "condition": "sketch",
        "sketcher.type": method_type,
        "sketcher.short_name": short_name,
        "sketcher.args.retention_ratio": retention_ratio,
        "sketcher.args.random_seed": random_seed,
        **flatten_dict(evaluation, prefix="evaluation"),
    }

    return row


def collect_rows(
    records_dir: Path,
    eval_root: Path,
    latest_only: bool,
) -> list[dict[str, Any]]:
    rows = []

    for method_dir in sorted(records_dir.iterdir()):
        if not method_dir.is_dir():
            continue

        method_name = method_dir.name

        if method_name_to_type(method_name) is None:
            continue

        for log_path in sorted(method_dir.glob("run_r*_s*.log")):
            row = build_row_from_log(
                log_path=log_path,
                eval_root=eval_root,
            )
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
            int(row["sketcher.args.random_seed"]),
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
            int(r["sketcher.args.random_seed"]),
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
    ylabel: str,
    title: str,
    group: str,
    xlabel: str = "Retention Ratio",
) -> None:
    for method in sorted(scores.keys(), key=lambda m: method_rank(m, group)):
        xs = sorted(scores[method].keys())
        means = [float(np.mean(scores[method][x])) for x in xs]
        stds = [float(np.std(scores[method][x])) for x in xs]

        color = method_color(method, group)
        emphasize = method in {"mog", "mog_node"}

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
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.2)


def write_plot(
    rows: list[dict[str, Any]],
    output_path: Path,
    label_key: str,
    group: str,
) -> None:
    nmi_key = f"evaluation.metrics_by_label_key.{label_key}.nmi_mean"

    nmi_scores = group_metric_by_method_and_x(rows, nmi_key)

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

    if not nmi_scores:
        available_nmi_cols = sorted(
            {
                key
                for row in rows
                for key in row
                if key.startswith("evaluation.metrics_by_label_key.") and key.endswith(".nmi_mean")
            },
        )

        msg = f"No valid rows found for `{nmi_key}` in group `{group}`.\n" f"Available NMI columns include:\n"

        if available_nmi_cols:
            msg += "\n".join(available_nmi_cols[:50])
        else:
            msg += "None"

        raise ValueError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4))

    plot_panel(
        axes[0],
        nmi_scores,
        ylabel=f"NMI ({label_key})",
        title=f"{group} NMI",
        group=group,
    )

    plot_panel(
        axes[1],
        gpu_memory_scores,
        ylabel="GPU Memory (GiB)",
        title=f"{group} Memory",
        group=group,
    )

    plot_panel(
        axes[2],
        runtime_scores,
        ylabel="Runtime (s)",
        title=f"{group} Runtime",
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

    plt.savefig(output_path, dpi=250, bbox_inches="tight")
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

    print("\nCollected runs:")

    for group in ["edge", "node"]:
        if group not in counter:
            continue

        print(f"\n[{group}]")
        order = group_method_order(group)

        for method in sorted(
            counter[group].keys(),
            key=lambda x: order.index(x) if x in order else 999,
        ):
            parts = [f"r={ratio:g}: n={counter[group][method][ratio]}" for ratio in sorted(counter[group][method])]
            print(f"  {method}: " + ", ".join(parts))


def main() -> None:
    args = parse_args()

    rows = collect_rows(
        records_dir=args.records_dir,
        eval_root=args.eval_root,
        latest_only=args.latest_only,
    )

    if not rows:
        raise FileNotFoundError(
            f"No valid rows found under {args.records_dir}.",
        )

    write_csv(rows, args.output_csv)
    print(f"Wrote all rows to {args.output_csv}")

    plot_prefix = args.plot_prefix or args.output_csv.with_suffix("")

    requested_groups = ["edge", "node"] if "all" in args.groups else args.groups

    for group in requested_groups:
        group_rows = [row for row in rows if row.get("method_group") == group]

        if not group_rows:
            print(f"Skipping group `{group}`: no rows.")
            continue

        group_csv = plot_prefix.with_name(f"{plot_prefix.name}_{group}.csv")
        group_plot = plot_prefix.with_name(f"{plot_prefix.name}_{group}.png")

        write_csv(group_rows, group_csv)
        print(f"Wrote {group} rows to {group_csv}")

        write_plot(
            group_rows,
            group_plot,
            label_key=args.label_key,
            group=group,
        )
        print(f"Wrote {group} plot to {group_plot}")

    print_summary(rows)


if __name__ == "__main__":
    main()
