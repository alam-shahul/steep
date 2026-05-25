import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark evaluation outputs into a flat CSV table.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing benchmark evaluation outputs.",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to the output CSV summary.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Optional output path for the retention-ratio vs NMI plot.",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default="original_leiden",
        help="Label key whose NMI should be plotted.",
    )
    parser.add_argument(
        "--gallery-path",
        type=Path,
        default=None,
        help="Optional output path for the in-situ clustering gallery plot.",
    )
    parser.add_argument(
        "--gallery-seed",
        type=int,
        default=0,
        help="Random seed whose runs should be shown in the gallery plot.",
    )
    return parser.parse_args()


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, prefix=full_key))
        else:
            flattened[full_key] = value
    return flattened


def build_row(evaluation: dict[str, Any], json_path: Path) -> dict[str, Any]:
    trainer_cfg = OmegaConf.load(Path(evaluation["results_folder"]) / "config.txt")
    trainer_cfg = OmegaConf.to_container(
        trainer_cfg,
        resolve=True,
    )
    dataset_cfg = trainer_cfg["dataset"]
    sketcher_cfg = trainer_cfg.get("sketcher", {})
    model_cfg = trainer_cfg["model"]
    has_sketcher = bool(sketcher_cfg)

    return {
        "evaluation_json": str(json_path),
        "evaluation_directory": str(json_path.parent),
        "condition": "sketch" if has_sketcher else "baseline",
        "dataset.name": dataset_cfg.get("name"),
        "dataset.data_directory": ((dataset_cfg.get("args") or {}).get("data_directory")),
        "model.type": model_cfg.get("type"),
        "sketcher.type": sketcher_cfg.get("type") if has_sketcher else None,
        **flatten_dict(sketcher_cfg, prefix="sketcher"),
        **flatten_dict(evaluation, prefix="evaluation"),
    }


def load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for json_path in sorted(input_dir.rglob("evaluation.json")):
        with open(json_path) as f:
            data = json.load(f)
        rows.append(build_row(data, json_path))

    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _short_method_name(method: str) -> str:
    return method.rsplit(".", 1)[-1]


def _group_metric_by_method_and_ratio(
    rows: list[dict[str, Any]],
    metric_key: str,
) -> dict[str, dict[float, list[float]]]:
    grouped_scores: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        method = row.get("sketcher.type")
        retention_ratio = row.get("sketcher.args.retention_ratio")
        metric_value = row.get(metric_key)
        if not method or retention_ratio in (None, "") or metric_value in (None, ""):
            continue
        grouped_scores[method][float(retention_ratio)].append(float(metric_value))

    return grouped_scores


def _group_total_epoch_time_by_method_and_ratio(rows: list[dict[str, Any]]) -> dict[str, dict[float, list[float]]]:
    grouped_scores: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        method = row.get("sketcher.type")
        retention_ratio = row.get("sketcher.args.retention_ratio")
        epoch_times = row.get("evaluation.epoch_times_seconds")
        if not method or retention_ratio in (None, "") or epoch_times in (None, ""):
            continue

        if isinstance(epoch_times, str):
            epoch_times = json.loads(epoch_times)

        total_epoch_time = float(sum(float(value) for value in epoch_times))
        grouped_scores[method][float(retention_ratio)].append(total_epoch_time)

    return grouped_scores


def _plot_metric_panel(ax, grouped_scores: dict[str, dict[float, list[float]]], ylabel: str, title: str) -> None:
    for method, scores_by_ratio in sorted(grouped_scores.items()):
        ratios = sorted(scores_by_ratio)
        means = [float(np.mean(scores_by_ratio[ratio])) for ratio in ratios]
        stds = [float(np.std(scores_by_ratio[ratio])) for ratio in ratios]
        ax.errorbar(
            ratios,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=_short_method_name(method),
        )

    ax.set_xlabel("Retention Ratio")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)


def write_plot(rows: list[dict[str, Any]], output_path: Path, label_key: str) -> None:
    metric_key = f"evaluation.metrics_by_label_key.{label_key}.nmi_mean"
    nmi_scores = _group_metric_by_method_and_ratio(rows, metric_key)
    gpu_memory_scores = {
        method: {ratio: [value / (1024**3) for value in values] for ratio, values in scores_by_ratio.items()}
        for method, scores_by_ratio in _group_metric_by_method_and_ratio(
            rows,
            "evaluation.peak_gpu_memory_bytes",
        ).items()
    }
    epoch_runtime_scores = _group_total_epoch_time_by_method_and_ratio(rows)

    if not nmi_scores:
        raise ValueError(
            f"No sketch rows with retention ratios and `{metric_key}` were found for plotting.",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _plot_metric_panel(
        axes[0],
        nmi_scores,
        ylabel=f"NMI ({label_key})",
        title="Retention Ratio vs NMI",
    )
    _plot_metric_panel(
        axes[1],
        gpu_memory_scores,
        ylabel="Peak GPU Memory (GiB)",
        title="Retention Ratio vs GPU Memory",
    )
    _plot_metric_panel(
        axes[2],
        epoch_runtime_scores,
        ylabel="Total Epoch Runtime (s)",
        title="Retention Ratio vs Total Epoch Runtime",
    )
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _load_plot_image(path: Path):
    return plt.imread(path)


def _get_plot_paths(row: dict[str, Any]) -> dict[str, Path]:
    plot_dir = Path(row["evaluation_directory"]) / "plots"
    return {path.name: path for path in sorted(plot_dir.glob("*.png"))}


def write_gallery_plot(
    rows: list[dict[str, Any]],
    output_path: Path,
    seed: int,
) -> None:
    sketch_rows = [
        row
        for row in rows
        if row.get("condition") == "sketch" and str(row.get("sketcher.args.random_seed")) == str(seed)
    ]
    if not sketch_rows:
        raise ValueError(f"No sketch rows found for gallery seed {seed}.")

    methods = sorted({row["sketcher.type"] for row in sketch_rows})
    ratios = sorted({float(row["sketcher.args.retention_ratio"]) for row in sketch_rows})
    baseline_rows = [row for row in rows if row.get("condition") == "baseline"]

    selected_rows = []
    for method in methods:
        method_rows = {}
        for ratio in ratios:
            matches = [
                row
                for row in sketch_rows
                if row["sketcher.type"] == method and float(row["sketcher.args.retention_ratio"]) == ratio
            ]
            if not matches:
                raise ValueError(f"Missing gallery row for method={method}, ratio={ratio}, seed={seed}.")
            method_rows[ratio] = matches[0]
        selected_rows.append((method, method_rows))

    common_plot_names = None
    for _, method_rows in selected_rows:
        for row in method_rows.values():
            plot_names = set(_get_plot_paths(row))
            common_plot_names = plot_names if common_plot_names is None else common_plot_names & plot_names
    if baseline_rows:
        baseline_plot_names = set(_get_plot_paths(baseline_rows[0]))
        common_plot_names = (
            baseline_plot_names if common_plot_names is None else common_plot_names & baseline_plot_names
        )
    if not common_plot_names:
        raise ValueError("Could not find a common slide plot across selected benchmark runs.")

    slide_name = sorted(common_plot_names)[0]
    reference_row = (
        baseline_rows[0]
        if baseline_rows
        else max(
            sketch_rows,
            key=lambda row: float(row["sketcher.args.retention_ratio"]),
        )
    )
    reference_plot = _get_plot_paths(reference_row)[slide_name]
    reference_title = "Original Leiden" if baseline_rows else "Reference Leiden"

    num_rows = len(methods)
    num_cols = len(ratios) + 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2.2 * num_cols, 2.6 * num_rows))
    axes = np.atleast_2d(axes)

    for row_idx, (method, method_rows) in enumerate(selected_rows):
        axes[row_idx, 0].imshow(_load_plot_image(reference_plot))
        axes[row_idx, 0].set_title(reference_title)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].text(
            -0.08,
            0.5,
            _short_method_name(method),
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10,
        )

        for col_idx, ratio in enumerate(ratios, start=1):
            plot_path = _get_plot_paths(method_rows[ratio])[slide_name]
            axes[row_idx, col_idx].imshow(_load_plot_image(plot_path))
            axes[row_idx, col_idx].set_title(f"r={ratio:g}")
            axes[row_idx, col_idx].axis("off")

    fig.subplots_adjust(wspace=0.02, hspace=0.08)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_dir)
    if not rows:
        raise FileNotFoundError(f"No benchmark evaluation JSON files found under {args.input_dir}.")

    write_csv(rows, args.output_path)
    logger.info("Wrote {} rows to {}", len(rows), args.output_path)
    plot_path = args.plot_path or args.output_path.with_suffix(".png")
    write_plot(rows, plot_path, label_key=args.label_key)
    logger.info("Wrote plot to {}", plot_path)
    gallery_path = args.gallery_path or args.output_path.with_name(f"{args.output_path.stem}_gallery.png")
    write_gallery_plot(rows, gallery_path, seed=args.gallery_seed)
    logger.info("Wrote gallery plot to {}", gallery_path)


if __name__ == "__main__":
    main()
