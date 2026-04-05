import argparse
import csv
import json
from pathlib import Path
from typing import Any

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


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_dir)
    if not rows:
        raise FileNotFoundError(f"No benchmark evaluation JSON files found under {args.input_dir}.")

    write_csv(rows, args.output_path)
    print(f"Wrote {len(rows)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
