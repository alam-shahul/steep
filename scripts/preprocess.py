import argparse
import os
import random
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import anndata as ad
from loguru import logger
from tqdm import tqdm

from steep.utils import preprocess


def _configure_warnings() -> None:
    warnings.simplefilter("ignore")


def _preprocess_file(input_path: Path, output_path: Path) -> None:
    _configure_warnings()
    adata = ad.read_h5ad(input_path)
    preprocess(adata)
    adata.write_h5ad(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize SRT .h5ad files and compute spatial neighbor graphs.",
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing input .h5ad files.")
    parser.add_argument("output_dir", type=Path, help="Directory where processed .h5ad files will be written.")
    parser.add_argument(
        "--num-slides",
        type=int,
        default=None,
        help="Randomly sample this many .h5ad files from the input directory before preprocessing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when `--num-slides` is provided.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes to use. Defaults to all available CPU cores.",
    )
    return parser


def main() -> None:
    _configure_warnings()
    args = build_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist or is not a directory: {input_dir}")

    input_paths = sorted(input_dir.glob("*.h5ad"))
    if not input_paths:
        raise FileNotFoundError(f"No .h5ad files found in input directory: {input_dir}")
    if args.num_slides is not None:
        if args.num_slides <= 0:
            raise ValueError("`--num-slides` must be a positive integer.")
        if args.num_slides > len(input_paths):
            raise ValueError(
                f"`--num-slides` ({args.num_slides}) cannot exceed the number of available .h5ad files "
                f"({len(input_paths)}).",
            )
        rng = random.Random(args.seed)
        input_paths = sorted(rng.sample(input_paths, k=args.num_slides))

    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(input_path, output_dir / input_path.name) for input_path in input_paths]
    for input_path, output_path in jobs:
        logger.info("Preprocessing {} -> {}", input_path, output_path)

    max_workers = args.num_workers if args.num_workers is not None else (os.cpu_count() or 1)
    if max_workers <= 0:
        raise ValueError("`--num-workers` must be a positive integer.")
    max_workers = min(max_workers, len(jobs))

    if max_workers == 1:
        iterator = jobs
        for job in tqdm(iterator, total=len(jobs), desc="Preprocessing", unit="file"):
            _preprocess_file(*job)
        return

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_configure_warnings) as executor:
        futures = [executor.submit(_preprocess_file, input_path, output_path) for input_path, output_path in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing", unit="file"):
            future.result()


if __name__ == "__main__":
    main()
