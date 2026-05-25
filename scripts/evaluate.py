import os
import warnings


def _configure_process_warnings() -> None:
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("DASK_DATAFRAME__QUERY_PLANNING", "True")
    warnings.simplefilter("ignore")


_configure_process_warnings()

import json

import hydra
from loguru import logger

from steep.trainer import setup_trainer
from steep.utils import instantiate_from_config


@hydra.main(config_path="../steep/config", config_name="config", version_base="1.3")
def main(config):
    trainer = setup_trainer(config)
    benchmark = instantiate_from_config(config.benchmark, cfg=config, trainer=trainer)
    summary = benchmark.run()
    logger.info("Benchmark summary:\n{}", json.dumps(summary, indent=2, default=str))
    baseline_json = summary.get("baseline", {}).get("evaluation_json")
    if baseline_json:
        logger.info("Saved baseline evaluation to {}", baseline_json)

    sketch_json = summary.get("sketch", {}).get("evaluation", {}).get("evaluation_json")
    if sketch_json:
        logger.info("Saved sketched evaluation to {}", sketch_json)


if __name__ == "__main__":
    main()
