import os
import warnings


def _configure_process_warnings() -> None:
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("DASK_DATAFRAME__QUERY_PLANNING", "True")
    warnings.simplefilter("ignore")


_configure_process_warnings()

import hydra

from steep.trainer import setup_trainer


@hydra.main(config_path="../steep/config", config_name="config", version_base="1.3")
def main(config):
    trainer = setup_trainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()
