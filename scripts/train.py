import os
import warnings


def _configure_process_warnings() -> None:
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("DASK_DATAFRAME__QUERY_PLANNING", "True")
    warnings.simplefilter("ignore")


_configure_process_warnings()

import hydra

from steep.lightning import (
    build_lightning_callbacks,
    build_wandb_logger,
    cleanup_distributed,
    finalize_lightning_logger,
)
from steep.training import infer_num_genes, initialize_checkpointing, load_pretrained_weights, prepare_resume_checkpoint
from steep.utils import instantiate_from_config


@hydra.main(config_path="../steep/config", config_name="config", version_base="1.3")
def main(config):
    data = instantiate_from_config(config.dataset)
    datamodule = instantiate_from_config(config.datamodule, data=data)
    model = instantiate_from_config(config.model, in_dim=infer_num_genes(config))
    loss_function = instantiate_from_config(config.loss)
    module = instantiate_from_config(
        config.lightning_module,
        cfg=config,
        model=model,
        loss_function=loss_function,
    )

    results_folder = initialize_checkpointing(config)
    logger = build_wandb_logger(config, bool(config.run_wandb))
    callbacks = build_lightning_callbacks(
        results_folder,
        enable_lr_monitor=bool(logger),
        enable_progress_bar=bool(config.trainer.args.enable_progress_bar),
    )
    trainer = instantiate_from_config(
        config.trainer,
        default_root_dir=str(results_folder),
        callbacks=callbacks,
        logger=logger or False,
    )

    load_pretrained_weights(model, config.get("pretrained_ckpt_path"), device="cpu")
    resume_path, _ = prepare_resume_checkpoint(model, results_folder, enabled=True)
    try:
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_path)
        if len(datamodule.datasets["test"]) > 0:
            trainer.test(module, datamodule=datamodule)
    finally:
        finalize_lightning_logger(trainer.logger)
        cleanup_distributed()


if __name__ == "__main__":
    main()
