from pathlib import Path

from omegaconf import OmegaConf

from steep.utils._general import hash_config

CACHE_SCHEMA_VERSION = 1

TRAINING_CACHE_KEYS = (
    "seed",
    "float_dtype",
    "dataset",
    "model",
    "optimizer",
    "scheduler",
    "loss",
    "trainer.type",
    "trainer.args.train_ratio",
    "trainer.args.val_ratio",
    "trainer.args.split_type",
    "trainer.args.spatial_block_grid_size",
    "trainer.args.epochs",
    "trainer.args.shuffle",
    "trainer.args.batchsize",
    "trainer.args.grad_norm_clip",
    "trainer.args.accelerator",
    "trainer.args.devices",
    "trainer.args.strategy",
    "trainer.args.precision",
    "trainer.args.accumulate_grad_batches",
)

SKETCH_CACHE_KEYS = (
    "seed",
    "dataset",
    "sketcher",
)

EVALUATION_CACHE_KEYS = TRAINING_CACHE_KEYS + (
    "benchmark.args.label_keys",
    "benchmark.args.n_neighbors",
    "benchmark.args.leiden_resolution",
    "benchmark.args.clustering_backend",
    "benchmark.args.classification_backend",
)

MOG_SCORE_CACHE_KEYS = (
    "random_seed",
    "spatial_key",
    "adjacency_matrix_key",
    "feature_key",
    "expr_prior_key",
    "expr_prior_dim",
    "use_topo",
    "use_expr_prior",
    "edge_attr_mode",
    "epochs",
    "lr",
    "temp_r",
    "temp_N",
    "best_score_eval_interval",
)


def dataset_fingerprint(data_directory: str | Path) -> str:
    """Hash the names and sizes of H5AD files in a dataset directory."""
    data_directory = Path(data_directory)
    files = [
        {
            "name": path.name,
            "size": path.stat().st_size,
        }
        for path in sorted(data_directory.glob("*.h5ad"))
    ]
    return hash_config(OmegaConf.create({"files": files}))


def cache_hash_vars(data_directory: str | Path) -> dict[str, str | int]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "data_fingerprint": dataset_fingerprint(data_directory),
    }
