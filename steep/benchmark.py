import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import Subset

import wandb
from steep.trainer import setup_trainer
from steep.utils import (
    DatasetSummary,
    LabelMetricStore,
    accumulate_label_results,
    compute_peak_gpu_memory,
    evaluate_and_plot_slide_embeddings,
    evaluate_sketch_cluster_agreement,
    evaluate_split_classification,
    extract_loss_metrics,
    extract_slide_embedding_records,
    get_fully_qualified_cache_paths,
    instantiate_from_config,
    iter_evaluation_results,
    serialize_dataclass,
    summarize_cluster_scores,
    summarize_split_classification_scores,
)


@dataclass
class EvaluationResult:
    """Cached evaluation outputs for a trained model."""

    run_name: str
    results_folder: str
    evaluation_directory: str
    evaluation_json: str
    train_loss: float | None = None
    val_loss: float | None = None
    test_loss: float | None = None
    label_keys: list[str] | None = None
    metrics_by_label_key: dict[str, Any] | None = None
    classification_metrics_by_label_key: dict[str, Any] | None = None
    training_time_seconds: float | None = None
    peak_gpu_memory_bytes: int | None = None
    epoch_times_seconds: list[float] | None = None
    epoch_time_mean_seconds: float | None = None
    epoch_time_mean_excluding_first_seconds: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationResult":
        """Rebuild an evaluation result from cached JSON data."""
        data = dict(data)
        if "metrics_by_label_key" not in data and "ari_by_label_key" in data:
            data["metrics_by_label_key"] = data.pop("ari_by_label_key")
        valid_keys = set(cls.__dataclass_fields__.keys())
        data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**data)


@dataclass
class SketchResult:
    """Sketch metadata together with its downstream evaluation."""

    sketcher_type: str
    original_num_cells: int
    sketched_num_cells: int
    compression_ratio: float
    original_num_edges: int
    sketched_num_edges: int
    original_disk_bytes: int
    sketched_disk_bytes: int
    sketch_time_seconds: float
    output_directory: str | None = None
    evaluation: EvaluationResult | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SketchResult":
        """Rebuild a sketch result from cached JSON data."""
        data = dict(data)
        evaluation = data.get("evaluation")
        if evaluation is not None:
            data["evaluation"] = EvaluationResult.from_dict(evaluation)
        return cls(**data)


class Benchmark:
    """Benchmarking class for sketching of SRT."""

    def __init__(
        self,
        cfg,
        trainer,
        label_keys: list[str] | tuple[str, ...] = (),
        num_workers: int = 1,
        random_seed: int = 0,
        n_neighbors: int = 15,
        leiden_resolution: float = 1.0,
        clustering_backend: str = "scanpy",
        classification_backend: str = "sklearn",
        resume_from_checkpoint: bool = True,
        run_wandb: bool = False,
        classification_train_ratio: float = 0.1,
        classification_max_iter: int = 1000,
    ):
        self.cfg = cfg
        self.trainer = trainer
        self.label_keys = tuple(label_keys)
        self.num_workers = int(num_workers)
        self.random_seed = int(random_seed)
        self.n_neighbors = int(n_neighbors)
        self.leiden_resolution = float(leiden_resolution)
        self.clustering_backend = str(clustering_backend)
        self.classification_backend = str(classification_backend)
        self.resume_from_checkpoint = bool(resume_from_checkpoint)
        self.run_wandb = bool(run_wandb)
        self.classification_train_ratio = float(classification_train_ratio)
        self.classification_max_iter = int(classification_max_iter)

    def _run_baseline(self) -> EvaluationResult:
        """Run the full-data baseline benchmark."""
        logger.info("Baseline training/evaluation")
        return self._run_training_evaluation(
            trainer=self.trainer,
            evaluation_data=self.trainer.data,
            stage_name="baseline",
            sketched_data_directory=None,
            resume_from_checkpoint=True,
        )

    def _run_sketch(self) -> SketchResult:
        """Run sketching, sketch-model training, and sketch evaluation."""
        logger.info("Sketching dataset materialization")
        sketched_dir, sketch_metrics = self._materialize_sketched_dataset()
        logger.info("Sketch training/evaluation")
        sketched_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        sketched_cfg.dataset.args.data_directory = str(sketched_dir)
        sketched_cfg.run_name = f"{self.cfg.run_name}_sketched"
        sketched_trainer = setup_trainer(sketched_cfg)
        sketch_evaluation = self._run_training_evaluation(
            trainer=sketched_trainer,
            evaluation_data=self.trainer.data,
            stage_name="sketch",
            sketched_data_directory=sketched_dir,
            resume_from_checkpoint=self.resume_from_checkpoint,
        )
        sketch = SketchResult(
            output_directory=str(sketched_dir),
            evaluation=sketch_evaluation,
            **sketch_metrics,
        )
        self._merge_sketch_agreement(sketch, sketched_trainer)

        return sketch

    def _merge_sketch_agreement(self, sketch: SketchResult, sketched_trainer) -> None:
        """Add original-clustering agreement metrics to a sketch evaluation."""
        if "original_leiden" in (sketch.evaluation.metrics_by_label_key or {}):
            logger.info("Sketch agreement already present in cached evaluation")
            return

        logger.info("Sketch agreement scoring")
        sketch_agreement = evaluate_sketch_cluster_agreement(
            reference_trainer=self.trainer,
            reference_data=self.trainer.data,
            target_trainer=sketched_trainer,
            target_data=sketched_trainer.data,
            embedding_eval_num_workers=self.num_workers,
            random_seed=self.random_seed,
            n_neighbors=self.n_neighbors,
            leiden_resolution=self.leiden_resolution,
            clustering_backend=self.clustering_backend,
        )
        sketch.evaluation.label_keys = [
            *(sketch.evaluation.label_keys or []),
            *[
                label_key
                for label_key in sketch_agreement["label_keys"]
                if label_key not in (sketch.evaluation.label_keys or [])
            ],
        ]
        if sketch.evaluation.metrics_by_label_key is None:
            sketch.evaluation.metrics_by_label_key = {}
        sketch.evaluation.metrics_by_label_key.update(sketch_agreement["metrics_by_label_key"])
        with open(sketch.evaluation.evaluation_json, "w") as f:
            json.dump(serialize_dataclass(sketch.evaluation), f, indent=2)

    def get_eval_dir(self, trainer_cfg) -> Path:
        """Return the cache directory used for a benchmark evaluation."""
        eval_dir = get_fully_qualified_cache_paths(
            trainer_cfg,
            Path(self.cfg.cache_dir) / "evaluations",
            keys=(
                "dataset.args.data_directory",
                "model",
                "trainer",
                "sketcher",
                "benchmark.args.label_keys",
                "benchmark.args.random_seed",
                "benchmark.args.n_neighbors",
                "benchmark.args.leiden_resolution",
                "benchmark.args.clustering_backend",
                "benchmark.args.classification_backend",
            ),
        )
        eval_dir.mkdir(parents=True, exist_ok=True)

        plot_dir = eval_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        return eval_dir

    def _dataset_paths(self, dataset) -> list[Path] | None:
        """Resolve slide file paths for datasets that preserve slide-level
        indexing."""
        if hasattr(dataset, "data_paths"):
            return [Path(data_path) for data_path in dataset.data_paths]

        if isinstance(dataset, Subset):
            parent_paths = self._dataset_paths(dataset.dataset)
            if parent_paths is None:
                return None
            return [parent_paths[int(index)] for index in dataset.indices]

        return None

    def _slide_names_for_split(self, trainer, split: str) -> set[str] | None:
        split_dataset = getattr(trainer, "datasets", {}).get(split)
        if split_dataset is None:
            return None

        paths = self._dataset_paths(split_dataset)
        if paths is None:
            return None

        return {path.name for path in paths}

    def _classification_record_splits(self, trainer, slide_records, stage: str):
        train_names = self._slide_names_for_split(trainer, "train")
        test_names = self._slide_names_for_split(trainer, "test")
        if train_names is None or test_names is None:
            logger.warning(
                "{} classification skipped because trainer splits are not slide-level datasets",
                stage.capitalize(),
            )
            return [], []

        if not train_names or not test_names:
            logger.warning(
                "{} classification skipped because train/test slide split is empty",
                stage.capitalize(),
            )
            return [], []

        records_by_name = {record.slide_name: record for record in slide_records}
        train_records = [records_by_name[name] for name in sorted(train_names) if name in records_by_name]
        test_records = [records_by_name[name] for name in sorted(test_names) if name in records_by_name]

        missing_train = sorted(train_names - set(records_by_name))
        missing_test = sorted(test_names - set(records_by_name))
        if missing_train or missing_test:
            logger.warning(
                "{} classification split has missing extracted slides (train missing: {}, test missing: {})",
                stage.capitalize(),
                len(missing_train),
                len(missing_test),
            )

        if not train_records or not test_records:
            logger.warning(
                "{} classification skipped because no extracted train/test records matched the split",
                stage.capitalize(),
            )

        return train_records, test_records

    def evaluate_embeddings(
        self,
        trainer,
        eval_dir: str | Path,
        evaluation_data=None,
        stage: str = "evaluation",
        sketched_data_directory: str | Path | None = None,
    ) -> dict[str, object]:
        """Cluster slide embeddings, score them, classify cell types, write
        plots."""
        slide_records = extract_slide_embedding_records(
            trainer=trainer,
            evaluation_data=evaluation_data,
            label_keys=self.label_keys,
            progress_desc=f"Extracting {stage} embeddings",
        )
        if not slide_records:
            return {
                "label_keys": list(self.label_keys),
                "metrics_by_label_key": {},
                "classification_metrics_by_label_key": {},
            }

        # === Clustering eval ===
        metric_store = {label_key: LabelMetricStore() for label_key in self.label_keys}
        slide_jobs = []
        plot_dir = Path(eval_dir / "plots")
        plot_dir.mkdir(parents=True, exist_ok=True)

        for slide_record in slide_records:
            plot_path = plot_dir / f"{Path(slide_record.slide_name).stem}.png"

            slide_jobs.append(
                {
                    "labels_by_key": slide_record.labels_by_key,
                    "embeddings": slide_record.embeddings,
                    "random_seed": self.random_seed,
                    "n_neighbors": self.n_neighbors,
                    "leiden_resolution": self.leiden_resolution,
                    "clustering_backend": self.clustering_backend,
                    "slide_name": slide_record.slide_name,
                    "obs_names": slide_record.obs_names,
                    "spatial": slide_record.spatial,
                    "plot_path": None if plot_path is None else str(plot_path),
                },
            )

        for slide_results in iter_evaluation_results(
            evaluate_and_plot_slide_embeddings,
            slide_jobs,
            max_workers=max(1, self.num_workers),
            progress_desc=f"Scoring {stage} embeddings",
        ):
            accumulate_label_results(metric_store, slide_results)

        clustering_summary = summarize_cluster_scores(self.label_keys, metric_store)

        logger.info("{} train-slide/test-slide classification", stage.capitalize())
        train_records, test_records = self._classification_record_splits(
            trainer=trainer,
            slide_records=slide_records,
            stage=stage,
        )
        classif_results = {}
        if train_records and test_records:
            classif_results = evaluate_split_classification(
                train_records=train_records,
                test_records=test_records,
                label_keys=self.label_keys,
                n_neighbors=self.n_neighbors,
                confusion_matrix_dir=Path(eval_dir) / "confusion_matrices",
                stage=stage,
                backend=self.classification_backend,
            )
            self._log_confusion_matrices_to_wandb(Path(eval_dir))

        classif_summary = summarize_split_classification_scores(self.label_keys, classif_results)

        return {
            **clustering_summary,
            **classif_summary,
        }

    def _run_training_evaluation(
        self,
        trainer,
        evaluation_data,
        stage_name: str,
        sketched_data_directory: str | Path | None = None,
        resume_from_checkpoint: bool = True,
    ) -> EvaluationResult:
        """Run training evaluation on a particular data object."""
        eval_dir = self.get_eval_dir(trainer.cfg)

        output_path = eval_dir / "evaluation.json"
        if output_path.exists():
            if not resume_from_checkpoint:
                logger.info("{} evaluation cache ignored for fresh run at {}", stage_name.capitalize(), output_path)
            else:
                logger.info("{} cache hit at {}", stage_name.capitalize(), output_path)
                with open(output_path) as f:
                    cached = EvaluationResult.from_dict(json.load(f))

                if cached.classification_metrics_by_label_key is None:
                    logger.info(
                        "{} cached evaluation missing classification metrics; backfilling",
                        stage_name.capitalize(),
                    )
                    trainer.warmup_dataloaders()
                    trainer.fit(resume_from_checkpoint=True)
                    eval_metrics = self.evaluate_embeddings(
                        trainer=trainer,
                        evaluation_data=evaluation_data,
                        eval_dir=eval_dir,
                        stage=stage_name,
                        sketched_data_directory=sketched_data_directory,
                    )

                    if cached.metrics_by_label_key is None:
                        cached.metrics_by_label_key = eval_metrics.get("metrics_by_label_key", {})
                    if cached.label_keys is None:
                        cached.label_keys = eval_metrics.get("label_keys", list(self.label_keys))
                    cached.classification_metrics_by_label_key = eval_metrics.get(
                        "classification_metrics_by_label_key",
                        {},
                    )
                    with open(output_path, "w") as f:
                        json.dump(serialize_dataclass(cached), f, indent=2)
                return cached

        logger.info("{} dataloader/model warmup", stage_name.capitalize())
        trainer.warmup_dataloaders()
        if trainer.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        logger.info("{} model fitting", stage_name.capitalize())
        start_time = time.perf_counter()
        trainer.fit(resume_from_checkpoint=resume_from_checkpoint)
        training_time_seconds = time.perf_counter() - start_time
        reused_checkpoint = bool(getattr(trainer, "last_fit_reused_checkpoint", False))

        logger.info("{} loss extraction and embedding evaluation", stage_name.capitalize())
        losses = extract_loss_metrics(trainer)
        logger.info("{} embedding scoring using {} clustering", stage_name.capitalize(), self.clustering_backend)
        eval_metrics = self.evaluate_embeddings(
            trainer=trainer,
            evaluation_data=evaluation_data,
            eval_dir=eval_dir,
            stage=stage_name,
            sketched_data_directory=sketched_data_directory,
        )

        metrics = EvaluationResult(
            run_name=trainer.cfg.run_name,
            results_folder=str(trainer.results_folder),
            evaluation_directory=str(eval_dir),
            evaluation_json=str(output_path),
            **losses,
            **eval_metrics,
        )
        if not reused_checkpoint:
            metrics.training_time_seconds = training_time_seconds
            metrics.peak_gpu_memory_bytes = compute_peak_gpu_memory(str(trainer.device))
            epoch_times_seconds = []
            if trainer.lightning_trainer is not None:
                for callback in trainer.lightning_trainer.callbacks:
                    callback_epoch_times = getattr(callback, "epoch_times_seconds", None)
                    if callback_epoch_times is not None:
                        epoch_times_seconds = [float(value) for value in callback_epoch_times]
                        break
            if epoch_times_seconds:
                metrics.epoch_times_seconds = epoch_times_seconds
                metrics.epoch_time_mean_seconds = sum(epoch_times_seconds) / len(epoch_times_seconds)
                if len(epoch_times_seconds) > 1:
                    metrics.epoch_time_mean_excluding_first_seconds = sum(epoch_times_seconds[1:]) / (
                        len(epoch_times_seconds) - 1
                    )

        with open(output_path, "w") as f:
            json.dump(serialize_dataclass(metrics), f, indent=2)

        logger.info("{} evaluation saved to {}", stage_name.capitalize(), output_path)
        return metrics

    def _materialize_sketched_dataset(self) -> tuple[Path, dict[str, Any]]:
        """Build or reuse a cached sketched dataset on disk."""

        input_dir = Path(self.cfg.dataset.args.data_directory)
        output_dir = get_fully_qualified_cache_paths(
            self.cfg,
            Path(self.cfg.cache_dir) / "sketches",
            keys=("dataset.args.data_directory", "sketcher"),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "sketch_metadata.json"
        if metadata_path.exists():
            logger.info("Sketch dataset cache hit at {}", metadata_path)
            with open(metadata_path) as f:
                return output_dir, json.load(f)

        sketcher = instantiate_from_config(self.cfg.sketcher)
        input_paths = sorted(input_dir.glob("*.h5ad"))
        logger.info("Sketching {} input slides into {}", len(input_paths), output_dir)

        total_sketch_time_seconds = 0.0
        total_original_num_cells = 0
        total_sketched_num_cells = 0
        total_original_num_edges = 0
        total_sketched_num_edges = 0
        total_original_disk_bytes = 0
        total_sketched_disk_bytes = 0

        for input_path in input_paths:
            output_path = output_dir / input_path.name
            metadata = sketcher.fit_transform_to_disk(input_path, output_path)
            total_sketch_time_seconds += float(metadata.sketch_time_seconds or 0.0)
            total_original_num_cells += metadata.original_num_cells
            total_sketched_num_cells += metadata.sketched_num_cells
            total_original_num_edges += int(metadata.original_num_edges or 0)
            total_sketched_num_edges += int(metadata.sketched_num_edges or 0)
            total_original_disk_bytes += int(metadata.original_disk_bytes or 0)
            total_sketched_disk_bytes += int(metadata.sketched_disk_bytes or 0)

        compression_ratio = total_sketched_num_cells / total_original_num_cells if total_original_num_cells > 0 else 0.0

        metadata = {
            "sketcher_type": self.cfg.sketcher.type,
            "original_num_cells": total_original_num_cells,
            "sketched_num_cells": total_sketched_num_cells,
            "compression_ratio": compression_ratio,
            "original_num_edges": total_original_num_edges,
            "sketched_num_edges": total_sketched_num_edges,
            "original_disk_bytes": total_original_disk_bytes,
            "sketched_disk_bytes": total_sketched_disk_bytes,
            "sketch_time_seconds": total_sketch_time_seconds,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Sketch metadata saved to {}", metadata_path)
        return output_dir, metadata

    def _start_wandb_run(self):
        """Start the benchmark-level WandB run when enabled."""
        if not self.run_wandb:
            return None

        extra_tags = OmegaConf.select(self.cfg, "wandb_tags", default=[]) or []
        if isinstance(extra_tags, str):
            extra_tags = [extra_tags]
        tags = tuple(dict.fromkeys([self.cfg.dataset.name, "evaluation", *extra_tags]))

        return wandb.init(
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=f"{self.cfg.run_name}_evaluation",
            group=OmegaConf.select(self.cfg, "wandb_group", default=None),
            job_type="evaluation",
            tags=tags,
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )

    def _finalize_wandb_run(self, run, summary: dict[str, Any]) -> None:
        """Log benchmark outputs to WandB and finish the run."""
        if run is None:
            return

        flattened = self._flatten_metrics(summary)
        run.log(flattened)
        run.summary.update(flattened)
        artifact_dirs = []
        baseline_eval_dir = summary.get("baseline", {}).get("evaluation_directory")
        if baseline_eval_dir:
            artifact_dirs.append(Path(baseline_eval_dir))
        sketch_eval_dir = summary.get("sketch", {}).get("evaluation", {}).get("evaluation_directory")
        if sketch_eval_dir:
            artifact_dirs.append(Path(sketch_eval_dir))

        for artifact_dir in artifact_dirs:
            if not artifact_dir.exists():
                continue
            self._log_confusion_matrices_to_wandb(artifact_dir, run=run)
            artifact = wandb.Artifact(f"{artifact_dir.name}_evaluation", type="evaluation_outputs")
            artifact.add_dir(str(artifact_dir))
            run.log_artifact(artifact)
        run.finish()

    def _log_confusion_matrices_to_wandb(self, artifact_dir: Path, run=None) -> None:
        run = wandb.run if run is None else run
        if run is None:
            return

        confusion_matrix_dir = artifact_dir / "confusion_matrices"
        if not confusion_matrix_dir.exists():
            return

        for image_path in sorted(confusion_matrix_dir.glob("*.png")):
            image_key = f"confusion_matrices/{artifact_dir.name}/{image_path.stem}"
            run.log({image_key: wandb.Image(str(image_path))})

    def _flatten_metrics(self, data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested benchmark metrics for WandB logging."""
        flattened = {}
        for key, value in data.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_metrics(value, prefix=full_key))
            else:
                flattened[full_key] = value
        return flattened

    def run(self) -> dict[str, Any]:
        """Run the configured benchmark and return the final summary."""
        logger.info("Benchmark start for run {}", self.cfg.run_name)
        wandb_run = self._start_wandb_run()
        try:
            dataset = DatasetSummary(name=self.cfg.dataset.name, data_directory=self.cfg.dataset.args.data_directory)
            logger.info(
                "Dataset summary: {} slides, {} cells",
                dataset.num_slides,
                dataset.num_cells,
            )
            baseline = self._run_baseline()
            summary = {
                "dataset": serialize_dataclass(dataset),
                "baseline": serialize_dataclass(baseline),
            }

            if "sketcher" in self.cfg:
                summary["sketch"] = serialize_dataclass(self._run_sketch())

            self._finalize_wandb_run(wandb_run, summary)
            logger.info("Benchmark complete for run {}", self.cfg.run_name)
            return summary
        except Exception:
            if wandb_run is not None:
                wandb_run.finish(exit_code=1)
            raise
