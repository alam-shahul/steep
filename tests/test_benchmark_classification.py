from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
from torch.utils.data import Subset

import steep.benchmark as benchmark_module
from steep.benchmark import Benchmark


class _SlideDataset:
    def __init__(self, data_paths):
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)


def test_benchmark_resolves_train_test_slide_records_from_subset(tmp_path: Path):
    data_paths = [tmp_path / "slide0.h5ad", tmp_path / "slide1.h5ad", tmp_path / "slide2.h5ad"]
    dataset = _SlideDataset(data_paths)
    trainer = SimpleNamespace(
        datasets={
            "train": Subset(dataset, [0, 2]),
            "test": Subset(dataset, [1]),
        },
    )
    records = [
        SimpleNamespace(slide_name="slide0.h5ad"),
        SimpleNamespace(slide_name="slide1.h5ad"),
        SimpleNamespace(slide_name="slide2.h5ad"),
    ]

    benchmark = Benchmark.__new__(Benchmark)
    train_records, test_records = benchmark._classification_record_splits(
        trainer=trainer,
        slide_records=records,
        stage="baseline",
    )

    assert [record.slide_name for record in train_records] == ["slide0.h5ad", "slide2.h5ad"]
    assert [record.slide_name for record in test_records] == ["slide1.h5ad"]


def test_benchmark_skips_classification_for_non_slide_level_splits():
    trainer = SimpleNamespace(datasets={"train": [object()], "test": [object()]})
    benchmark = Benchmark.__new__(Benchmark)

    train_records, test_records = benchmark._classification_record_splits(
        trainer=trainer,
        slide_records=[],
        stage="baseline",
    )

    assert train_records == []
    assert test_records == []


def test_fresh_sketch_run_ignores_dataset_and_score_caches(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "slide.h5ad").write_text("placeholder")

    output_dir = tmp_path / "sketch_cache"
    output_dir.mkdir()
    stale_metadata = output_dir / "sketch_metadata.json"
    stale_metadata.write_text('{"sketch_time_seconds": 999}')

    sketchers = []

    class FakeSketcher:
        def __init__(self):
            self.cache_scores = True

        def fit_transform_to_disk(self, input_path, output_path):
            sketchers.append(self)
            Path(output_path).write_text("sketched")
            return SimpleNamespace(
                sketch_time_seconds=1.0,
                original_num_cells=10,
                sketched_num_cells=5,
                original_num_edges=20,
                sketched_num_edges=8,
                original_disk_bytes=100,
                sketched_disk_bytes=50,
                original_num_edge_pairs=10,
                sketched_num_edge_pairs=4,
            )

    monkeypatch.setattr(benchmark_module, "get_fully_qualified_cache_paths", lambda *args, **kwargs: output_dir)
    monkeypatch.setattr(benchmark_module, "instantiate_from_config", lambda cfg, **kwargs: FakeSketcher())

    benchmark = Benchmark.__new__(Benchmark)
    benchmark.resume_from_checkpoint = False
    benchmark.random_seed = 0
    benchmark.cfg = OmegaConf.create(
        {
            "cache_dir": str(tmp_path / "cache"),
            "seed": 0,
            "dataset": {"args": {"data_directory": str(input_dir)}},
            "sketcher": {"type": "fake"},
        },
    )

    _, metadata = benchmark._materialize_sketched_dataset()

    assert len(sketchers) == 1
    assert sketchers[0].cache_scores is False
    assert metadata["sketch_time_seconds"] == 1.0
