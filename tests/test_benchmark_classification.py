from pathlib import Path
from types import SimpleNamespace

from torch.utils.data import Subset

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
