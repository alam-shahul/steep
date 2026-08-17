"""Shared interfaces and metadata for AnnData sketchers."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import anndata as ad

from steep.utils import num_edges_from_adata


@dataclass
class SketchMetadata:
    original_num_cells: int
    sketched_num_cells: int
    compression_ratio: float
    original_num_edges: int | None = None
    sketched_num_edges: int | None = None
    original_disk_bytes: int | None = None
    sketched_disk_bytes: int | None = None
    sketch_time_seconds: float | None = None


class AnnDataSketcher(ABC):
    """Base API for sketchers that shrink processed AnnData objects."""

    def fit(self, adata: ad.AnnData) -> "AnnDataSketcher":
        del adata
        return self

    @abstractmethod
    def transform(self, adata: ad.AnnData) -> ad.AnnData:
        """Return a smaller AnnData object."""

    def fit_transform(self, adata: ad.AnnData) -> ad.AnnData:
        self.fit(adata)
        return self.transform(adata)

    def fit_transform_to_disk(self, input_path: str | Path, output_path: str | Path) -> SketchMetadata:
        input_path = Path(input_path)
        output_path = Path(output_path)

        start_time = time.perf_counter()
        backed_adata = ad.read_h5ad(input_path, backed="r")
        adata = backed_adata.to_memory()
        backed_adata.file.close()
        sketched_adata = self.fit_transform(adata)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sketched_adata.write_h5ad(output_path)
        sketch_time_seconds = time.perf_counter() - start_time

        original_num_cells = int(adata.n_obs)
        sketched_num_cells = int(sketched_adata.n_obs)
        compression_ratio = sketched_num_cells / original_num_cells if original_num_cells > 0 else 0.0

        return SketchMetadata(
            original_num_cells=original_num_cells,
            sketched_num_cells=sketched_num_cells,
            compression_ratio=compression_ratio,
            original_num_edges=num_edges_from_adata(adata),
            sketched_num_edges=num_edges_from_adata(sketched_adata),
            original_disk_bytes=input_path.stat().st_size,
            sketched_disk_bytes=output_path.stat().st_size,
            sketch_time_seconds=sketch_time_seconds,
        )
