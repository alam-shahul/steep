from pathlib import Path

import anndata as ad
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import subgraph as pyg_subgraph

from steep.utils import anndata_to_pyg


class SRTDataset(Dataset):
    def __init__(self, data_directory: str, transform=None):
        super().__init__(None, transform=transform)
        self.data_directory = Path(data_directory)
        self.data_paths = sorted(self.data_directory.glob("*.h5ad"))

    def len(self):  # noqa: A003
        return len(self.data_paths)

    def get(self, idx):
        adata = ad.read_h5ad(self.data_paths[idx])
        data = anndata_to_pyg(adata)
        # print(f'{data.x.size()=}')
        # print(f'{data.edge_index.size()=}')
        return data


def _resolve_partition_sizes(
    total_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, int]:
    requested = {
        "train": max(float(train_ratio), 0.0),
        "val": max(float(val_ratio), 0.0),
        "test": max(float(test_ratio), 0.0),
    }
    active_splits = [name for name, ratio in requested.items() if ratio > 0]
    if total_size < len(active_splits):
        raise ValueError(f"Dataset size {total_size} is too small for active splits {active_splits}.")

    sizes = dict.fromkeys(requested, 0)
    for name in active_splits:
        sizes[name] = 1

    remaining = total_size - len(active_splits)
    if remaining <= 0 or not active_splits:
        return sizes

    total_ratio = sum(requested[name] for name in active_splits)
    raw_allocations = {name: remaining * requested[name] / total_ratio for name in active_splits}
    fractional_parts = []
    for name in active_splits:
        extra = int(raw_allocations[name])
        sizes[name] += extra
        fractional_parts.append((raw_allocations[name] - extra, name))

    leftover = total_size - sum(sizes.values())
    for _, name in sorted(fractional_parts, reverse=True):
        if leftover <= 0:
            break
        sizes[name] += 1
        leftover -= 1

    return sizes


def _normalize_positions_to_unit_square(pos: torch.Tensor) -> torch.Tensor:
    pos = pos.float()
    min_pos = pos.min(dim=0).values
    max_pos = pos.max(dim=0).values
    span = (max_pos - min_pos).clamp_min(1e-6)
    return (pos - min_pos) / span


def _select_contiguous_block_ids(
    counts: torch.Tensor,
    grid_size: int,
    target_count: int,
    seed_xy: tuple[int, int],
    excluded_ids: set[int] | None = None,
) -> set[int]:
    if target_count <= 0:
        return set()

    excluded_ids = set() if excluded_ids is None else set(excluded_ids)
    ranked_cells: list[tuple[int, int, int]] = []
    for y_idx in range(grid_size):
        for x_idx in range(grid_size):
            cell_id = y_idx * grid_size + x_idx
            if cell_id in excluded_ids:
                continue
            distance = abs(x_idx - seed_xy[0]) + abs(y_idx - seed_xy[1])
            ranked_cells.append((distance, y_idx * grid_size + x_idx, cell_id))

    ranked_cells.sort()

    selected_ids: set[int] = set()
    accumulated = 0
    for _, _, cell_id in ranked_cells:
        selected_ids.add(cell_id)
        accumulated += int(counts[cell_id].item())
        if accumulated >= target_count and accumulated > 0:
            break

    if target_count > 0 and accumulated == 0:
        remaining_nonempty = [
            int(cell_id) for cell_id, count in enumerate(counts.tolist()) if count > 0 and cell_id not in excluded_ids
        ]
        if remaining_nonempty:
            densest = max(remaining_nonempty, key=lambda cell_id: int(counts[cell_id].item()))
            selected_ids = {densest}

    return selected_ids


def _build_spatial_band_masks(
    pos: torch.Tensor,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, torch.Tensor]:
    sizes = _resolve_partition_sizes(
        total_size=pos.size(0),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    order = torch.argsort(pos[:, 0], stable=True)

    masks = {name: torch.zeros(pos.size(0), dtype=torch.bool) for name in ("train", "val", "test")}
    start = 0
    for split in ("train", "val", "test"):
        stop = start + sizes[split]
        if stop > start:
            masks[split][order[start:stop]] = True
        start = stop
    return masks


def build_spatial_block_node_subsets(
    data: Data,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    grid_size: int = 4,
) -> dict[str, torch.Tensor]:
    if data.pos.size(0) <= 2:
        masks = _build_spatial_band_masks(data.pos, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        return {name: mask.nonzero(as_tuple=False).flatten() for name, mask in masks.items()}

    normalized_pos = _normalize_positions_to_unit_square(data.pos)
    cell_x = torch.clamp((normalized_pos[:, 0] * grid_size).long(), max=grid_size - 1)
    cell_y = torch.clamp((normalized_pos[:, 1] * grid_size).long(), max=grid_size - 1)
    cell_ids = cell_y * grid_size + cell_x
    cell_counts = torch.bincount(cell_ids, minlength=grid_size * grid_size)

    split_sizes = _resolve_partition_sizes(
        total_size=int(data.num_nodes),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    test_cells = _select_contiguous_block_ids(
        counts=cell_counts,
        grid_size=grid_size,
        target_count=split_sizes["test"],
        seed_xy=(grid_size - 1, grid_size - 1),
    )
    val_cells = _select_contiguous_block_ids(
        counts=cell_counts,
        grid_size=grid_size,
        target_count=split_sizes["val"],
        seed_xy=(0, grid_size - 1),
        excluded_ids=test_cells,
    )

    masks = {
        "test": torch.isin(cell_ids, torch.tensor(sorted(test_cells), dtype=cell_ids.dtype)),
        "val": torch.isin(cell_ids, torch.tensor(sorted(val_cells), dtype=cell_ids.dtype)),
    }
    masks["train"] = ~(masks["test"] | masks["val"])

    if any(split_sizes[name] > 0 and not torch.any(masks[name]) for name in ("train", "val", "test")):
        masks = _build_spatial_band_masks(
            normalized_pos,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    return {name: mask.nonzero(as_tuple=False).flatten() for name, mask in masks.items()}


class SpatialBlockSubsetDataset(Dataset):
    def __init__(self, base_dataset: Dataset, node_subsets: list[tuple[int, torch.Tensor]]):
        super().__init__(None)
        self.base_dataset = base_dataset
        self.node_subsets = node_subsets

    def len(self):  # noqa: A003
        return len(self.node_subsets)

    def get(self, idx):
        base_idx, node_ids = self.node_subsets[idx]
        data = self.base_dataset[base_idx]
        node_ids = node_ids.to(dtype=torch.long)
        edge_index, _ = pyg_subgraph(node_ids, data.edge_index, relabel_nodes=True)
        subgraph = data.subgraph(node_ids)
        subgraph.edge_index = edge_index
        return subgraph
