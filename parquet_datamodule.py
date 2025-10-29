"""
Multi-GPU DataModule for efficiently loading pre-shuffled parquet files using sequential slicing.

This module provides efficient data loading for PyTorch Lightning with:
- Sequential slicing from pre-shuffled parquet files (no random access needed)
- Multi-GPU support with proper data partitioning
- Lazy loading using Polars LazyFrame
- Minimal memory overhead
"""

import polars as pl
import torch as th
import lightning.pytorch as pl_lightning
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
from typing import Optional, Union, List, Callable


class ParquetIterableDataset(IterableDataset):
    """
    Iterable dataset that sequentially reads from pre-shuffled parquet files.

    This dataset is designed for multi-GPU training where:
    1. Data is already shuffled in the parquet files
    2. Each GPU/worker gets a unique, non-overlapping slice of data
    3. Sequential reading provides maximum I/O efficiency

    Args:
        parquet_path: Path to parquet file(s). Can be a glob pattern like "data/*.parquet"
        batch_size: Number of samples per batch
        transform: Optional transform function to apply to each batch
        columns: Optional list of columns to read. If None, reads all columns
        total_rows: Optional total number of rows. If None, will be computed (slower first time)
    """

    def __init__(
        self,
        parquet_path: Union[str, Path, List[str], List[Path]],
        batch_size: int,
        transform: Optional[Callable] = None,
        columns: Optional[List[str]] = None,
        total_rows: Optional[int] = None,
    ):
        super().__init__()
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.transform = transform
        self.columns = columns

        # Create lazy frame for efficient scanning
        self.lazy_df = pl.scan_parquet(parquet_path)

        # Get total number of rows if not provided
        if total_rows is None:
            # This requires a scan but only happens once
            self.total_rows = self.lazy_df.select(pl.len()).collect().item()
        else:
            self.total_rows = total_rows

    def __iter__(self):
        """
        Iterate over batches, properly partitioned for multi-GPU/multi-worker.

        The partitioning works as follows:
        - In distributed training: Each GPU gets a unique slice
        - With multiple workers: Each worker within a GPU gets a unique slice
        - The slices are sequential and non-overlapping
        """
        worker_info = th.utils.data.get_worker_info()

        # Get distributed training info
        if th.distributed.is_available() and th.distributed.is_initialized():
            world_size = th.distributed.get_world_size()
            rank = th.distributed.get_rank()
        else:
            world_size = 1
            rank = 0

        # Calculate per-GPU slice
        rows_per_rank = self.total_rows // world_size
        rank_start = rank * rows_per_rank
        rank_end = rank_start + rows_per_rank if rank < world_size - 1 else self.total_rows

        # Further partition if using multiple workers per GPU
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            rows_per_worker = (rank_end - rank_start) // num_workers
            worker_start = rank_start + worker_id * rows_per_worker
            worker_end = worker_start + rows_per_worker if worker_id < num_workers - 1 else rank_end
        else:
            worker_start = rank_start
            worker_end = rank_end

        # Calculate number of complete batches for this worker
        num_rows = worker_end - worker_start
        num_batches = num_rows // self.batch_size

        # Read and yield batches sequentially
        for batch_idx in range(num_batches):
            offset = worker_start + batch_idx * self.batch_size
            limit = self.batch_size

            # Use Polars' efficient slice operation
            if self.columns is not None:
                batch_df = (
                    self.lazy_df
                    .select(self.columns)
                    .slice(offset, limit)
                    .collect()
                )
            else:
                batch_df = self.lazy_df.slice(offset, limit).collect()

            # Apply transform if provided
            if self.transform is not None:
                batch_data = self.transform(batch_df)
            else:
                # Default: convert to dict of tensors
                batch_data = {
                    col: th.from_numpy(batch_df[col].to_numpy())
                    for col in batch_df.columns
                }

            yield batch_data

    def __len__(self):
        """
        Return number of batches this dataset will produce.

        Note: This is approximate in distributed settings as we return
        the total number of batches across all GPUs/workers.
        """
        return self.total_rows // self.batch_size


class ParquetDataModule(pl_lightning.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading pre-shuffled parquet files.

    This module handles:
    - Automatic multi-GPU data partitioning
    - Efficient sequential reading from parquet files
    - Train/val/test splits

    Args:
        train_path: Path to training parquet file(s)
        val_path: Path to validation parquet file(s)
        test_path: Optional path to test parquet file(s)
        batch_size: Batch size for training
        num_workers: Number of worker processes per GPU
        transform: Optional transform function
        columns: Optional list of columns to load
        train_rows: Optional total rows in training data
        val_rows: Optional total rows in validation data
        test_rows: Optional total rows in test data

    Example:
        >>> datamodule = ParquetDataModule(
        ...     train_path="data/train/*.parquet",
        ...     val_path="data/val/*.parquet",
        ...     batch_size=256,
        ...     num_workers=4,
        ...     columns=['coords', 'signal', 'config_idx']
        ... )
        >>> trainer = pl.Trainer(devices=4, accelerator='gpu')
        >>> trainer.fit(model, datamodule)
    """

    def __init__(
        self,
        train_path: Union[str, Path, List[str], List[Path]],
        val_path: Optional[Union[str, Path, List[str], List[Path]]] = None,
        test_path: Optional[Union[str, Path, List[str], List[Path]]] = None,
        batch_size: int = 256,
        num_workers: int = 4,
        transform: Optional[Callable] = None,
        columns: Optional[List[str]] = None,
        train_rows: Optional[int] = None,
        val_rows: Optional[int] = None,
        test_rows: Optional[int] = None,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.columns = columns
        self.train_rows = train_rows
        self.val_rows = val_rows
        self.test_rows = test_rows

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = ParquetIterableDataset(
                self.train_path,
                self.batch_size,
                self.transform,
                self.columns,
                self.train_rows
            )

            if self.val_path is not None:
                self.val_dataset = ParquetIterableDataset(
                    self.val_path,
                    self.batch_size,
                    self.transform,
                    self.columns,
                    self.val_rows
                )

        if stage == "test" or stage is None:
            if self.test_path is not None:
                self.test_dataset = ParquetIterableDataset(
                    self.test_path,
                    self.batch_size,
                    self.transform,
                    self.columns,
                    self.test_rows
                )

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=None,  # Batching is handled by the dataset
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        if self.val_path is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=None,  # Batching is handled by the dataset
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        if self.test_path is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=None,  # Batching is handled by the dataset
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )


# Example transform function
def default_transform(batch_df: pl.DataFrame) -> dict:
    """
    Default transform that converts a Polars DataFrame to a dict of PyTorch tensors.

    You can create custom transforms for your specific data format.

    Example custom transform:
        def my_transform(df):
            coords = th.from_numpy(df['coords'].to_numpy()).float()
            signal = th.from_numpy(df['signal'].to_numpy()).float()
            idx = th.from_numpy(df['config_idx'].to_numpy()).long()
            return coords, signal, idx
    """
    return {
        col: th.from_numpy(batch_df[col].to_numpy())
        for col in batch_df.columns
    }


if __name__ == "__main__":
    # Example usage
    print("Example 1: Basic usage")

    # Create a simple transform for your data
    def my_transform(df):
        """Transform parquet batch to your model's expected format."""
        # Example: assuming your parquet has columns: coords, signal, config_idx
        coords = th.from_numpy(df['coords'].to_numpy()).float()
        signal = th.from_numpy(df['signal'].to_numpy()).float()
        config_idx = th.from_numpy(df['config_idx'].to_numpy()).long()
        return coords, signal, config_idx

    # Create datamodule
    datamodule = ParquetDataModule(
        train_path="data/train/*.parquet",  # Can use glob patterns
        val_path="data/val/*.parquet",
        batch_size=256,
        num_workers=4,
        transform=my_transform,
        columns=['coords', 'signal', 'config_idx']  # Only load needed columns
    )

    print("\nExample 2: With known row counts (faster initialization)")
    datamodule = ParquetDataModule(
        train_path="data/train/*.parquet",
        val_path="data/val/*.parquet",
        batch_size=256,
        num_workers=4,
        transform=my_transform,
        train_rows=1_000_000,  # If you know the total rows, it's faster
        val_rows=100_000,
    )

    print("\nExample 3: Multi-GPU training")
    print("""
    import lightning.pytorch as pl

    # Your model (e.g., ModSirenDFL or ModSirenMAML)
    model = YourLightningModule()

    # Create trainer for multi-GPU
    trainer = pl.Trainer(
        devices=4,              # Use 4 GPUs
        accelerator='gpu',
        strategy='ddp',         # Distributed Data Parallel
        max_epochs=10
    )

    # Train - each GPU will automatically get its own slice of data
    trainer.fit(model, datamodule)
    """)
