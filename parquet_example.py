"""
Example showing how to use ParquetDataModule with ModSiren models.

This demonstrates how to replace your existing ImageDataModule with
the more efficient ParquetDataModule for pre-shuffled parquet files.
"""

import torch as th
import lightning.pytorch as pl
from pathlib import Path
from parquet_datamodule import ParquetDataModule
from modsiren_dfl import ModSirenDFL, ModSirenMAML


def create_modsiren_transform(num_dfl_bins=16, min_log_sigmat=-2.0):
    """
    Create a transform function for ModSiren models that matches
    the expected batch format from your existing code.

    The transform converts a Polars DataFrame batch to the tuple format:
    (coords, signal, left_bin_idx, w, idx)

    Args:
        num_dfl_bins: Number of DFL bins
        min_log_sigmat: Minimum log sigma_t value

    Returns:
        Transform function
    """
    def transform(df):
        """
        Transform parquet DataFrame to ModSiren batch format.

        Expected DataFrame columns:
        - origin_x, origin_y, origin_z: Ray origin coordinates
        - direction_x, direction_y, direction_z: Ray direction (or angles)
        - hit: Binary mask indicating if ray hit geometry
        - distance: Distance value (or throughput)
        - config_idx: Configuration index

        Returns:
        - coords: (B, 3) tensor of coordinates
        - signal: (B,) tensor of signal values
        - left_bin_idx: (B,) tensor of left bin indices for DFL
        - w: (B,) tensor of interpolation weights
        - idx: (B,) tensor of configuration indices
        """
        # Extract coordinates (origin + direction, or whatever your input is)
        coords = th.stack([
            th.from_numpy(df['origin_x'].to_numpy()),
            th.from_numpy(df['origin_y'].to_numpy()),
            th.from_numpy(df['origin_z'].to_numpy()),
        ], dim=-1).float()

        # Extract signal (distance, throughput, etc.)
        signal = th.from_numpy(df['distance'].to_numpy()).float()

        # Extract config indices
        idx = th.from_numpy(df['config_idx'].to_numpy()).long()

        # Compute DFL bins (simplified - adjust based on your actual binning)
        # You may want to precompute these and store in the parquet file
        import numpy as np

        # Load or compute bin edges (this should match your DFL setup)
        # For now, using a simple example - replace with your actual bin computation
        bin_edges = np.linspace(min_log_sigmat, 0, num_dfl_bins + 1)

        # Find which bin each signal falls into
        left_bin_idx = th.searchsorted(
            th.from_numpy(bin_edges).float(),
            th.clamp(signal, min=bin_edges[0], max=bin_edges[-1]),
            right=False
        ).long()
        left_bin_idx = th.clamp(left_bin_idx - 1, 0, num_dfl_bins - 2)

        # Compute interpolation weight
        left_edge = bin_edges[left_bin_idx]
        right_edge = bin_edges[left_bin_idx + 1]
        w = (signal - th.from_numpy(left_edge).float()) / (
            th.from_numpy(right_edge - left_edge).float() + 1e-8
        )

        return coords, signal, left_bin_idx, w, idx

    return transform


def create_simple_transform():
    """
    Simpler transform if you have left_bin_idx and w precomputed in parquet.

    This is more efficient as it avoids recomputing bins on every batch.
    """
    def transform(df):
        coords = th.stack([
            th.from_numpy(df['origin_x'].to_numpy()),
            th.from_numpy(df['origin_y'].to_numpy()),
            th.from_numpy(df['origin_z'].to_numpy()),
        ], dim=-1).float()

        signal = th.from_numpy(df['distance'].to_numpy()).float()
        left_bin_idx = th.from_numpy(df['left_bin_idx'].to_numpy()).long()
        w = th.from_numpy(df['w'].to_numpy()).float()
        idx = th.from_numpy(df['config_idx'].to_numpy()).long()

        return coords, signal, left_bin_idx, w, idx

    return transform


# Example 1: Basic training with pre-shuffled parquet files
def example_basic_training():
    """Example: Basic multi-GPU training with parquet files."""

    # Define data paths
    data_dir = Path("data/nrt-mesh-intersection-dataset/shuffled/my_scene")

    # Create datamodule
    datamodule = ParquetDataModule(
        train_path=str(data_dir / "train" / "*.parquet"),
        val_path=str(data_dir / "val" / "*.parquet"),
        batch_size=256,
        num_workers=4,  # Workers per GPU
        transform=create_simple_transform(),  # Or create_modsiren_transform()
        columns=['origin_x', 'origin_y', 'origin_z',
                'distance', 'left_bin_idx', 'w', 'config_idx']
    )

    # Create model (using your existing ModSiren classes)
    model_args = {
        'input_features': 3,
        'hidden_features': 256,
        'hidden_layers': 8,
        'output_features': 17,  # 16 DFL bins + 1 mask
        'z_dim': 128,
        'learning_rate': 1e-3,
        'batch_size': 256,
        # ... other args from your parser
    }

    model = ModSirenMAML(**model_args)

    # Create trainer for multi-GPU
    trainer = pl.Trainer(
        max_epochs=40,
        accelerator='gpu',
        devices=4,  # Use 4 GPUs
        strategy='ddp',  # Distributed Data Parallel
        precision='16-mixed',  # Mixed precision for faster training
    )

    # Train - each GPU automatically gets its own data slice
    trainer.fit(model, datamodule)


# Example 2: More advanced setup matching your existing code
def example_advanced_training():
    """Example: Advanced training matching your modsiren_dfl.py setup."""

    import argparse
    from modsiren_dfl import ModSirenParser

    # Use your existing argument parser
    parser = ModSirenParser()
    args = vars(parser.parse_args([
        '--batch_size', '256',
        '--num_configs', '-1',
        '--level', '2',
        # ... other args
    ]))

    # Create datamodule with parquet files instead of ImageDataModule
    datamodule = ParquetDataModule(
        train_path="data/train/*.parquet",
        val_path="data/val/*.parquet",
        batch_size=args['batch_size'],
        num_workers=4,
        transform=create_modsiren_transform(
            num_dfl_bins=args['num_dfl_bins'],
            min_log_sigmat=args['min_log_sigmat']
        ),
    )

    # Create model
    args['output_features'] = args['num_dfl_bins'] + 1
    if args['model_type'] == 'standard':
        model = ModSirenDFL(**args)
        # Note: You'll need to handle latent initialization differently
        # since we don't have config indices in the same way
    else:
        model = ModSirenMAML(**args)

    # Setup logger
    logger = None
    if args['logger_on']:
        logger = pl.loggers.WandbLogger(
            project='modsiren',
            log_model='all',
            group=args['run_group'],
            notes=args['run_notes']
        )
        logger.watch(model, log='all')

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args['max_epochs'],
        accelerator='gpu',
        devices=-1,  # Use all available GPUs
        strategy='ddp',
        logger=logger,
        gradient_clip_val=args['clip_grad_norm'],
        accumulate_grad_batches=args['accumulate_grad_batches'],
    )

    # Train
    trainer.fit(model, datamodule)


# Example 3: Custom IterableDataset for more control
def example_custom_dataset():
    """
    Example showing how to create a custom dataset if you need
    more control over the data loading process.
    """
    from torch.utils.data import IterableDataset, DataLoader
    import polars as pl

    class CustomParquetDataset(IterableDataset):
        def __init__(self, parquet_path, batch_size):
            super().__init__()
            self.parquet_path = parquet_path
            self.batch_size = batch_size
            self.lazy_df = pl.scan_parquet(parquet_path)
            self.total_rows = self.lazy_df.select(pl.len()).collect().item()

        def __iter__(self):
            # Get worker and rank info
            worker_info = th.utils.data.get_worker_info()

            if th.distributed.is_available() and th.distributed.is_initialized():
                rank = th.distributed.get_rank()
                world_size = th.distributed.get_world_size()
            else:
                rank = 0
                world_size = 1

            # Calculate this worker's slice
            rows_per_rank = self.total_rows // world_size
            start_row = rank * rows_per_rank
            end_row = start_row + rows_per_rank

            if worker_info is not None:
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
                rows_per_worker = (end_row - start_row) // num_workers
                start_row = start_row + worker_id * rows_per_worker
                end_row = start_row + rows_per_worker

            # Iterate over batches
            for offset in range(start_row, end_row, self.batch_size):
                limit = min(self.batch_size, end_row - offset)

                # Read batch
                batch_df = self.lazy_df.slice(offset, limit).collect()

                # Your custom processing here
                coords = th.from_numpy(batch_df['coords'].to_numpy()).float()
                signal = th.from_numpy(batch_df['signal'].to_numpy()).float()

                yield coords, signal

    # Use the custom dataset
    dataset = CustomParquetDataset("data/*.parquet", batch_size=256)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=4,
        persistent_workers=True
    )


if __name__ == "__main__":
    print("ParquetDataModule Examples for ModSiren")
    print("=" * 60)
    print("\nThis file contains examples of how to use ParquetDataModule")
    print("with your ModSiren models for efficient multi-GPU training.")
    print("\nKey features:")
    print("- Sequential slicing (no random access overhead)")
    print("- Automatic multi-GPU data partitioning")
    print("- Efficient Polars-based loading")
    print("- Compatible with PyTorch Lightning DDP")
    print("\nRun the example functions to see different usage patterns.")
