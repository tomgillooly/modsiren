# Multi-GPU Parquet DataLoader for PyTorch Lightning

Efficient data loading solution for pre-shuffled parquet files with multi-GPU support.

## Overview

This solution provides efficient sequential slicing from pre-shuffled parquet files for PyTorch Lightning training across multiple GPUs. It avoids the overhead of random access by using sequential reads, while ensuring each GPU and worker gets unique, non-overlapping data slices.

## Key Features

- **Sequential I/O**: Leverages sequential reads for maximum disk throughput
- **Multi-GPU Support**: Automatic data partitioning across GPUs using DDP
- **Worker Support**: Each worker per GPU gets unique data slices
- **Lazy Loading**: Uses Polars LazyFrames for memory efficiency
- **Lightning Integration**: Drop-in replacement for standard PyTorch Lightning DataModules
- **Zero Duplication**: Each sample is seen by exactly one GPU/worker

## How It Works

### Data Partitioning Strategy

```
Total Data: [======================================] 100M rows

GPU 0: [==========] rows 0-25M
GPU 1: [==========] rows 25M-50M
GPU 2: [==========] rows 50M-75M
GPU 3: [==========] rows 75M-100M

Within GPU 0 (with 4 workers):
  Worker 0: [==] rows 0-6.25M
  Worker 1: [==] rows 6.25M-12.5M
  Worker 2: [==] rows 12.5M-18.75M
  Worker 3: [==] rows 18.75M-25M
```

Each GPU and worker calculates its slice based on:
- `rank`: GPU index in distributed training
- `world_size`: Total number of GPUs
- `worker_id`: Worker index within a GPU
- `num_workers`: Total workers per GPU

### Sequential Slicing

Instead of random indexing (which requires random disk seeks), we use:
```python
# Efficient: sequential read from calculated offset
batch = lazy_df.slice(offset, batch_size).collect()

# vs. Inefficient: random access
batch = df[random_indices]  # Causes many disk seeks
```

## Quick Start

### 1. Basic Usage

```python
from parquet_datamodule import ParquetDataModule
import lightning.pytorch as pl

# Define a transform for your data format
def my_transform(df):
    coords = th.from_numpy(df['coords'].to_numpy()).float()
    signal = th.from_numpy(df['signal'].to_numpy()).float()
    return coords, signal

# Create datamodule
datamodule = ParquetDataModule(
    train_path="data/train/*.parquet",
    val_path="data/val/*.parquet",
    batch_size=256,
    num_workers=4,
    transform=my_transform
)

# Train on multiple GPUs
trainer = pl.Trainer(devices=4, accelerator='gpu', strategy='ddp')
trainer.fit(model, datamodule)
```

### 2. Integration with ModSiren

```python
from parquet_datamodule import ParquetDataModule
from modsiren_dfl import ModSirenMAML

# Create transform matching ModSiren's expected format
def modsiren_transform(df):
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

# Create datamodule
datamodule = ParquetDataModule(
    train_path="data/nrt-mesh-intersection-dataset/shuffled/*/*.parquet",
    batch_size=256,
    num_workers=4,
    transform=modsiren_transform,
    columns=['origin_x', 'origin_y', 'origin_z',
             'distance', 'left_bin_idx', 'w', 'config_idx']
)

# Create model and train
model = ModSirenMAML(**model_args)
trainer = pl.Trainer(devices=4, accelerator='gpu', strategy='ddp')
trainer.fit(model, datamodule)
```

## Performance Optimization Tips

### 1. Pre-shuffle Your Data

Since this uses sequential slicing, make sure your parquet files are pre-shuffled:
```python
# Use your shuffle_data.ipynb notebook to shuffle
shuffler = MemoryAwareChunkShuffler(
    input_dir='data/raw',
    output_dir='data/shuffled'
)
shuffled_chunks = shuffler.shuffle_dataset(seed=42)
```

### 2. Specify Total Rows

If you know the total number of rows, pass it to avoid an initial scan:
```python
datamodule = ParquetDataModule(
    train_path="data/*.parquet",
    train_rows=10_000_000,  # Much faster initialization
    batch_size=256
)
```

### 3. Load Only Required Columns

Specify only the columns you need:
```python
datamodule = ParquetDataModule(
    train_path="data/*.parquet",
    columns=['coords', 'signal', 'config_idx'],  # Don't load unused columns
    batch_size=256
)
```

### 4. Tune Number of Workers

Balance between CPU cores and I/O:
```python
# More workers = more parallel I/O, but more CPU/memory overhead
# Start with: num_workers = num_cpu_cores // num_gpus
datamodule = ParquetDataModule(
    train_path="data/*.parquet",
    num_workers=4,  # Tune based on your system
    batch_size=256
)
```

### 5. Use Persistent Workers

Already enabled by default in `ParquetDataModule`:
```python
# In the dataloader
DataLoader(
    dataset,
    persistent_workers=True,  # Keeps workers alive between epochs
    num_workers=4
)
```

## Advanced Usage

### Custom Transform Functions

Create sophisticated transforms for your data:

```python
def advanced_transform(df):
    """Transform with data augmentation."""
    # Extract data
    coords = th.from_numpy(df['coords'].to_numpy()).float()
    signal = th.from_numpy(df['signal'].to_numpy()).float()

    # Apply augmentation
    if random.random() > 0.5:
        coords = coords + th.randn_like(coords) * 0.01

    # Normalize
    coords = (coords - coords.mean(0)) / coords.std(0)

    return coords, signal

datamodule = ParquetDataModule(
    train_path="data/*.parquet",
    transform=advanced_transform,
    batch_size=256
)
```

### Multiple Parquet Files

Use glob patterns or lists:

```python
# Glob pattern
datamodule = ParquetDataModule(
    train_path="data/train/**/*.parquet",  # Recursive
    batch_size=256
)

# Explicit list
train_files = [
    "data/scene1/chunk_0000.parquet",
    "data/scene1/chunk_0001.parquet",
    "data/scene2/chunk_0000.parquet",
]
datamodule = ParquetDataModule(
    train_path=train_files,
    batch_size=256
)
```

### Custom IterableDataset

For maximum control, extend `ParquetIterableDataset`:

```python
from parquet_datamodule import ParquetIterableDataset

class MyCustomDataset(ParquetIterableDataset):
    def __iter__(self):
        # Call parent to get properly partitioned batches
        for batch_data in super().__iter__():
            # Add custom processing
            batch_data = self.custom_processing(batch_data)
            yield batch_data

    def custom_processing(self, batch_data):
        # Your custom logic here
        return batch_data
```

## Comparison with Standard DataLoader

### Standard (Random Access) Approach
```python
# Problems:
# - Random indexing causes disk seeks
# - Shuffling requires loading all indices
# - Harder to partition across GPUs

class StandardDataset(Dataset):
    def __init__(self, parquet_path):
        self.df = pl.read_parquet(parquet_path)  # Loads all to memory!

    def __getitem__(self, idx):
        return self.df[idx]  # Random access

    def __len__(self):
        return len(self.df)

# Requires DistributedSampler for multi-GPU
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, ...)
```

### Our Sequential Approach
```python
# Benefits:
# - Sequential reads (fast)
# - Minimal memory usage (lazy loading)
# - Automatic GPU/worker partitioning
# - Works with pre-shuffled data

datamodule = ParquetDataModule(
    parquet_path="data/*.parquet",
    batch_size=256
)
# Just works with multi-GPU!
```

## Troubleshooting

### Issue: "RuntimeError: DataLoader worker exited unexpectedly"

**Cause**: Worker ran out of data or memory issue.

**Solution**:
- Ensure your parquet files have enough data for all workers
- Reduce `num_workers` or `batch_size`
- Check if `total_rows` is correct

### Issue: Different GPUs see same data

**Cause**: Distributed training not initialized properly.

**Solution**:
```python
# Make sure to use DDP strategy
trainer = pl.Trainer(
    devices=4,
    accelerator='gpu',
    strategy='ddp',  # Important!
)
```

### Issue: Slow first epoch

**Cause**: Initial parquet scan to count rows.

**Solution**: Provide `total_rows`:
```python
# Count once:
import polars as pl
total = pl.scan_parquet("data/*.parquet").select(pl.len()).collect().item()

# Then use it:
datamodule = ParquetDataModule(
    train_path="data/*.parquet",
    train_rows=total,  # Skip re-counting
    batch_size=256
)
```

### Issue: Out of memory

**Causes**:
- Batch size too large
- Too many workers
- Transform creates large intermediate tensors

**Solutions**:
- Reduce `batch_size`
- Reduce `num_workers`
- Use `columns` parameter to load only needed data
- Optimize transform function

## Files

- `parquet_datamodule.py`: Core DataModule and IterableDataset implementation
- `parquet_example.py`: Usage examples
- `PARQUET_DATALOADER_README.md`: This documentation

## See Also

- PyTorch Lightning [IterableDataset Guide](https://pytorch-lightning.readthedocs.io/en/stable/data/iterabledataset.html)
- Polars [LazyFrame Documentation](https://pola-rs.github.io/polars/py-polars/html/reference/lazyframe/index.html)
- PyTorch [Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
