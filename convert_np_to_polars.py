import argparse
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
import tqdm

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--level', type=int, default=6, help='Octree level to process')

args = parser.parse_args()

level = args.level
cube_size = 2**level

image_source = '../pbgnn/data/shapenet_config_ortho_vis_1_128'

angle_id = 18

ray_dir = np.load(f'../pbgnn/data/intersect_edge_lt_ray_dirs_1.npz')['ray_dirs'][angle_id]

pbrt_pixel_to_pos = np.load(f'{image_source}/pixel_to_pos_1_{angle_id}.npy')
print(pbrt_pixel_to_pos.shape)
intersect_mask = pbrt_pixel_to_pos[..., -1].astype(bool).flatten()
coords = pbrt_pixel_to_pos[..., 1:4].reshape(-1, 3)[intersect_mask]

# Find hit positions on the surface of a 4x4x4 cube
cube_corners = np.array([
    [0, 0, 0],
    [1, 1,  1],
])*4

t_pos = (cube_corners[None] - coords[:, None]) / -ray_dir
t_min, t_max = np.min(t_pos, axis=1), np.max(t_pos, axis=1)
t_min = t_min.max(axis=-1)
t_max = t_max.min(axis=-1)

intersect_mask = t_max >= t_min

print(intersect_mask.sum(), intersect_mask.shape[0])

t = t_min[intersect_mask]

coords = coords[intersect_mask] - t[:, None] * ray_dir
coords = coords / 4.0  # Normalize to unit cube

image_fp = np.memmap(f'{image_source}/level_{level}/masked_images_{angle_id}.npy', dtype=np.float16)

samples_per_config = intersect_mask.sum()

# image_fp layout is config_idx major
def get_config_idx(flattened_idx: int) -> int:
    return flattened_idx // samples_per_config

polars_schema = {
    'x': pl.Float32,
    'y': pl.Float32,
    'z': pl.Float32,
    'hit': pl.Boolean,
    'distance': pl.Float32
}

def estimate_size_in_memory(num_rows: int, schema: dict) -> int:
    total_size = 0
    for dtype in schema.values():
        if dtype == pl.Float32:
            total_size += 4
        elif dtype == pl.Boolean:
            total_size += 1
        else:
            raise ValueError(f"Unsupported data type: {dtype}")
    return total_size * num_rows

def estimate_rows(size_in_bytes: int, schema: dict) -> int:
    row_size = 0
    for dtype in schema.values():
        if dtype == pl.Float32:
            row_size += 4
        elif dtype == pl.Boolean:
            row_size += 1
        else:
            raise ValueError(f"Unsupported data type: {dtype}")
    return size_in_bytes // row_size

def determine_cube_face(coords: np.ndarray, cube_size: int) -> np.ndarray:
    abs_coords = np.abs(coords)
    max_axes = np.argmax(abs_coords, axis=1)
    signs = np.sign(coords[np.arange(coords.shape[0]), max_axes])
    face_indices = max_axes * signs
    return face_indices


target_size_in_memory = 256*1024**2  # 256 MB
num_rows = estimate_rows(target_size_in_memory, polars_schema)

output_dir = Path(f'data/chunked/level_{level}/')
output_dir.mkdir(parents=True, exist_ok=True)

"""
coords and intersect mask are both per-config, so have a length of 9484
image_fp has the throughput results for all configs, so has length of num_configs * 9484

If the chunk size is larger than 9484 we need to wrap-around/repeat the coords and intersect mask
so that if we cross over into the next config we have the right coords and intersect mask values
for that config.
"""

def convert_chunk_idx_to_coord_idx(chunk_idx: np.ndarray) -> np.ndarray:
    return chunk_idx % coords.shape[0]

chunk_start_indices = np.arange(0, image_fp.shape[0], num_rows)

for file_idx in tqdm.tqdm(range(len(chunk_start_indices))):
    chunk_start_idx = chunk_start_indices[file_idx]
    chunk_end_idx = min(chunk_start_idx + num_rows, image_fp.shape[0])

    coord_idx = convert_chunk_idx_to_coord_idx(np.arange(chunk_start_idx, chunk_end_idx))
    
    coords_chunk = coords[coord_idx]

    distance_chunk = image_fp[chunk_start_idx:chunk_end_idx].astype(np.float32)
    hit_chunk = np.isclose(distance_chunk, 0.0, atol=1e-3)
    config_idx = get_config_idx(np.arange(chunk_start_idx, chunk_end_idx))
    
    pl_coords_chunk = pl.DataFrame({
        'x': pl.Series(coords_chunk[:, 0].astype(np.float32)),
        'y': pl.Series(coords_chunk[:, 1].astype(np.float32)),
        'z': pl.Series(coords_chunk[:, 2].astype(np.float32)),
        'hit': pl.Series(hit_chunk.astype(np.bool_)),
        'distance': pl.Series(distance_chunk.astype(np.float32)),
        'config_idx': pl.Series(config_idx.astype(np.int32))
    })
    
    pl_coords_chunk.write_parquet(output_dir / f'chunk_{file_idx:04d}.parquet')
