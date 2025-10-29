# shuffle_worker.py
# Save this as a separate file in the same directory as your notebook

import polars as pl
from pathlib import Path
from typing import List, Tuple
import random
import math
import logging

logger = logging.getLogger(__name__)


def process_batch_worker(
    batch_chunks_str: List[str],
    pass_number: int,
    batch_number: int,
    seed: int,
    avg_rows_per_chunk: int,
    chunk_offset: int,
    temp_dir_str: str,
    chunk_size_variation: Tuple[float, float]
) -> Tuple[List[str], int, str]:
    """
    Worker function to process a batch of chunks in parallel.
    Returns (output_files, num_output_chunks, status_message)
    
    All Path objects are passed as strings for better pickling.
    """
    try:
        # Convert string paths back to Path objects
        batch_chunks = [Path(p) for p in batch_chunks_str]
        temp_dir = Path(temp_dir_str)
        
        # Load and combine chunks
        chunks = []
        total_rows = 0
        
        for file_path in batch_chunks:
            try:
                chunk = pl.read_parquet(file_path)
                chunks.append(chunk)
                total_rows += len(chunk)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        if not chunks:
            return ([], 0, f"Batch {batch_number}: No chunks loaded")
        
        # Combine all chunks
        combined = pl.concat(chunks)
        
        # Shuffle the combined data
        random.seed(seed)
        shuffled_data = combined.sample(fraction=1.0, shuffle=True)
        
        # Split and save new chunks
        total_rows = len(shuffled_data)
        
        # Vary chunk size slightly
        size_variation = random.uniform(*chunk_size_variation)
        chunk_size = int(avg_rows_per_chunk * size_variation)
        chunk_size = max(1, chunk_size)
        
        n_chunks = math.ceil(total_rows / chunk_size)
        output_files = []
        
        for chunk_idx in range(n_chunks):
            start_row = chunk_idx * chunk_size
            end_row = min(start_row + chunk_size, total_rows)
            
            chunk = shuffled_data.slice(start_row, end_row - start_row)
            
            # Generate output filename with offset
            output_file = temp_dir / f"pass_{pass_number:02d}_chunk_{chunk_idx + chunk_offset:04d}.parquet"
            
            chunk.write_parquet(output_file)
            output_files.append(output_file)  # Return as strings
        
        status = f"Batch {batch_number}: Created {len(output_files)} chunks from {len(batch_chunks)} input chunks"
        return (output_files, len(output_files), status)
        
    except Exception as e:
        error_msg = f"Batch {batch_number}: Error - {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return ([], 0, error_msg)