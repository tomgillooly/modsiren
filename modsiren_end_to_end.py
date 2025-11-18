import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import wandb

from pathlib import Path
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

from modsiren import ModSiren
from voxel_dataset_generator.datasets import (
    HierarchicalVoxelRayDataset,
    RayBatchSampler,
    collate_ray_batch,
    transforms,
)

# %%

class SparseGraphEncoder(nn.Module):
    """
    Graph-based sparse voxel encoder using PyTorch Geometric.

    Uses graph convolutions to encode sparse voxel occupancy grids efficiently.
    Only occupied voxels are processed, dramatically reducing memory usage
    for sparse voxel grids (typical 1-5% occupancy).

    Architecture:
    - Initial feature projection from voxel coordinates
    - Multiple graph convolutional layers with residual connections
    - Global pooling to get per-graph embedding
    - Final MLP to project to latent dimension
    """
    def __init__(self, z_dim, hidden_dim=256, num_layers=3, dropout_p=0.0, norm_weights=True):
        super().__init__()

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initial feature projection: (x, y, z) -> hidden_dim
        # Input features: 3D coordinates only (all voxels are occupied by definition)
        self.input_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.GroupNorm(1, hidden_dim) if norm_weights else nn.Identity(),
        )

        # Graph convolution layers with residual connections
        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.GroupNorm(1, hidden_dim) if norm_weights else nn.Identity())
            self.dropouts.append(nn.Dropout(dropout_p))

        # Output projection: hidden_dim -> z_dim
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.GroupNorm(1, hidden_dim) if norm_weights else nn.Identity(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, voxel_pos, edge_index, batch_index):
        """
        Args:
            voxel_pos: (N_total_voxels, 3) - 3D coordinates of occupied voxels
            edge_index: (2, E) - Edge connectivity in COO format
            batch_index: (N_total_voxels,) - Graph assignment for each voxel

        Returns:
            graph_embeddings: (batch_size, z_dim) - Per-graph latent embeddings
        """
        # Use position as initial features
        x = self.input_proj(voxel_pos)  # (N, 3) -> (N, hidden_dim)

        # Apply graph convolutions with residual connections
        for conv, norm, dropout in zip(self.conv_layers, self.norms, self.dropouts):
            identity = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = dropout(x)
            x = x + identity  # Residual connection

        # Global pooling: aggregate all voxels in each graph
        # This gives us one embedding per voxel grid
        pooled = global_mean_pool(x, batch_index)  # (batch_size, hidden_dim)

        # Final projection to latent dimension
        out = self.output_proj(pooled)  # (batch_size, z_dim)

        return out

class FactorizedConv3d(nn.Module):
    """
    Factorized 3D convolution that decomposes a 3D kernel into three 1D convolutions
    along each axis (X, Y, Z). This dramatically reduces parameters and memory usage.

    For example, a Conv3d(1, C, kernel_size=K) has K³ * C parameters.
    This factorized version has approximately 3 * K * C parameters.

    For K=256, C=256:
    - Standard Conv3d: ~4.3B parameters
    - Factorized: ~196K parameters (22,000x reduction!)

    This is particularly useful for "global" convolutions that reduce a volume to 1×1×1.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Intermediate channels
        mid_channels_1 = max(in_channels, out_channels // 2)
        mid_channels_2 = max(mid_channels_1, out_channels)

        # Convolve along X axis (reduce width)
        # Input: (B, C, D, H, W)
        self.conv_x = nn.Conv3d(
            in_channels, mid_channels_1,
            kernel_size=(1, 1, kernel_size),
            stride=(1, 1, stride),
            padding=(0, 0, padding),
            bias=False
        )
        self.norm_x = nn.GroupNorm(1, mid_channels_1)

        # Convolve along Y axis (reduce height)
        self.conv_y = nn.Conv3d(
            mid_channels_1, mid_channels_2,
            kernel_size=(1, kernel_size, 1),
            stride=(1, stride, 1),
            padding=(0, padding, 0),
            bias=False
        )
        self.norm_y = nn.GroupNorm(1, mid_channels_2)

        # Convolve along Z axis (reduce depth)
        self.conv_z = nn.Conv3d(
            mid_channels_2, out_channels,
            kernel_size=(kernel_size, 1, 1),
            stride=(stride, 1, 1),
            padding=(padding, 0, 0),
            bias=bias
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C_in, D, H, W)
        Returns:
            Output tensor of shape (B, C_out, D', H', W')
        """
        # Convolve along X (width) dimension
        x = self.conv_x(x)
        x = self.norm_x(x)
        x = F.relu(x)

        # Convolve along Y (height) dimension
        x = self.conv_y(x)
        x = self.norm_y(x)
        x = F.relu(x)

        # Convolve along Z (depth) dimension
        x = self.conv_z(x)

        return x

class ModSirenEndToEnd(ModSiren):
    """ModSiren model with PyTorch Lightning wrapper using sparse graph convolutions."""

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

        # Config embed - using sparse graph convolution encoder
        # This replaces dense 3D convolutions with graph operations on occupied voxels only
        self.register_module('config_encoder',
            SparseGraphEncoder(
                z_dim=kwargs['z_dim'],
                hidden_dim=kwargs.get('graph_hidden_dim', 256),
                num_layers=kwargs.get('graph_num_layers', 3),
                dropout_p=kwargs['dropout_p'],
                norm_weights=kwargs['norm_weights']
            )
        )

    def encode_config_sparse(self, voxel_pos, edge_index, batch_index):
        """
        Encode sparse voxel grids using graph convolutions.

        Args:
            voxel_pos: (N_total_voxels, 3) - Coordinates of occupied voxels
            edge_index: (2, E) - Edge connectivity
            batch_index: (N_total_voxels,) - Graph assignment for each voxel

        Returns:
            latent: (batch_size, z_dim) - Per-graph latent embeddings
        """
        return self.config_encoder(voxel_pos, edge_index, batch_index)

    def forward(self, coords, latent):
        """
        Forward pass with pre-computed latent codes.

        Args:
            coords: (N_rays, input_features) - Ray origins + directions
            latent: (N_rays, z_dim) - Latent codes for each ray

        Returns:
            predictions: (N_rays, 2) - [mask_logits, distance]
        """
        return super().forward(coords, latent)
            
    def training_step(self, batch, batch_idx):
        """
        Training step for the Lightning module with sparse graph data.

        Args:
            batch: Dictionary containing sparse voxel graphs and ray data
            batch_idx: Index of the batch

        Returns:
            Loss value
        """
        # Extract ray data
        coords = batch['origins']
        directions = batch['directions']
        signal = batch['distances']
        gt_mask = batch['hits'].bool()
        ray_to_voxel = batch['ray_to_voxel']

        # Extract sparse voxel graph data
        voxel_pos = batch['voxel_pos']  # (N_voxels, 3)
        edge_index = batch['voxel_edge_index']  # (2, E)
        batch_index = batch['voxel_batch']  # (N_voxels,) - which graph each voxel belongs to

        # Encode sparse voxel grids using graph convolutions
        # Returns one embedding per voxel grid in the batch
        latent = self.encode_config_sparse(voxel_pos, edge_index, batch_index)  # (batch_size, z_dim)

        # Map ray-specific latent codes using ray_to_voxel index
        # Each ray gets the latent code of its corresponding voxel grid
        ray_latent = latent[ray_to_voxel]  # (N_rays, z_dim)

        # Forward pass through ModSiren
        mask_pred_logits, pred_dist = super().forward(
            th.cat((coords, directions), dim=-1),
            ray_latent
        ).unbind(-1)

        # Compute losses
        mask_loss = F.binary_cross_entropy_with_logits(
            mask_pred_logits.squeeze(-1),
            gt_mask.float().squeeze(-1)
        )

        mse_loss = F.mse_loss(
            pred_dist[gt_mask].squeeze(-1),
            signal[gt_mask].squeeze(-1)
        )

        loss = mse_loss + self.hparams['mask_loss_weight'] * mask_loss

        # Metrics
        psnr = -10 * th.log10(mse_loss + 1e-8)
        accuracy = ((th.sigmoid(mask_pred_logits) > 0.5) == gt_mask).float().mean()

        # Logging
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('train_psnr', psnr, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('train_acc', accuracy, prog_bar=True, on_epoch=True, sync_dist=True)

        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, on_step=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the Lightning module with sparse graph data.

        Args:
            batch: Dictionary containing sparse voxel graphs and ray data
            batch_idx: Index of the batch
        """
        # Extract ray data
        coords = batch['origins']
        directions = batch['directions']
        signal = batch['distances']
        gt_mask = batch['hits'].bool()
        ray_to_voxel = batch['ray_to_voxel']

        # Extract sparse voxel graph data
        voxel_pos = batch['voxel_pos']
        edge_index = batch['voxel_edge_index']
        batch_index = batch['voxel_batch']

        # Encode sparse voxel grids
        latent = self.encode_config_sparse(voxel_pos, edge_index, batch_index)

        # Map ray-specific latent codes
        ray_latent = latent[ray_to_voxel]

        # Forward pass
        mask_pred_logits, pred_dist = super().forward(
            th.cat((coords, directions), dim=-1),
            ray_latent
        ).unbind(-1)

        mask_pred_logits = mask_pred_logits.squeeze(-1)

        # Compute metrics
        mse_loss = th.nn.functional.mse_loss(pred_dist[gt_mask].squeeze(), signal[gt_mask])
        psnr = -10 * th.log10(mse_loss + 1e-8)
        accuracy = ((th.sigmoid(mask_pred_logits) > 0.5) == gt_mask).float().mean()

        # Logging
        self.log('val_psnr', psnr, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_acc', accuracy, prog_bar=True, on_epoch=True, sync_dist=True)
    
    def configure_optimizers(self):
        """
        Configure optimizers for the Lightning module.
        
        Returns:
            Optimizer for model parameters
        """
        optimizer = th.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'], betas=(self.hparams['beta1'], self.hparams['beta2']))

        
        warmup_scheduler = th.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.hparams['lr_warmup'],
                                                          end_factor=1, 
                                                          total_iters=self.hparams['lr_warmup_steps']*self.hparams['steps_per_epoch'])
        scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[(s-self.hparams['lr_warmup_steps'])*self.hparams['steps_per_epoch'] for s in self.hparams['lr_milestones']], gamma=self.hparams['lr_gamma'])
        # Combine warmup and step scheduler
        seq_scheduler = th.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[self.hparams['lr_warmup_steps']*self.hparams['steps_per_epoch']])
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': seq_scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

def create_dataloaders(
    dataset_dir: Path,
    ray_dataset_dir: Path,
    batch_size: int = 8,
    rays_per_chunk: int = 4096,
    num_workers: int = 4,
    sparse_connectivity: int = 6
):
    """Create train and validation dataloaders with sparse graph voxel representation.

    Args:
        dataset_dir: Path to voxel dataset
        ray_dataset_dir: Path to ray dataset
        batch_size: Number of subvolumes per batch
        rays_per_chunk: Total rays per batch
        num_workers: Number of worker processes
        sparse_connectivity: Graph connectivity (6, 18, or 26 neighbors)

    Returns:
        train_loader, val_loader
    """
    # Define augmentation pipeline for training
    train_transform = transforms.Compose([
        transforms.RandomRotation90(p=0.5),
        transforms.RandomFlip(axes=[0, 1, 2], p=0.5),
        transforms.NormalizeRayOrigins(voxel_size=1.0),
    ])

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.NormalizeRayOrigins(voxel_size=1.0),
    ])

    # Create datasets with sparse graph representation
    train_dataset = HierarchicalVoxelRayDataset(
        dataset_dir=dataset_dir,
        ray_dataset_dir=ray_dataset_dir,
        split='train',
        levels=[0],
        cache_size=100,
        # transform=train_transform,
        rays_per_chunk=rays_per_chunk,
        sparse_voxels=True,  # Enable sparse representation
        sparse_mode='graph',  # Use graph mode for GNN
        sparse_connectivity=sparse_connectivity,  # 6-connected neighbors
        seed=42
    )

    val_dataset = HierarchicalVoxelRayDataset(
        dataset_dir=dataset_dir,
        ray_dataset_dir=ray_dataset_dir,
        split='val',
        levels=[0],
        cache_size=50,
        # transform=val_transform,
        rays_per_chunk=rays_per_chunk,
        sparse_voxels=True,
        sparse_mode='graph',
        sparse_connectivity=sparse_connectivity,
        seed=42
    )

    # Create dataloaders with custom batch sampler
    train_sampler = RayBatchSampler(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_sampler = RayBatchSampler(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_ray_batch,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_ray_batch,
        pin_memory=True,
    )

    return train_loader, val_loader

class ModSirenParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description='ModSiren MAML Retrain')

        self.add_argument('--voxel_dataset_dir', type=str, default='../voxel-dataset-generator/dataset')
        self.add_argument('--ray_dataset_dir', type=str, default='../voxel-dataset-generator/ray_dataset_hierarchical')
        self.add_argument('--run_group', type=str, default='modsiren_end_to_end')
        self.add_argument('--run_notes', type=str, default=None)
        self.add_argument('--max_epochs', type=int, default=40)
        self.add_argument('--num-workers', type=int, default=24)
        self.add_argument('--batch_size', type=int, default=8)
        self.add_argument('--rays_per_chunk', type=int, default=2048)
        self.add_argument('--accumulate_grad_batches', type=int, default=1)
        self.add_argument('--project', type=str, default='thingi10k-modsiren')
        self.add_argument('--sine_layer_type', type=str, default='sine')
        self.add_argument('--input_features', type=int, default=6)
        self.add_argument('--num_dfl_bins', type=int, default=16)
        self.add_argument('--min_log_sigmat', type=float, default=-2)
        self.add_argument('--hidden_layers', type=int, default=4)
        self.add_argument('--mod_layers', type=int, default=4)
        self.add_argument('--first_layer_features', type=int, default=64)
        self.add_argument('--min_hidden_features', type=int, default=64)
        self.add_argument('--hidden_features', type=int, default=64)
        self.add_argument('--z_dim', type=int, default=32)
        self.add_argument('--norm_weights', action='store_true')
        self.add_argument('--no_norm_weights', dest='norm_weights', action='store_false')
        self.set_defaults(norm_weights=True)
        self.add_argument('--dropout_p', type=float, default=0.0)
        self.add_argument('--learning_rate', type=float, default=1e-3)
        self.add_argument('--lr_gamma', type=float, default=0.1)
        self.add_argument('--lr_milestones', type=int, nargs='+', default=[10, 35])
        self.add_argument('--lr_warmup_steps', type=int, default=3)
        self.add_argument('--lr_warmup', type=int, default=1e-2)
        self.add_argument('--lr_step', type=int, default=10)
        self.add_argument('--freq_scaling', type=float, default=30)
        self.add_argument('--init_freq_scaling', type=float, default=30)
        self.add_argument('--optimiser', type=str, default='adam', choices=['adam', 'sgd'])
        self.add_argument('--beta1', type=float, default=0.9)
        self.add_argument('--beta2', type=float, default=0.99)
        self.add_argument('--incode_weight_decay', type=float, default=0.0)
        self.add_argument('--grad_scale', type=float, default=1.0)  # Gradient scaling factor
        self.add_argument('--clip_grad_norm', type=float, default=0.0)  # Gradient clipping
        self.add_argument('--mask_loss_weight', type=float, default=1e-2)
        self.add_argument('--dfl_beta', type=float, default=0)
        self.add_argument('--dfl_temperature', type=float, default=0.01)
        self.add_argument('--logger_on', action='store_true')
        self.add_argument('--logger_off', dest='logger_on', action='store_false')
        self.set_defaults(logger_on=False)
        self.add_argument('--sparse_connectivity', type=int, default=26, choices=[6, 18, 26])
        self.add_argument('--graph_hidden_dim', type=int, default=256)
        self.add_argument('--graph_num_layers', type=int, default=3)
        
if __name__ == '__main__':
    parser = ModSirenParser()

    args = vars(parser.parse_args())

    train_loader, val_loader = create_dataloaders(
        args['voxel_dataset_dir'],
        args['ray_dataset_dir'],
        args['batch_size'],
        args['rays_per_chunk'],
        args['num_workers'],
        args['sparse_connectivity']
    )

    args['steps_per_epoch'] = int(np.ceil(len(train_loader) / int(os.environ.get('SLURM_GPUS_PER_NODE', 1))))

    model = ModSirenEndToEnd(**args)

    print(model)
    
    # Setup logger
    if args['logger_on']:
        logger = pl.loggers.WandbLogger(project='modsiren', 
                                        log_model='all', 
                                        group=args['run_group'], 
                                        notes=args['run_notes'])
        logger.watch(model, log='all')
    else:
        logger = None

    # Setup checkpoint callback with appropriate saving behavior
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='train_loss',  # Change this to whichever metric you're monitoring
        save_top_k=1,
        save_last=True,
        mode='min',
        filename='{epoch}-{train_loss:.4f}'
    )

    # Example of trainer setup with resume capability
    trainer = pl.Trainer(
        max_epochs=args['max_epochs'],
        accelerator='auto',
        # devices=-1,
        logger=logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,  # Disable sanity check
        accumulate_grad_batches=args['accumulate_grad_batches'],
        # Resume from checkpoint if available
        # limit_val_batches=0
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)