import argparse
import numpy as np
import os
import polars
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import wandb

from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from modsiren import ModSiren
# %%

def get_dataloader_length(level, batch_size, dataset_type='point', num_configs=-1):
    """
    Get the number of batches per epoch for a given level and batch size.
    
    Args:
        level (int): Level of the dataset.
        batch_size (int): Batch size for training.
        
    Returns:
        int: Number of batches per epoch.
    """
    if num_configs == -1:
        num_configs = len(get_config_indices(level, 'train'))
    num_data_points = num_configs*(9484 if dataset_type == 'point' else 1)
    num_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', 1))  # Number of GPUs available
    
    num_batches = num_data_points // (batch_size * num_gpus)

    if num_data_points % (batch_size * num_gpus) != 0:
        num_batches += 1

    return num_batches


class StepScheduler(object):
    def __init__(self, base_value, gamma, num_steps, total_iters, warmup_iters=0, start_warmup_value=0):
        super().__init__()
        self.total_iters = total_iters

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = total_iters - warmup_iters
        step_size = int(np.ceil(iters / num_steps))
        schedule = np.ones(iters) * base_value
        for i in range(num_steps):
            start = i * step_size
            end = (i + 1) * step_size
            schedule[start:end] *= gamma ** i

        self.schedule = np.concatenate((warmup_schedule, schedule))
        self.final_value = self.schedule[-1]

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]

class DistributionFocalLoss(nn.Module):
    def __init__(self, image_source, angle_id, level, beta=2.0, temperature=1, num_bins=16, min_log_sigmat=-2.0):
        """
        Simplified Distribution Focal Loss implementation.
        
        Args:
            bin_centres (torch.Tensor): Centers of the discretized bins
            beta (float): Focusing parameter for the loss
        """
        super(DistributionFocalLoss, self).__init__()
        self.register_buffer('bin_centres', th.from_numpy(np.load(f'{image_source}/level_{level}/cdf_bins_{angle_id}_{num_bins}_{min_log_sigmat:.02f}.npz')['thpt_inv_cdf']))
        self.beta = beta
        self.temperature = temperature
        
    def forward(self, pred_logits, left_bin_idx, w, target_values):
        """
        Calculate the Distribution Focal Loss.
        
        Args:
            pred_logits (torch.Tensor): Raw logits from network [batch_size, num_bins]
            target_dist (torch.Tensor): Target distributions [batch_size, num_bins]
                                       (should sum to 1 for each sample)
            
        Returns:
            torch.Tensor: Calculated loss
        """
        # Convert logits to probabilities
        pred_probs = F.softmax(pred_logits / self.temperature, dim=-1)
        
        # Calculate expected values
        pred_values = th.sum(pred_probs * self.bin_centres, dim=-1)
        pyl = th.take_along_dim(pred_probs, left_bin_idx[:,  None], dim=-1).squeeze(-1)
        pyr = th.take_along_dim(pred_probs, left_bin_idx[:,  None]+1, dim=-1).squeeze(-1)
        # yl = self.bin_centres[left_bin_idx]
        # yr = self.bin_centres[left_bin_idx+1]
        # pred_values = pyl*yl + pyr*yr
        
        # target_dist = th.zeros_like(pred_probs)
        # target_dist[th.arange(pred_probs.shape[0]), left_bin_idx] = w.float()
        # target_dist[th.arange(pred_probs.shape[0]), left_bin_idx+1] = 1 - w.float()

        with th.no_grad():
            # Modulating factor based on difference between predicted and target values
            mod_factor = th.abs(pred_values - target_values).pow(self.beta)
        
        # Calculate cross-entropy part of the loss
        # ce_loss = -th.sum(target_dist * th.log(pred_probs + 1e-8), dim=-1)
        # ce_loss = F.cross_entropy(pred_probs, left_bin_idx, reduction='none')*w + F.cross_entropy(pred_probs, left_bin_idx+1, reduction='none')*(1-w)
        # ce_loss = -(w*th.log(pyl + 1e-8) + (1 - w)*th.log(pyr + 1e-8))
        # ce_loss = -(w*th.log(pyl + 1e-8) + (1 - w)*th.log(pyr + 1e-8))

        # kl_loss = (w - pyl)*(th.log(w + 1e-8) - th.log(pyl + 1e-8)) + (1 - w - pyr)*(th.log(1 - w + 1e-8) - th.log(pyr + 1e-8))
        kl_loss = w*(th.log(w + 1e-8) - th.log(pyl + 1e-8)) + (1 - w)*(th.log(1 - w + 1e-8) - th.log(pyr + 1e-8))

        # mse_loss = F.mse_loss(pred_probs, target_dist, reduction='none').sum(dim=-1)
        
        # Apply the modulating factor and return mean loss
        return pred_values, (mod_factor * kl_loss).mean()


class ModSirenDFL(ModSiren):
    """ModSiren model with PyTorch Lightning wrapper."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Store z_dim for initializing random latents
        self.z_dim = kwargs['z_dim']
        
        self.dfl_loss = DistributionFocalLoss(kwargs['image_source'], kwargs['angle_id'], kwargs['level'], kwargs['dfl_beta'], kwargs['dfl_temperature'], kwargs['num_dfl_bins'], kwargs['min_log_sigmat'])

    def init_latents(self, num_latents):
        self.register_parameter('latents', nn.Parameter(F.normalize(th.randn(num_latents, self.z_dim) * 0.01)))
    
    def _compute_metrics(self, coords, signal, gt_mask, latents, left_bin_idx, w, prefix=''):
        """
        Compute and log metrics for the current latents.
        
        Args:
            coords: Coordinate tensor
            signal: Signal tensor
            latents: Latent codes
            prefix: Prefix for metric logging (train/val/test)
            
        Returns:
            metrics: Dictionary of computed metrics
        """
        # gt_mask = signal < 1.0
        # gt_mask = th.ones_like(signal, dtype=th.bool)
        norm_latents = F.normalize(latents, dim=1)
        
        # Forward pass
        mask_pred_logits, pred_logits = self(coords.float(), latents).split((1, len(self.dfl_loss.bin_centres)), dim=-1)
        mask_pred_logits = mask_pred_logits.squeeze(-1)

        mask_loss = F.binary_cross_entropy_with_logits(mask_pred_logits, gt_mask.float())
        
        pred_values, thpt_loss = self.dfl_loss(pred_logits[gt_mask], left_bin_idx[gt_mask], w[gt_mask], signal[gt_mask])

        total_loss = thpt_loss + self.hparams['mask_loss_weight'] * mask_loss

        with th.no_grad():
            if not gt_mask.any():
                mse_loss = th.tensor(0.0, device=signal.device)
                psnr = th.tensor(50.0, device=signal.device)
            else:
                mse_loss = th.nn.functional.mse_loss(pred_values, signal[gt_mask])

                psnr = -10 * th.log10(mse_loss)

        # gt_mask = signal < 1.0
        accuracy = ((th.sigmoid(mask_pred_logits) > 0.5) == gt_mask).float().mean()
        
        # Create metrics dictionary
        metrics = {
            f'{prefix}loss': total_loss,
            f'{prefix}mse': mse_loss,
            f'{prefix}psnr': psnr,
            f'{prefix}accuracy': accuracy,
        }
        
        # Log all metrics
        for key, value in metrics.items():
            self.log(key, value, prog_bar=True, on_epoch=True, sync_dist=True)
            
        return {
            'loss': total_loss,
            'psnr': psnr,
        }
            
    def training_step(self, batch, batch_idx):
        """
        Training step for the Lightning module.
        
        Args:
            batch: Tuple of (coords, signal, idx)
            batch_idx: Index of the batch
            
        Returns:
            Loss value
        """
        coords, signal, left_bin_idx, w, idx = batch
        batch_size = coords.shape[0]

        latents = self.latents[idx]
        
        metrics = self._compute_metrics(coords, signal, latents, left_bin_idx, w, prefix='train_')
        loss = metrics['loss']
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        pass
    # def validation_step(self, batch, batch_idx):
    #     """
    #     Validation step for the Lightning module.
        
    #     Args:
    #         batch: Tuple of (coords, signal, idx)
    #         batch_idx: Index of the batch
    #     """
    #     coords, signal, latent, left_bin_idx, w, idx = batch
        
    #     gt_mask = signal < 1.0

    #     mask_pred_logits, pred_logits = self(coords.float(), latent).split((1, len(self.dfl_loss.bin_centres)), dim=-1)
    #     mask_pred_logits = mask_pred_logits.squeeze(-1)

    #     pred_values, _ = self.dfl_loss(pred_logits[gt_mask], left_bin_idx[gt_mask], w[gt_mask], signal[gt_mask])

    #     with th.no_grad():
    #         mse_loss = th.nn.functional.mse_loss(pred_values, signal[gt_mask])

    #     psnr = -10 * th.log10(mse_loss)

    #     accuracy = ((th.sigmoid(mask_pred_logits) > 0.5) == gt_mask).float().mean()

    #     self.log('val_psnr', psnr, prog_bar=True, on_epoch=True, sync_dist=True)
    #     self.log('val_acc', accuracy, prog_bar=True, on_epoch=True, sync_dist=True)
        
        
    def configure_optimizers(self):
        """
        Configure optimizers for the MAML implementation.
        """
        # We need to use manual optimization for MAML
        self.automatic_optimization = False
        
        params = [p for n, p in self.named_parameters() if 'sine_params' not in n]
        sine_params = [p for n, p in self.named_parameters() if 'sine_params' in n]
        # Create optimizer for all model parameters
        optimizer = th.optim.Adam(params, lr=self.hparams['learning_rate'], 
                                 betas=(self.hparams['beta1'], self.hparams['beta2']))
        # Add sine parameters to the optimizer with different learning rate
        if sine_params:
            optimizer.add_param_group({'params': sine_params, 'lr': self.hparams['learning_rate'],
                                       'betas': (self.hparams['beta1'], self.hparams['beta2']),
                                                 'weight_decay': self.hparams['incode_weight_decay']})
        
        warmup_scheduler = th.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.hparams['lr_warmup'],
                                                          end_factor=1, 
                                                          total_iters=self.hparams['lr_warmup_steps']*self.hparams['steps_per_epoch'])
        # scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[(s-self.hparams['lr_warmup_steps'])*self.hparams['steps_per_epoch'] for s in self.hparams['lr_milestones']], gamma=self.hparams['lr_gamma'])
        cosine_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams['lr_milestones'][-1]*self.hparams['steps_per_epoch'], eta_min=self.hparams['learning_rate']*self.hparams['lr_gamma'])
        final_scheduler = th.optim.lr_scheduler.ConstantLR(optimizer, factor=self.hparams['lr_gamma'], total_iters= self.hparams['steps_per_epoch']*self.hparams['max_epochs'] - self.hparams['lr_milestones'][-1]*self.hparams['steps_per_epoch'])

        # Combine warmup and step scheduler
        seq_scheduler = th.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler, final_scheduler], milestones=[self.hparams['lr_warmup_steps']*self.hparams['steps_per_epoch'], self.hparams['lr_milestones'][-1]*self.hparams['steps_per_epoch']])
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': seq_scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

class ModSirenMAML(ModSirenDFL):    
    def _init_latents(self, batch_size, device):
        """Initialize random latent codes for adaptation."""
        # return nn.Parameter(F.normalize(th.randn(batch_size, self.z_dim, device=device) * 0.01))
        return nn.Parameter(th.randn(batch_size, self.z_dim, device=device)*1e-3)
    
    def _adapt_latents(self, coords, signal, gt_mask, initial_latents, left_bin_idx, w, num_steps, create_graph=False):
        """
        Perform inner loop adaptation of latent codes.
        
        Args:
            coords: Coordinate tensor
            signal: Signal tensor
            initial_latents: Initial latent codes
            num_steps: Number of adaptation steps
            create_graph: Whether to create computational graph for higher-order gradients
            
        Returns:
            adapted_latents: Adapted latent codes
            latent_losses: List of loss values during adaptation
        """
        # gt_mask = signal < 1.0
        # gt_mask = th.ones_like(signal, dtype=th.bool)
        adapted_latents = initial_latents.clone()
        adapted_latents.requires_grad_(True)
        latent_losses = []
        
        # Inner loop adaptation
        for inner_step in range(num_steps):
            # Normalize latents
            # norm_latents = F.normalize(adapted_latents, dim=1)
            
            # Forward pass with current latents
            mask_pred_logits, pred_logits = self(coords.float(), adapted_latents).split((1, len(self.dfl_loss.bin_centres)), dim=-1)
            mask_pred_logits = mask_pred_logits.squeeze(-1)

            mask_loss = F.binary_cross_entropy_with_logits(mask_pred_logits, gt_mask.float())
        
            _, thpt_loss = self.dfl_loss(pred_logits[gt_mask], left_bin_idx[gt_mask], w[gt_mask], signal[gt_mask])

            loss = thpt_loss + self.hparams['mask_loss_weight'] * mask_loss

            # Store loss for logging
            latent_losses.append(loss.item())
            
            # Compute latent gradients
            latent_grad = th.autograd.grad(
                loss, 
                adapted_latents,
                create_graph=create_graph,
                retain_graph=True
            )[0]
            
            # Update latents
            # adapted_latents = F.normalize(adapted_latents - self.hparams['inner_lr']*(self.hparams['inner_gamma']**inner_step) * latent_grad)
            adapted_latents = adapted_latents - self.hparams['inner_lr']*(self.hparams['inner_gamma']**inner_step) * latent_grad
        
        return adapted_latents, latent_losses

    # def on_train_epoch_end(self):
        # for n, p in self.named_parameters():
        #     if 'sine_params' in n:
        #         a, b, c, d = p.unbind()
        #         self.log('params/'+n+'.a', a, prog_bar=False, on_epoch=True, sync_dist=True)
        #         self.log('params/'+n+'.b', b, prog_bar=False, on_epoch=True, sync_dist=True)
        #         self.log('params/'+n+'.c', c, prog_bar=False, on_epoch=True, sync_dist=True)
        #         self.log('params/'+n+'.d', d, prog_bar=False, on_epoch=True, sync_dist=True)
        
    def training_step(self, batch, batch_idx):
        """
        Implement First-Order MAML training step:
        1. Inner loop: Adapt latent codes for each task (without computational graph)
        2. Outer loop: Update model parameters using first-order approximation
        
        Returns:
            Loss value from the meta-optimization
        """
        coords, signal, gt_mask, left_bin_idx, w, idx = batch
        batch_size = coords.shape[0]
        
        # Make sure model is in training mode
        self.train()
        
        # Initialize random latents
        initial_latents = self._init_latents(batch_size, coords.device)
        
        # Inner loop adaptation
        adapted_latents, latent_losses = self._adapt_latents(
            coords, 
            signal, 
            gt_mask,
            initial_latents, 
            left_bin_idx, w,
            self.hparams['inner_loop_steps'],
            create_graph=True
        )
        
        # Compute metrics and meta-loss
        metrics = self._compute_metrics(coords, signal, gt_mask, adapted_latents, left_bin_idx, w, prefix='train_')
        meta_loss = metrics['loss']
        
        # Log inner loop adaptation loss
        # self.log('train_latent_loss', sum(latent_losses) / len(latent_losses) if latent_losses else 0, 
        #          sync_dist=True, prog_bar=True, on_epoch=True)
        latent_change = th.norm(adapted_latents - initial_latents, dim=-1).mean()
        self.log('train_latent_change', latent_change, sync_dist=True, prog_bar=True, on_epoch=True)
        self.log('train_latent_loss_ratio', latent_losses[-1] / latent_losses[0] if latent_losses else 0, 
                 sync_dist=True, prog_bar=True, on_epoch=True)
        
        # First-order MAML: compute gradients of the meta-loss
        self.manual_backward(meta_loss)

        if th.isnan(self.synthesizer.output.weight.grad).any():
            raise ValueError('NaN detected in synthesizer output weights, skipping step')
        
        # Apply optional gradient scaling
        if hasattr(self.hparams, 'grad_scale') and self.hparams['grad_scale'] > 1.0:
            for param in self.parameters():
                if param.grad is not None:
                    param.grad *= self.hparams['grad_scale']
        
        # Log gradient magnitudes
        with th.no_grad():
            grad_norm = 0.0
            param_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm += p.norm(2).item() ** 2
                    grad_norm += p.grad.norm(2).item() ** 2
            
            param_norm = param_norm ** 0.5
            grad_norm = grad_norm ** 0.5
            
            self.log('grad_norm', grad_norm, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log('param_norm', param_norm, on_epoch=True, sync_dist=True)
        
        for n, p in self.named_parameters():
            if 'sine_params' in n:
                layer_name = n.split('.')[1]

                a, b, c, d = p.unbind()
                self.log('params/'+layer_name+'.a', a, prog_bar=False, on_step=True, sync_dist=True)
                self.log('params/'+layer_name+'.b', b, prog_bar=False, on_step=True, sync_dist=True)
                self.log('params/'+layer_name+'.c', c, prog_bar=False, on_step=True, sync_dist=True)
                self.log('params/'+layer_name+'.d', d, prog_bar=False, on_step=True, sync_dist=True)

                grad_a, grad_b, grad_c, grad_d = p.grad.unbind()
                self.log('grads/'+layer_name+'.a', grad_a, prog_bar=False, on_step=True, sync_dist=True)
                self.log('grads/'+layer_name+'.b', grad_b, prog_bar=False, on_step=True, sync_dist=True)
                self.log('grads/'+layer_name+'.c', grad_c, prog_bar=False, on_step=True, sync_dist=True)
                self.log('grads/'+layer_name+'.d', grad_d, prog_bar=False, on_step=True, sync_dist=True)

        # Manual optimization required
        if (batch_idx + 1) % self.hparams['accumulate_grad_batches'] == 0:
            opt = self.optimizers()
            opt.step()
            if th.isnan(list(self.modulator.input.children())[0].weight).any():
                raise ValueError('NaN detected in modulator weights, skipping step')
            opt.zero_grad()

        scheduler = self.lr_schedulers()
        scheduler.step()

        return meta_loss

    # def on_train_epoch_end(self):
        
    def validation_step(self, batch, batch_idx):
        """
        Validation step using MAML adaptation on hold-out data.
        """
        coords, signal, gt_mask, left_bin_idx, w, idx = batch
        batch_size = coords.shape[0]
        
        # Initialize random latents
        initial_latents = self._init_latents(batch_size, coords.device)
        
        # Enable gradients for adaptation
        with th.enable_grad():
            for p in self.parameters():
                p.requires_grad_(True)
                
            # Adapt latents
            adapted_latents, _ = self._adapt_latents(
                coords, 
                signal, 
                gt_mask,
                initial_latents, 
                left_bin_idx, w,
                self.hparams['inner_loop_steps'],
                create_graph=False
            )
        
        # Evaluate with adapted latents
        with th.no_grad():
            metrics = self._compute_metrics(coords, signal, adapted_latents, left_bin_idx, w, prefix='val_')
        
        if self.training:
            # Ensure optimizer is reset
            opt = self.optimizers()
            opt.zero_grad()
        
        return {self.hparams['inner_loop_steps']: {
            'loss': metrics['loss'].item(),
            'psnr': metrics['psnr'].item(),
        }}
             
    def test_step(self, batch, batch_idx):
        """
        Test step using MAML adaptation on hold-out data.
        """
        coords, signal, gt_mask, left_bin_idx, w, idx = batch
        batch_size = coords.shape[0]
        
        # Initialize random latents
        initial_latents = self._init_latents(batch_size, coords.device)
        
        # Enable gradients for adaptation
        with th.enable_grad():
            for p in self.parameters():
                p.requires_grad_(True)
                
            # Adapt latents (using fixed 10 steps for test)
            adapted_latents, _ = self._adapt_latents(
                coords, 
                signal, 
                gt_mask,
                initial_latents, 
                left_bin_idx, w,
                self.hparams['inner_loop_steps'],
                create_graph=False
            )
        
        # Evaluate with adapted latents
        with th.no_grad():
            metrics = self._compute_metrics(coords, signal, adapted_latents, left_bin_idx, w, prefix='test_')

        metrics['latents'] = adapted_latents.cpu().numpy()

        return metrics
        
    def predict_step(self, batch, batch_idx):
        coords, signal, gt_mask, left_bin_idx, w, idx = batch
        batch_size = coords.shape[0]
        
        # Initialize random latents
        initial_latents = self._init_latents(batch_size, coords.device)
        
        # Enable gradients for adaptation
        with th.enable_grad():
            for p in self.parameters():
                p.requires_grad_(True)
                
            # Adapt latents (using fixed 10 steps for prediction)
            adapted_latents, _ = self._adapt_latents(
                coords, 
                signal,
                gt_mask, 
                initial_latents, 
                left_bin_idx, w,
                3,  # Fixed number of steps for prediction
                create_graph=False
            )
        
        # Return normalized latents and batch indices
        
        return adapted_latents, idx




# %%

class ModSirenParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description='ModSiren MAML Retrain')

        self.add_argument('--image_source', type=str, default='data/shuffled')
        self.add_argument('--angle_id', type=int, default=18)
        self.add_argument('--run_group', type=str, default='preshuffle')
        self.add_argument('--run_notes', type=str, default=None)
        self.add_argument('--max_epochs', type=int, default=40)
        self.add_argument('--level', type=int, default=5)
        self.add_argument('--num_configs', type=int, default=-1)
        self.add_argument('--batch_size', type=int, default=8)
        self.add_argument('--accumulate_grad_batches', type=int, default=1)
        self.add_argument('--project', type=str, default='modsiren')
        self.add_argument('--model_type', type=str, default='maml', choices='standard maml'.split())
        self.add_argument('--sine_layer_type', type=str, default='sine')
        self.add_argument('--input_features', type=int, default=3)
        self.add_argument('--num_dfl_bins', type=int, default=16)
        self.add_argument('--min_log_sigmat', type=float, default=-2)
        self.add_argument('--hidden_layers', type=int, default=8)
        self.add_argument('--mod_layers', type=int, default=8)
        self.add_argument('--first_layer_features', type=int, default=256)
        self.add_argument('--hidden_features', type=int, default=512)
        self.add_argument('--min_hidden_features', type=int, default=16)
        self.add_argument('--z_dim', type=int, default=256)
        self.add_argument('--inner_loop_steps', type=int, default=3)  # Reduced steps
        self.add_argument('--inner_lr', type=float, default=1e-1)     # Higher inner learning rate
        self.add_argument('--inner_gamma', type=float, default=1)     # Higher inner learning rate
        self.add_argument('--learning_rate', type=float, default=5e-4)
        self.add_argument('--lr_gamma', type=float, default=0.01)
        self.add_argument('--lr_milestones', type=int, nargs='+', default=[5, 35])
        self.add_argument('--lr_warmup_steps', type=int, default=10)
        self.add_argument('--lr_warmup', type=float, default=1e-8)
        self.add_argument('--lr_step', type=int, default=10)
        self.add_argument('--freq_scaling', type=float, default=30)
        self.add_argument('--init_freq_scaling', type=float, default=30)
        self.add_argument('--optimiser', type=str, default='adam', choices=['adam', 'sgd'])
        self.add_argument('--beta1', type=float, default=0.9)
        self.add_argument('--beta2', type=float, default=0.99)
        self.add_argument('--incode_weight_decay', type=float, default=0.0)
        self.add_argument('--grad_scale', type=float, default=1.0)  # Gradient scaling factor
        self.add_argument('--clip_grad_norm', type=float, default=10.0)  # Gradient clipping
        self.add_argument('--mask_loss_only', action='store_true')
        self.add_argument('--mask_loss_weight', type=float, default=1)
        self.add_argument('--dfl_beta', type=float, default=0)
        self.add_argument('--dfl_temperature', type=float, default=0.01)
        self.add_argument('--logger_on', action='store_true')
        self.add_argument('--logger_off', dest='logger_on', action='store_false')
        self.set_defaults(logger_on=True)
        
        # Add MAML-specific arguments
        self.add_argument('--first_order', action='store_true', help='Use first-order MAML approximation')
        self.set_defaults(first_order=False, logger_on=True)
        
        # We don't need these for pure MAML approach
        self.add_argument('--pretrained_latents', action='store_true', help='If True, use ModSiren, otherwise use ModSirenMAML')
        self.add_argument('--frozen_latents', action='store_true', help='Only used with ModSiren, not with MAML')
        self.set_defaults(pretrained_latents=False, frozen_latents=False)

if __name__ == '__main__':
    parser = ModSirenParser()
    # Add argument for resuming from a run ID
    parser.add_argument('--resume_id', type=str, default=None, help='Run ID to resume training from')
    args = vars(parser.parse_args())

    logger = None

    # Setup checkpoint callback with appropriate saving behavior
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='train_loss',  # Change this to whichever metric you're monitoring
        save_top_k=1,
        save_last=True,
        mode='min',
        filename='{epoch}-{train_loss:.4f}'
    )

    bins = np.load(f'{args["image_source"]}/level_{args["level"]}/cdf_bins_{args["angle_id"]}_{args["num_dfl_bins"]}_{args["min_log_sigmat"]:.02f}.npz')['thpt_inv_cdf']
    
    def modsiren_transform(df):
        """Transform parquet batch to your model's expected format."""
        # Example: assuming your parquet has columns: coords, signal, config_idx
        coords = th.from_numpy(df[['x', 'y', 'z']].to_numpy()).float()
        hit = th.from_numpy(~df['hit'].to_numpy()).bool()

        distance = df['distance'].to_numpy()
        scale_factor = 10**(args["min_log_sigmat"] + np.log10(2**(7 - args["level"])))
        distance = np.exp(-distance * scale_factor)

        right_bin_idx = np.searchsorted(bins, distance, side='right').clip(1, len(bins)-1)
        right_bin = bins[right_bin_idx]
        left_bin = bins[right_bin_idx - 1]
        bin_widths = right_bin - left_bin
        w = (right_bin - distance) / bin_widths

        config_idx = th.from_numpy(df['config_idx'].to_numpy()).long()
        return coords, distance, hit, right_bin_idx-1, w, config_idx

    args['output_features'] = args['num_dfl_bins'] + 1  # +1 for mask output
    
    # Create datamodule
    data_module = ParquetDataModule(
        train_path=f'{args["image_source"]}/level_{args["level"]}',  # Can use glob patterns
        val_path=None,
        batch_size=args['batch_size'],
        transform=modsiren_transform,
        columns=None  # Only load needed columns
    )

    total_samples = polars.scan_parquet(list(Path(data_module.train_path).glob('*.parquet'))).select(polars.len()).collect().item()
    num_devices = th.cuda.device_count() if th.cuda.is_available() else 1
    steps_per_epoch = int(np.ceil(total_samples / data_module.batch_size / num_devices))
    args['steps_per_epoch'] = steps_per_epoch

    # Setup wandb connection early if we're resuming or using pretrained latents
    if args['pretrained_latents']:
        api = wandb.Api()
        
        upstream_run = api.run(f'{args["project"]}/{args["run_id"]}')
        upstream_config = upstream_run.config
        args['z_dim'] = upstream_config['hidden_dim']
        model = ModSirenDFL(**args)
    elif args['model_type'] == 'standard':
        model = ModSirenDFL(**args)
        model.init_latents(len(get_config_indices(args['level'], 'train')))
    else:
        model = ModSirenMAML(**args)

    if args['logger_on']:
        logger = pl.loggers.WandbLogger(project='modsiren', 
                                        log_model='all', 
                                        group=args['run_group'], 
                                        notes=args['run_notes'])
        logger.watch(model, log='all')

    # Example of trainer setup with resume capability
    trainer = pl.Trainer(
        max_epochs=args['max_epochs'],
        accelerator='gpu',
        devices=-1,
        logger=logger,
        inference_mode=not args['pretrained_latents'],
        callbacks=[checkpoint_callback],
        gradient_clip_val=args['clip_grad_norm'],  # Apply gradient clipping
    )

    print(model)

    # Example with dummy data (not actually run)
    # train_loader, val_loader = get_data_loaders()
    trainer.fit(model, data_module)