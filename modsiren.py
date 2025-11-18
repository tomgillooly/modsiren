import argparse
# import matplotlib.pyplot as plt
import numpy as np
import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import wandb

from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler


# %%

class SineLayer(nn.Sequential):
    """Linear layer followed by a sine activation."""

    class _Activation(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, x):
            return th.sin(x*self.scale)

    def __init__(self, num_in, num_out, freq_scaling=30.0, init_scale=None):
        super().__init__()

        self.add_module("linear", nn.Linear(num_in, num_out))
        self.add_module("activation", SineLayer._Activation(freq_scaling))

        # Reset weights
        if init_scale is None:
            init_scale = np.sqrt(6.0 / num_in) / freq_scaling
        self.linear.weight.data.uniform_(-init_scale, init_scale)

        # Reset bias
        self.linear.bias.data.zero_()


# %%

class IncodeSineLayer(nn.Sequential):
    """Linear layer followed by a sine activation."""

    class _Activation(nn.Module):
        def __init__(self, dim, scale):
            super().__init__()
            self.scale = scale
            self.register_parameter('sine_params', nn.Parameter(th.zeros(4)))

            nn.init.trunc_normal_(self.sine_params, std=0.01)
            # nn.init.zeros_(self.sine_params)

        def forward(self, x):
            a, b, c, d = self.sine_params.unbind()
            # a = th.sqrt(th.exp(a))
            # b = th.exp(b)

            a = ((a-1)**2)/2
            b = ((b-1)**2)/2

            x = b * x
            x = x + c
            x = th.sin(self.scale * x)
            x = x * a
            x = x + d
            
            return x

    def __init__(self, num_in, num_out, freq_scaling=30.0, init_scale=None):
        super().__init__()

        self.add_module("linear", nn.Linear(num_in, num_out))
        self.add_module("activation", IncodeSineLayer._Activation(num_out, freq_scaling))


        # Reset weights
        if init_scale is None:
            init_scale = np.sqrt(6.0 / num_in) / freq_scaling
        self.linear.weight.data.uniform_(-init_scale, init_scale)

        # Reset bias
        self.linear.bias.data.zero_()


# %%


class Siren(nn.Sequential):
    def __init__(self, input_features=2, output_features=1,
                 first_layer_features=256,
                 hidden_layers=4, hidden_features=64,
                 min_hidden_features=16,
                 freq_scaling=30.0, init_freq=10.0,
                 sine_layer_type='sine'):
        super().__init__()

        if hidden_layers < 1:
            raise ValueError("Model should have at least 1 hidden layer.")

        if sine_layer_type == 'sine':
            sine_layer_fn = SineLayer
        elif sine_layer_type == 'incode':
            sine_layer_fn = IncodeSineLayer

        self.add_module(
            "input",
            sine_layer_fn(input_features, hidden_features,
                      init_scale=1.0/input_features,
                      freq_scaling=init_freq))

        in_dim = hidden_features
        out_dim = max(hidden_features // 2, min_hidden_features)

        for i in range(hidden_layers-1):
            self.add_module(
                f"hidden{i:02d}",
                sine_layer_fn(in_dim, out_dim,
                          freq_scaling=freq_scaling))

            in_dim = max(out_dim, min_hidden_features)
            out_dim = max(out_dim//2, min_hidden_features)

        self.add_module("output", nn.Linear(in_dim, output_features))



# %%

class ReLUMLP(nn.Sequential):
    def __init__(self, input_features=2, output_features=1,
                 first_layer_features=256,
                 hidden_layers=4, hidden_features=64,
                 min_hidden_features=16
                 ):
        super().__init__()

        if hidden_layers < 1:
            raise ValueError("Model should have at least 1 hidden layer.")

        self.add_module("input", 
                        nn.Sequential(
                            nn.Linear(input_features, hidden_features, bias=True),
                            nn.ReLU(),
                            )
        )

        # input_hidden_features = input_features

        in_dim = hidden_features
        out_dim = max(hidden_features // 2, min_hidden_features)

        for i in range(hidden_layers-1):
            self.add_module(f"hidden{i:02d}",
                            nn.Sequential(
                                # nn.LayerNorm(in_dim, elementwise_affine=False),
                                nn.Linear(in_dim, out_dim, bias=True),
                                nn.ReLU(),
                            ))
            
            in_dim = max(out_dim, min_hidden_features)
            out_dim = max(out_dim//2, min_hidden_features)
            # first_layer_features = hidden_featuress

        # self.add_module("output", nn.Linear(hidden_features, output_features))

        # Reset biases
        for n, p in self.named_parameters():
            if 'bias' in n:
                p.data.zero_()

                

# %%


class ModSiren(pl.LightningModule):
    """ModSiren model with PyTorch Lightning wrapper."""

    def __init__(self, input_features=2, output_features=2,
                 hidden_layers=4, hidden_features=64,
                 first_layer_features=256,
                 mod_layers=4,
                 init_freq_scaling=10.0,
                 freq_scaling=30.0, z_dim=64,
                 learning_rate=1e-3, lr_gamma=0.9,
                 *args, **kwargs):
        super().__init__()

        # Create modulator network
        self.add_module('modulator', ReLUMLP(
            input_features=z_dim, 
            first_layer_features=first_layer_features, 
            output_features=output_features,
            hidden_layers=mod_layers,
            hidden_features=hidden_features,
            min_hidden_features=kwargs['min_hidden_features']
            )
        )

        # Create synthesizer network
        self.add_module('synthesizer', Siren(
            input_features=input_features, 
            first_layer_features=first_layer_features,
            output_features=output_features,
            hidden_layers=hidden_layers,
            hidden_features=hidden_features,
            freq_scaling=freq_scaling,
            init_freq=init_freq_scaling,
            sine_layer_type=kwargs['sine_layer_type'],
            min_hidden_features=kwargs['min_hidden_features']
            )
        )

        self.mod_layers = mod_layers
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.lr_gamma = lr_gamma

        self.save_hyperparameters()

    def forward(self, coords, latent):
        """
        Forward pass of the ModSiren model.
        
        Args:
            coords: Input coordinates of shape [batch_size, 2]
            idx: Indices into latent codes buffer of shape [batch_size]
            
        Returns:
            Output signal of shape [batch_size, output_features]
        """
        # Get latent codes for the given indices
        # latent = latent / th.norm(latent, dim=-1, keepdim=True).detach()
        
        # # Modulator forward pass
        h_mod = self.modulator.input(latent)
        # h_mod_scale, h_mod_shift = h_mod.chunk(2, dim=-1)
        # h_mod = F.normalize(h_mod, dim=-1)
        
        # Synthesizer forward pass
        h_synth = self.synthesizer.input(coords) * h_mod #* h_mod_scale + h_mod_shift

        if th.isnan(h_mod).any() or th.isnan(h_synth).any() or th.isinf(h_mod).any() or th.isinf(h_synth).any():
            raise ValueError("NaN or Inf detected in modulator or synthesizer outputs.")
        
        # h_mod = latent
        # Process through hidden layers with modulation
        for i in range(self.mod_layers-1):
            synth_layer = getattr(self.synthesizer, f"hidden{i:02d}")
            mod_layer = getattr(self.modulator, f"hidden{i:02d}")

            h_mod = mod_layer(h_mod)
            # h_mod_scale, h_mod_shift = h_mod.chunk(2, dim=-1)
            # h_mod = F.normalize(h_mod, dim=-1)
            h_synth = synth_layer(h_synth) * h_mod # * h_mod_scale + h_mod_shift

            if th.isnan(h_mod).any() or th.isnan(h_synth).any() or th.isinf(h_mod).any() or th.isinf(h_synth).any():
                raise ValueError(f"NaN or Inf detected in modulator or synthesizer outputs at layer {i}.")

        for i in range(self.mod_layers-1, self.hidden_layers-1):
            synth_layer = getattr(self.synthesizer, f"hidden{i:02d}")
            h_synth = synth_layer(h_synth) * h_mod # * h_mod_scale + h_mod_shift
            
        # Final output layer
        out = self.synthesizer.output(h_synth) #* self.modulator.output(h_mod)
        
        return out
    
    def on_train_epoch_start(self):
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True)
        
    def training_step(self, batch, batch_idx):
        """
        Training step for the Lightning module.
        
        Args:
            batch: Tuple of (coords, signal, idx)
            batch_idx: Index of the batch
            
        Returns:
            Loss value
        """
        coords, signal, latent, idx = batch
        mask = signal < 1.0
        pred = self(coords.float(), F.normalize(latent, dim=-1))
        pred, logits = pred.unbind(dim=-1)
        mask_loss = th.nn.functional.binary_cross_entropy_with_logits(logits, mask.float())

        with th.no_grad():
            pred_mask = (th.sigmoid(logits) > 0.5).detach()
        # fp_mask = pred_mask & ~mask
        
        loss = th.nn.functional.mse_loss(th.sigmoid(pred[mask]), signal[mask].float())
        # if fp_mask.any():
        #     fp_loss = th.nn.functional.mse_loss(th.sigmoid(pred[fp_mask]), th.ones(fp_mask.sum()).to(signal.device))
        #     loss += fp_loss
        
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('train_mask_loss', mask_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('train_acc', (pred_mask == mask).float().mean(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('train_psnr', -10*th.log10(loss), prog_bar=True, on_epoch=True, sync_dist=True)
        return loss + mask_loss*self.hparams['mask_loss_weight']
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the Lightning module.
        
        Args:
            batch: Tuple of (coords, signal, idx)
            batch_idx: Index of the batch
        """
        coords, signal, latent, idx = batch
        mask = signal < 1.0
        pred = self(coords.float(), latent)
        pred, logits = pred.unbind(dim=-1)
        
        pred_mask = (th.sigmoid(logits) > 0.5).detach()

        loss = th.nn.functional.mse_loss(th.sigmoid(pred[mask]), signal[mask].float())
        
        self.log('val_acc', (pred_mask == mask).float().mean(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_psnr', -10*th.log10(loss), prog_bar=True, on_epoch=True, sync_dist=True)
    
    def configure_optimizers(self):
        """
        Configure optimizers for the Lightning module.
        
        Returns:
            Optimizer for model parameters
        """
        model_params = [p for n, p in self.named_parameters() if not n.startswith('latent_codes')]
        optimizer = th.optim.Adam(model_params, lr=self.hparams['learning_rate'], betas=(self.hparams['beta1'], self.hparams['beta2']))

        if hasattr(self, 'latent_codes') and isinstance(self.latent_codes, nn.Parameter):
            latent_params = [self.latent_codes]
            optimizer.add_param_group({'params': latent_params, 'betas': (0.5, 0.5)})
        scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams['lr_step'], gamma=self.hparams['lr_gamma'])
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

class ModSirenMAML(ModSiren):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # No need to store latent codes as parameters
        # We'll randomly initialize them for each task during training
        
        # Store z_dim for initializing random latents
        self.z_dim = kwargs['z_dim']
        
    def _inner_loop_update(self, coords, signal, init_latents=None, first_order=False, is_validation=False):
        """
        Perform inner loop updates for MAML.
        
        Args:
            coords: Input coordinates
            signal: Target signals
            init_latents: Initial latent vectors (randomly initialized if None)
            first_order: Whether to use first-order approximation
            is_validation: If True, don't compute or track gradients (for validation/testing)
            
        Returns:
            adapted_latents: Adapted latent codes after inner loop updates
            losses: List of losses during adaptation
        """
        # Get mask from signal
        mask = signal < 1.0
        batch_size = coords.shape[0]
        
        # Randomly initialize latent codes if not provided
        if init_latents is None:
            # Random initialization of latent vectors
            if is_validation:
                # For validation, we don't need gradients
                with th.no_grad():
                    adapted_latents = th.zeros((batch_size, self.z_dim), device=coords.device)
            else:
                adapted_latents = th.zeros((batch_size, self.z_dim), device=coords.device).requires_grad_(True)
        else:
            if is_validation:
                adapted_latents = init_latents.clone()
            else:
                adapted_latents = init_latents.clone().requires_grad_(True)
        
        losses = []
        
        for inner_step in range(self.hparams['inner_loop_steps']):
            # Normalize latents
            norm_latents = F.normalize(adapted_latents, dim=1)
            
            # Forward pass with current latents
            pred = self(coords.float(), norm_latents)
            pred, logits = pred.unbind(dim=-1)
            
            # Compute losses
            mask_loss = th.nn.functional.binary_cross_entropy_with_logits(logits, mask.float())
            recon_loss = th.nn.functional.mse_loss(th.sigmoid(pred[mask]), signal[mask].float())
            loss = recon_loss + mask_loss * self.hparams['mask_loss_weight']
            
            # Store loss for logging
            losses.append(loss.item())
            
            if is_validation:
                # For validation, just compute the gradient using autograd.grad
                # but don't build the computation graph
                with th.no_grad():
                    # Compute gradients using backward+grad instead of autograd.grad for validation
                    # This is more memory efficient and doesn't store the graph
                    if th.is_grad_enabled():
                        # Create a temporary copy for gradient calculation
                        temp_latents = norm_latents.clone().detach().requires_grad_(True)
                        
                        # Forward pass with temp latents
                        temp_pred = self(coords.float(), temp_latents)
                        temp_pred, temp_logits = temp_pred.unbind(dim=-1)
                        
                        # Compute temporary loss
                        temp_mask_loss = th.nn.functional.binary_cross_entropy_with_logits(temp_logits, mask.float())
                        temp_recon_loss = th.nn.functional.mse_loss(th.sigmoid(temp_pred[mask]), signal[mask].float())
                        temp_loss = temp_recon_loss + temp_mask_loss * self.hparams['mask_loss_weight']
                        
                        # Compute gradient
                        temp_loss.backward()
                        grads = temp_latents.grad
                    else:
                        # If gradients are disabled, just use zeros as gradients
                        grads = th.zeros_like(adapted_latents)
                    
                    # Update latents
                    adapted_latents = adapted_latents - self.hparams['inner_lr'] * grads
            else:
                # For training, compute gradients and build computation graph for meta-learning
                grads = th.autograd.grad(
                    loss, 
                    adapted_latents,
                    create_graph=not first_order,  # Create computation graph for second-order gradients
                    retain_graph=True
                )
                
                # Update parameters manually (simulating optimizer step)
                with th.no_grad():
                    adapted_latents = adapted_latents - self.hparams['inner_lr'] * grads[0]
        
        return adapted_latents, losses
        
    def training_step(self, batch, batch_idx):
        """
        Implement MAML training step:
        1. Inner loop: Adapt latent codes for each task
        2. Outer loop: Update model parameters using gradients through the adaptation
        
        Returns:
            Loss value from the meta-optimization
        """
        # Manual optimization required
        opt = self.optimizers()
        opt.zero_grad()
        
        coords, signal, idx = batch
        mask = signal < 1.0
        
        # Make sure model is in training mode
        self.train()
        
        # Initialize random latents that require gradients
        batch_size = coords.shape[0]
        init_latents = th.zeros((batch_size, self.z_dim), device=coords.device, requires_grad=True)
        
        # Manually perform the inner loop adaptation
        adapted_latents = init_latents.clone()
        latent_losses = []
        
        for inner_step in range(self.hparams['inner_loop_steps']):
            # Normalize latents
            norm_latents = adapted_latents / th.norm(adapted_latents, dim=-1, keepdim=True).detach()
            
            # Forward pass with current latents
            pred = self(coords.float(), norm_latents)
            pred, logits = pred.unbind(dim=-1)
            
            # Compute losses
            mask_loss = th.nn.functional.binary_cross_entropy_with_logits(logits, mask.float())
            recon_loss = th.nn.functional.mse_loss(th.sigmoid(pred[mask]), signal[mask].float())
            loss = recon_loss + mask_loss * self.hparams['mask_loss_weight']
            
            # Store loss for logging
            latent_losses.append(loss.item())
            
            # Compute gradients
            # Important: Create a graph so we can differentiate through the adaptation
            grads = th.autograd.grad(
                loss, 
                adapted_latents,
                create_graph=True,  # Always use create_graph=True for training
                retain_graph=True,
                allow_unused=False
            )
            
            # Update parameters manually (simulating optimizer step)
            adapted_latents = adapted_latents - self.hparams['inner_lr'] * grads[0]
        
        # Normalize adapted latents
        adapted_latents = adapted_latents / th.norm(adapted_latents, dim=-1, keepdim=True).detach()
        
        # 2. Outer loop - compute loss with adapted parameters
        pred = self(coords.float(), adapted_latents)
        pred, logits = pred.unbind(dim=-1)
        
        # Compute losses with adapted latents
        mask_loss = th.nn.functional.binary_cross_entropy_with_logits(logits, mask.float())
        recon_loss = th.nn.functional.mse_loss(th.sigmoid(pred[mask]), signal[mask].float())
        meta_loss = recon_loss + mask_loss * self.hparams['mask_loss_weight']
        
        # Manual backward for model parameters
        # This will compute gradients through the entire adaptation process
        self.manual_backward(meta_loss)
        
        # Step optimizer (updates model parameters)
        opt.step()
        
        # Log metrics
        self.log('train_loss', meta_loss, prog_bar=True, on_epoch=True)
        self.log('train_latent_loss', sum(latent_losses) / len(latent_losses) if latent_losses else 0, 
                prog_bar=True, on_epoch=True)
        self.log('train_acc', ((th.sigmoid(logits) > 0.5) == mask).float().mean(), prog_bar=True, on_epoch=True)
        self.log('train_psnr', -10*th.log10(recon_loss), prog_bar=True, on_epoch=True)
        
        return meta_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step using MAML adaptation on hold-out data.
        
        For proper MAML validation, we:
        1. Randomly initialize latents
        2. Adapt them using the inner loop (similar to training but without gradient tracking)
        3. Evaluate performance after adaptation
        4. Optionally, evaluate with more adaptation steps than training
        """
        coords, signal, idx = batch
        mask = signal < 1.0
        
        # Track metrics for different numbers of adaptation steps
        metrics = {}
        
        # Test adaptation with the same number of steps as training
        adapted_latents, adaptation_losses = self._inner_loop_update(
            coords, signal, first_order=True, is_validation=True  # Use validation mode
        )
        adapted_latents = adapted_latents / th.norm(adapted_latents, dim=-1, keepdim=True).detach()
        
        # Forward pass with adapted latents
        with th.no_grad():
            pred = self(coords.float(), adapted_latents)
            pred, logits = pred.unbind(dim=-1)
            
            # Compute validation metrics
            recon_loss = th.nn.functional.mse_loss(th.sigmoid(pred[mask]), signal[mask].float())
            accuracy = ((th.sigmoid(logits) > 0.5) == mask).float().mean()
            psnr = -10 * th.log10(recon_loss)
            
            # Log metrics for standard adaptation
            self.log('val_acc', accuracy, prog_bar=True, on_epoch=True)
            self.log('val_psnr', psnr, prog_bar=True, on_epoch=True)
            self.log('val_loss', recon_loss, on_epoch=True)
            
            # Store in metrics dict
            metrics[self.hparams['inner_loop_steps']] = {
                'loss': recon_loss.item(),
                'psnr': psnr.item(),
                'acc': accuracy.item()
            }
        
        # Test adaptation with more steps to evaluate learning trajectory
        # This helps us understand if the model benefits from additional adaptation
        extra_steps = [5, 10]  # Test with 5 and 10 adaptation steps
        
        for num_steps in extra_steps:
            if num_steps <= self.hparams['inner_loop_steps']:
                continue  # Skip if we've already evaluated with this many or more steps
                
            # Save original inner loop steps
            original_steps = self.hparams['inner_loop_steps']
            
            # Temporarily change inner loop steps
            self.hparams['inner_loop_steps'] = num_steps
            
            # Perform adaptation with more steps
            adapted_latents_extra, _ = self._inner_loop_update(
                coords, signal, first_order=True, is_validation=True  # Use validation mode
            )
            adapted_latents_extra = adapted_latents_extra / th.norm(adapted_latents_extra, dim=-1, keepdim=True).detach()
            
            # Evaluate with extra adaptation steps
            with th.no_grad():
                pred_extra = self(coords.float(), adapted_latents_extra)
                pred_extra, logits_extra = pred_extra.unbind(dim=-1)
                
                # Compute metrics
                recon_loss_extra = th.nn.functional.mse_loss(th.sigmoid(pred_extra[mask]), signal[mask].float())
                accuracy_extra = (th.sigmoid(logits_extra) > 0.5).float().mean()
                psnr_extra = -10 * th.log10(recon_loss_extra)
                
                # Log metrics for extra adaptation steps
                self.log(f'val_psnr_{num_steps}steps', psnr_extra, on_epoch=True)
                self.log(f'val_acc_{num_steps}steps', accuracy_extra, on_epoch=True)
                
                # Store in metrics dict
                metrics[num_steps] = {
                    'loss': recon_loss_extra.item(),
                    'psnr': psnr_extra.item(),
                    'acc': accuracy_extra.item()
                }
            
            # Restore original inner loop steps
            self.hparams['inner_loop_steps'] = original_steps
        
        # Log the learning curve metrics across different numbers of steps
        # This is useful to see how quickly the model adapts
        steps = sorted(metrics.keys())
        psnrs = [metrics[s]['psnr'] for s in steps]
        accs = [metrics[s]['acc'] for s in steps]
        
        # Log improvement from additional adaptation steps
        if len(steps) > 1:
            psnr_improvement = psnrs[-1] - psnrs[0]
            acc_improvement = accs[-1] - accs[0]
            self.log('val_psnr_improvement', psnr_improvement, on_epoch=True)
            self.log('val_acc_improvement', acc_improvement, on_epoch=True)
                
        return metrics

    def test_step(self, batch, batch_idx):
        """
        Test step for evaluating on completely held-out data.
        Uses the same adaptation process as validation but with potentially more steps.
        """
        coords, signal, idx = batch
        mask = signal < 1.0
        
        # Use more adaptation steps for testing to see full adaptation capabilities
        test_inner_steps = max(10, self.hparams['inner_loop_steps'] * 2)
        
        # Save original inner loop steps
        original_steps = self.hparams['inner_loop_steps']
        
        # Temporarily change inner loop steps
        self.hparams['inner_loop_steps'] = test_inner_steps
        
        # Adapt latents with more steps for thorough evaluation
        adapted_latents, adaptation_losses = self._inner_loop_update(
            coords, signal, first_order=True, is_validation=True  # Use validation mode
        )
        adapted_latents = adapted_latents / th.norm(adapted_latents, dim=-1, keepdim=True).detach()
        
        # Forward pass with adapted latents
        with th.no_grad():
            pred = self(coords.float(), adapted_latents)
            pred, logits = pred.unbind(dim=-1)
            
            # Compute test metrics
            recon_loss = th.nn.functional.mse_loss(th.sigmoid(pred[mask]), signal[mask].float())
            accuracy = ((th.sigmoid(logits) > 0.5) == mask).float().mean()
            psnr = -10 * th.log10(recon_loss)
            
            # Log metrics
            self.log('test_acc', accuracy, prog_bar=True)
            self.log('test_psnr', psnr, prog_bar=True)
            self.log('test_loss', recon_loss)
            
            # Log adaptation curve
            self.log('test_final_step_loss', adaptation_losses[-1] if adaptation_losses else float('nan'))
            self.log('test_initial_step_loss', adaptation_losses[0] if adaptation_losses else float('nan'))
            
            # Calculate adaptation speed (how quickly loss decreases)
            if len(adaptation_losses) > 1:
                adaptation_speed = (adaptation_losses[0] - adaptation_losses[-1]) / len(adaptation_losses)
                self.log('test_adaptation_speed', adaptation_speed)
        
        # Restore original inner loop steps
        self.hparams['inner_loop_steps'] = original_steps
        
        return {
            'loss': recon_loss.item(),
            'psnr': psnr.item(),
            'acc': accuracy.item(),
            'adaptation_losses': adaptation_losses
        }

    def predict_step(self, batch, batch_idx):
        # Save original inner loop steps
        original_steps = self.hparams['inner_loop_steps']
        
        # Temporarily change inner loop steps
        self.hparams['inner_loop_steps'] = 10
        
        coords, signal, idx = batch
        mask = signal < 1.0
        
        # Perform adaptation with more steps
        adapted_latents, _ = self._inner_loop_update(
            coords, signal, first_order=True, is_validation=True  # Use validation mode
        )
        adapted_latents = adapted_latents / th.norm(adapted_latents, dim=-1, keepdim=True).detach()
              
        # Restore original inner loop steps
        self.hparams['inner_loop_steps'] = original_steps

        return adapted_latents
    
    def configure_optimizers(self):
        """
        Configure optimizers for the MAML implementation.
        
        Returns:
            Optimizer for model parameters
        """
        # We need to use manual optimization for MAML
        self.automatic_optimization = False
        
        # Create optimizer for all model parameters
        if self.hparams['optimiser'] == 'sgd':
            optimizer = th.optim.SGD(self.parameters(), lr=self.hparams['learning_rate'])
        else:
            optimizer = th.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'], 
                                    betas=(self.hparams['beta1'], self.hparams['beta2']))
        
        scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams['lr_step'], 
                                                gamma=self.hparams['lr_gamma'])
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

class ModSirenFirstOrderMAML(ModSiren):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Store z_dim for initializing random latents
        self.z_dim = kwargs['z_dim']
        
    def training_step(self, batch, batch_idx):
        """
        Implement First-Order MAML training step:
        1. Inner loop: Adapt latent codes for each task (without computational graph)
        2. Outer loop: Update model parameters using first-order approximation
        
        Returns:
            Loss value from the meta-optimization
        """
        
        coords, signal, idx = batch
        mask = signal < 1.0
        
        # Make sure model is in training mode
        self.train()
        
        # Initialize random latents
        batch_size = coords.shape[0]
        initial_latents = nn.Parameter(th.randn(batch_size, self.z_dim, device=coords.device)*.01)
        adapted_latents = initial_latents.clone()
        
        # Save initial model parameters
        initial_params = {name: param.clone() for name, param in self.named_parameters()}
        
        # Store inner loop losses for logging
        latent_losses = []
        
        #----------------------------------------
        # 1. Inner loop - adapt latent codes
        #----------------------------------------
        for inner_step in range(self.hparams['inner_loop_steps']):
            # Normalize latents
            norm_latents = F.normalize(adapted_latents, dim=1)
            
            # Forward pass with current latents
            pred = self(coords.float(), norm_latents)
            pred, logits = pred.unbind(dim=-1)
            
            # Compute losses
            mask_loss = th.nn.functional.binary_cross_entropy_with_logits(logits, mask.float())
            recon_loss = th.nn.functional.mse_loss(th.sigmoid(pred[mask]), signal[mask].float())
            loss = recon_loss + mask_loss * self.hparams['mask_loss_weight']
            
            # Store loss for logging
            latent_losses.append(loss.item())
            
            # Compute latent gradients
            latent_grad = th.autograd.grad(
                    loss, 
                    adapted_latents,
                    create_graph=True,
                    retain_graph=True
                )[0]
            
            # Update latents (no gradient tracking)
            # with th.no_grad():
            adapted_latents = adapted_latents.clone() - self.hparams['inner_lr'] * latent_grad
                # adapted_latents = adapted_latents.detach()
                # adapted_latents.requires_grad_(True)
        
        #----------------------------------------
        # 2. Outer loop - compute meta-loss and update model parameters
        #----------------------------------------
        # Normalize final adapted latents
        # with th.no_grad():
        norm_latents = F.normalize(adapted_latents, dim=-1)
            # norm_latents = norm_latents.detach()


        config_vmf = vmf_regularizer(norm_latents)
        config_koleo = koleo_regularizer(norm_latents, k=1)
        
        # Forward pass with adapted latents
        pred = self(coords.float(), norm_latents)
        pred, logits = pred.unbind(dim=-1)
        pred_mask = th.sigmoid(logits) > 0.5
        
        # Compute losses with adapted latents
        mask_loss = th.nn.functional.binary_cross_entropy_with_logits(logits, mask.float())
        recon_loss = th.nn.functional.mse_loss(th.sigmoid(pred[mask]), signal[mask].float())
        meta_loss = recon_loss + mask_loss * self.hparams['mask_loss_weight']
        
        # First-order MAML: simply compute gradients of the meta-loss
        # with respect to model parameters
        self.manual_backward(meta_loss)
        
        # Apply optional gradient scaling to address small gradients
        if hasattr(self.hparams, 'grad_scale') and self.hparams['grad_scale'] > 1.0:
            for param in self.parameters():
                if param.grad is not None:
                    param.grad *= self.hparams['grad_scale']        
                
        # Log metrics
        self.log('train_loss', meta_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('train_mask_loss', mask_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('train_latent_loss', sum(latent_losses) / len(latent_losses) if latent_losses else 0, sync_dist=True,
                prog_bar=True, on_epoch=True)
        self.log('train_acc', (pred_mask == mask).float().mean(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('train_psnr', -10*th.log10(recon_loss), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('train_config_koleo', config_koleo, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('train_config_vmf', config_vmf, prog_bar=True, on_epoch=True, sync_dist=True)
        
        # Optionally log gradient magnitudes to debug
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
            # self.log('effective_lr', self.hparams['learning_rate'] * grad_norm / param_norm, on_epoch=True, sync_dist=True)
        
        # Manual optimization required
        if (batch_idx + 1) % self.hparams['accumulate_grad_batches'] == 0:
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()

        return meta_loss

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        scheduler.step()
        
    def validation_step(self, batch, batch_idx):
        """
        Validation step using MAML adaptation on hold-out data.
        """
        coords, signal, idx = batch
        mask = signal < 1.0
        
        # Track metrics for different numbers of adaptation steps
        metrics = {}
        
        #------------------------
        # Adaptation process
        #------------------------
        # Initialize random latents
        batch_size = coords.shape[0]
        adapted_latents = nn.Parameter(th.randn(batch_size, self.z_dim, device=coords.device)*0.01)
        adapted_latents = adapted_latents.requires_grad_(True)

        for p in self.parameters():
            p.requires_grad_(True)

        adaptation_losses = []
        
        with th.enable_grad():
            # Inner loop adaptation
            for inner_step in range(self.hparams['inner_loop_steps']):
                # Normalize latents
                norm_latents = F.normalize(adapted_latents)
                
                # Forward pass with current latents
                pred = self(coords.float(), norm_latents)
                pred, logits = pred.unbind(dim=-1)
                
                # Compute losses
                mask_loss = th.nn.functional.binary_cross_entropy_with_logits(logits, mask.float())
                recon_loss = th.nn.functional.mse_loss(th.sigmoid(pred[mask]), signal[mask].float())
                loss = recon_loss + mask_loss * self.hparams['mask_loss_weight']
                
                # Store loss for logging
                adaptation_losses.append(loss.item())
                
                # Compute gradient
                latent_grad = th.autograd.grad(
                        loss, 
                        adapted_latents
                    )[0]
                
                # Update latents
                adapted_latents = adapted_latents - self.hparams['inner_lr'] * latent_grad
        
        # Normalize adapted latents for final evaluation
        with th.no_grad():
            norm_latents = F.normalize(adapted_latents)

            config_vmf = vmf_regularizer(norm_latents)
            config_koleo = koleo_regularizer(norm_latents, k=1)
            
            # Final evaluation
            pred = self(coords.float(), norm_latents)
            pred, logits = pred.unbind(dim=-1)
            pred_mask = th.sigmoid(logits) > 0.5
            
            # Compute validation metrics
            recon_loss = th.nn.functional.mse_loss(th.sigmoid(pred[mask]), signal[mask].float())
            accuracy = (pred_mask == mask).float().mean()
            psnr = -10 * th.log10(recon_loss)
            
            # Log metrics
            self.log('val_acc', accuracy, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log('val_psnr', psnr, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log('val_loss', recon_loss, on_epoch=True, sync_dist=True)
            self.log('val_config_koleo', config_koleo, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log('val_config_vmf', config_vmf, prog_bar=True, on_epoch=True, sync_dist=True)
            
            # Store in metrics dict
            metrics[self.hparams['inner_loop_steps']] = {
                'loss': recon_loss.item(),
                'psnr': psnr.item(),
                'acc': accuracy.item()
            }

        
        opt = self.optimizers()
        opt.zero_grad()
        
        return metrics
             
    def test_step(self, batch, batch_idx):
        """
        Test step using MAML adaptation on hold-out data.
        """
        coords, signal, idx = batch
        mask = signal < 1.0
        
        # Track metrics for different numbers of adaptation steps
        metrics = {}
        
        #------------------------
        # Adaptation process
        #------------------------
        # Initialize random latents
        batch_size = coords.shape[0]
        adapted_latents = nn.Parameter(th.randn(batch_size, self.z_dim, device=coords.device)*0.01)
        adapted_latents = adapted_latents.requires_grad_(True)

        for p in self.parameters():
            p.requires_grad_(True)

        adaptation_losses = []
        
        with th.enable_grad():
            # Inner loop adaptation
            for inner_step in range(10):
                # Normalize latents
                norm_latents = F.normalize(adapted_latents)
                
                # Forward pass with current latents
                pred = self(coords.float(), norm_latents)
                pred, logits = pred.unbind(dim=-1)
                
                # Compute losses
                mask_loss = th.nn.functional.binary_cross_entropy_with_logits(logits, mask.float())
                recon_loss = th.nn.functional.mse_loss(th.sigmoid(pred[mask]), signal[mask].float())
                loss = recon_loss + mask_loss * self.hparams['mask_loss_weight']
                
                # Store loss for logging
                adaptation_losses.append(loss.item())
                
                # Compute gradient
                latent_grad = th.autograd.grad(
                        loss, 
                        adapted_latents
                    )[0]
                
                # Update latents
                adapted_latents = adapted_latents - self.hparams['inner_lr'] * latent_grad
        
        # Normalize adapted latents for final evaluation
        with th.no_grad():
            norm_latents = F.normalize(adapted_latents)
            
            # Final evaluation
            pred = self(coords.float(), norm_latents)
            pred, logits = pred.unbind(dim=-1)
            pred_mask = th.sigmoid(logits) > 0.5
            pred[~pred_mask] = 1.0
            
            # Compute validation metrics
            recon_loss = th.nn.functional.mse_loss(th.sigmoid(pred[mask]), signal[mask].float())
            accuracy = (pred_mask == mask).float().mean()
            psnr = -10 * th.log10(recon_loss)
            
            # Log metrics
            self.log('test_acc', accuracy, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
            self.log('test_psnr', psnr, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
        
        
    def predict_step(self, batch, batch_idx):
        coords, signal, idx = batch
        mask = signal < 1.0
        
        # Track metrics for different numbers of adaptation steps
        metrics = {}
        
        #------------------------
        # Adaptation process
        #------------------------
        # Initialize random latents
        batch_size = coords.shape[0]
        adapted_latents = nn.Parameter(th.randn(batch_size, self.z_dim, device=coords.device)*0.01)
        adapted_latents = adapted_latents.requires_grad_(True)

        for p in self.parameters():
            p.requires_grad_(True)

        adaptation_losses = []
        
        with th.enable_grad():
            # Inner loop adaptation
            for inner_step in range(10):
                # Normalize latents
                norm_latents = F.normalize(adapted_latents)
                
                # Forward pass with current latents
                pred = self(coords.float(), norm_latents)
                pred, logits = pred.unbind(dim=-1)
                
                # Compute losses
                mask_loss = th.nn.functional.binary_cross_entropy_with_logits(logits, mask.float())
                recon_loss = th.nn.functional.mse_loss(th.sigmoid(pred[mask]), signal[mask].float())
                loss = recon_loss + mask_loss * self.hparams['mask_loss_weight']
                
                # Store loss for logging
                adaptation_losses.append(loss.item())
                
                # Compute gradient
                latent_grad = th.autograd.grad(
                        loss, 
                        adapted_latents
                    )[0]
                
                # Update latents
                adapted_latents = adapted_latents - self.hparams['inner_lr'] * latent_grad
        
        # Normalize adapted latents for final evaluation
        norm_latents = F.normalize(adapted_latents)
        
        return norm_latents, idx
        
    def configure_optimizers(self):
        """
        Configure optimizers for the MAML implementation.
        """
        # We need to use manual optimization for MAML
        self.automatic_optimization = False
        
        # Create optimizer for all model parameters
        optimizer = th.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'], 
                                 betas=(self.hparams['beta1'], self.hparams['beta2']))
        
        scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams['lr_step'], 
                                                gamma=self.hparams['lr_gamma'])
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }




# %%


# Example usage for image representation task
class ImageDataModule(pl.LightningDataModule):
    """
    Data module for image representation tasks.
    """
    def __init__(self, level, batch_size=1024, run_id=None, num_configs=-1, pretrained_latents=True, *args, **kwargs):
        super().__init__()
        self.level = level
        self.batch_size = batch_size

        if num_configs == -1:
            self.num_configs = len(get_config_indices(level, 'train'))
        else:
            self.num_configs = num_configs

        if pretrained_latents:
            self.run_id = run_id
            self.project = kwargs['project']
            def dataset_fn(*args, **kwargs):
                return FlattenedImageLatentNerfDataset(*args, **kwargs, run_id=self.run_id, project=self.project)
            self.dataset_fn = dataset_fn
        else:
            self.dataset_fn = FlattenedImageNerfDataset
        

    def setup(self, stage=None):
        self.train_dataset = self.dataset_fn(self.level, phase='train', num_configs=self.num_configs)
        self.val_dataset = self.dataset_fn(self.level, phase='val', num_configs=self.num_configs)
        self.pred_dataset = self.dataset_fn(self.level, phase='predict', num_configs=self.num_configs)
    
    def train_dataloader(self):
        num_workers = int(os.environ.get('SLURM_CPUS_ON_NODE', 1)) // self.trainer.world_size
        # num_workers = (num_cpus-1)
        
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=num_workers, drop_last=False, persistent_workers=num_workers > 0)
    
    def val_dataloader(self):
        num_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE', 1))
        num_workers = (num_cpus-1)
        num_workers = int(os.environ.get('SLURM_CPUS_ON_NODE', 1)) // self.trainer.world_size
        
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=num_workers, drop_last=False, persistent_workers=num_workers > 0)
    
    def pred_dataloader(self):
        num_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE', 1))
        num_workers = (num_cpus-1)
        num_workers = int(os.environ.get('SLURM_CPUS_ON_NODE', 1)) // self.trainer.world_size
        
        return DataLoader(self.pred_dataset, shuffle=False, batch_size=self.batch_size, num_workers=num_workers, drop_last=False, persistent_workers=num_workers > 0)

class ModSirenParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description='ModSiren MAML Retrain')

        self.add_argument('--run_group', type=str, default=None)
        self.add_argument('--run_notes', type=str, default=None)
        self.add_argument('--max_epochs', type=int, default=40)
        self.add_argument('--level', type=int, default=2)
        self.add_argument('--num_configs', type=int, default=-1)
        self.add_argument('--batch_size', type=int, default=8)
        self.add_argument('--accumulate_grad_batches', type=int, default=1)
        self.add_argument('--project', type=str, default='image_validity')
        self.add_argument('--run_id', type=str, default='psmbkk6s')
        self.add_argument('--sine_layer_type', type=str, default='incode')
        self.add_argument('--input_features', type=int, default=3)
        self.add_argument('--output_features', type=int, default=2)
        self.add_argument('--hidden_layers', type=int, default=8)
        self.add_argument('--hidden_features', type=int, default=256)
        self.add_argument('--z_dim', type=int, default=128)
        self.add_argument('--inner_loop_steps', type=int, default=3)  # Reduced steps
        self.add_argument('--inner_lr', type=float, default=1)     # Higher inner learning rate
        self.add_argument('--learning_rate', type=float, default=1e-3)
        self.add_argument('--lr_gamma', type=float, default=0.1)
        self.add_argument('--lr_step', type=int, default=10)
        self.add_argument('--freq_scaling', type=float, default=30)
        self.add_argument('--init_freq_scaling', type=float, default=30)
        self.add_argument('--optimiser', type=str, default='adam', choices=['adam', 'sgd'])
        self.add_argument('--beta1', type=float, default=0.9)
        self.add_argument('--beta2', type=float, default=0.99)
        self.add_argument('--grad_scale', type=float, default=1.0)  # Gradient scaling factor
        self.add_argument('--clip_grad_norm', type=float, default=None)  # Gradient clipping
        self.add_argument('--mask_loss_weight', type=float, default=1e-2)
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

    data_module = ImageDataModule(**args)

    # Setup wandb connection early if we're resuming or using pretrained latents
    if args['pretrained_latents'] or args['resume_id']:
        api = wandb.Api()

    # Handle pretrained latents case
    if args['pretrained_latents']:
        upstream_run = api.run(f'{args["project"]}/{args["run_id"]}')
        upstream_config = upstream_run.config
        args['z_dim'] = upstream_config['hidden_dim']
        model = ModSiren(**args)
    else:
        model = ModSirenFirstOrderMAML(**args)

    # Setup logger
    if args['logger_on']:
        # If resuming, use the same run ID
        if args['resume_id']:
            logger = pl.loggers.WandbLogger(project='modsiren', 
                                            log_model='all', 
                                            group=args['run_group'], 
                                            notes=args['run_notes'],
                                            id=args['resume_id'],
                                            resume="must")
        else:
            logger = pl.loggers.WandbLogger(project='modsiren', 
                                            log_model='all', 
                                            group=args['run_group'], 
                                            notes=args['run_notes'])
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

    # Find the latest checkpoint if resuming
    resume_checkpoint_path = None
    if args['resume_id']:
        # Get the run to resume from
        # resume_run = api.run(f'modsiren/{args["resume_id"]}')
        
        latest_checkpoint = api.artifact(f'modsiren/model-{args["resume_id"]}:latest').download()
        resume_checkpoint_path = latest_checkpoint + '/model.ckpt'

        # Handle pretrained latents case
        if args['pretrained_latents']:
            upstream_run = api.run(f'{args["project"]}/{args["run_id"]}')
            upstream_config = upstream_run.config
            args['z_dim'] = upstream_config['hidden_dim']
            model = ModSiren.load_from_checkpoint(resume_checkpoint_path)
        else:
            model = ModSirenFirstOrderMAML.load_from_checkpoint(resume_checkpoint_path)
            
        # Example of trainer setup with resume capability
        trainer = pl.Trainer(
            max_epochs=args['max_epochs'],
            accelerator='gpu',
            devices=-1,
            logger=logger,
            inference_mode=False,
            callbacks=[checkpoint_callback],
            gradient_clip_val=args['clip_grad_norm'],  # Apply gradient clipping
            # limit_val_batches=0
        )

        # Example with dummy data (not actually run)
        # train_loader, val_loader = get_data_loaders()
        trainer.fit(model, data_module, ckpt_path=resume_checkpoint_path)
    else:
        # Example of trainer setup with resume capability
        trainer = pl.Trainer(
            max_epochs=args['max_epochs'],
            accelerator='gpu',
            devices=-1,
            logger=logger,
            inference_mode=not args['pretrained_latents'],
            callbacks=[checkpoint_callback],
        )

        # Example with dummy data (not actually run)
        # train_loader, val_loader = get_data_loaders()
        trainer.fit(model, data_module)