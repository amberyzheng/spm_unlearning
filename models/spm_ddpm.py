import torch
import torch.nn as nn
import pytorch_lightning as pl
import yaml
import wandb
from torchvision.utils import make_grid
from ema_pytorch import EMA
from nets.unet_spm import UNet
from nets.ddpm_cond import DDPM
import os
from torchmetrics.image.fid import FrechetInceptionDistance


class Config(object):
    def __init__(self, dic):
        for key in dic:
            setattr(self, key, dic[key])

class SemiParametricDDPM(pl.LightningModule):
    def __init__(self, config, n_classes=None):
        super(SemiParametricDDPM, self).__init__()
        
        self.config = Config(config)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device_str = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        
        if hasattr(self.config, 'image_shape'):
            self.sample_shape = self.config.image_shape
        else:
            self.sample_shape = None
        
        # Set n_classes and register a buffer for validation condition tensor
        if n_classes is None:
            self.n_classes = self.config.network.get('n_classes', 0)
        else:
            self.n_classes = n_classes
        if self.n_classes:
            self.register_buffer('val_cond', torch.arange(self.n_classes, device=torch.device(self.device_str)))
        
        self.diffusion = DDPM(
            nn_model=UNet(**self.config.network, n_classes=self.n_classes),
            **self.config.diffusion,
            device=self.device_str,
        )
        if self.config.ema is not None:
            self.ema = EMA(self.diffusion, beta=self.config.ema, update_after_step=0, update_every=1)
        else:
            self.ema = None
        self.save_hyperparameters()
        self.fid_metric = FrechetInceptionDistance(normalize=True).to(device=self.device_str)

    def forward(self, x, support, c, c_support, use_amp=False):
        loss = self.diffusion(x, support, c, c_support, use_amp=use_amp)
        return loss

    def training_step(self, batch, batch_idx):
        x, support, c, c_support = batch
        with torch.amp.autocast('cuda', enabled=getattr(self.config, 'use_amp', False)):
            loss = self(x, support, c, c_support, use_amp=getattr(self.config, 'use_amp', False))
        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=False)
        if self.ema is not None:
            self.ema.update()
        return loss
    

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        x, support, c, c_support = batch
        loss = self(x, support, c, c_support, use_amp=getattr(self.config, 'use_amp', False))
        self.log("val_loss", loss)
        
        if batch_idx == 0 and self.ema is not None:
            n_per_class = self.config.n_sample_per_class
            sample_shape = self.sample_shape if self.sample_shape is not None else x.shape[1:]
            
            steps = self.config.steps if hasattr(self.config, 'steps') else 50
            eta = self.config.eta if hasattr(self.config, 'eta') else 0.0
            cond = self.val_cond.repeat(n_per_class)

            with torch.no_grad():
                samples = self.ema.ema_model.ddim_sample(
                    self.n_classes * n_per_class,
                    sample_shape,
                    guide_w=self.config.guide_w,
                    steps=steps,
                    eta=eta,
                    cond=cond,
                    support=support,
                    cond_support=c_support
                )
                grid = make_grid(samples, nrow=n_per_class)
                self.logger.experiment.log({
                    'validation_samples': wandb.Image(grid, caption='Validation conditional samples')
                })
        
            # Use clone() to avoid keeping references to the original tensors
            real_samples = x[:samples.shape[0]].clone()
            self.fid_metric.update(real_samples, real=True)
            self.fid_metric.update(samples.clone(), real=False)
            
            del real_samples, samples, grid
            torch.cuda.empty_cache()

        return loss

    def configure_optimizers(self):
        optim_cfg = self.config
        params = self.parameters()
        beta2 = getattr(optim_cfg, "beta2", 0.999)
        optimizer = torch.optim.Adam(
            params,
            lr=optim_cfg.lrate,
            betas=(optim_cfg.beta1, beta2),
            weight_decay=optim_cfg.weight_decay,
            amsgrad=optim_cfg.amsgrad,
            eps=optim_cfg.eps,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min((epoch + 1.0) / getattr(self.config, "warm_epoch", 1), 1.0)
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        clip_val = self.config.grad_clip
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip_val)

    def on_validation_epoch_end(self):
        fid = self.fid_metric.compute()
        self.log("val_fid", fid)
        self.fid_metric = FrechetInceptionDistance(normalize=True).to(device=self.device_str)
        del fid
        import gc
        gc.collect()
        torch.cuda.empty_cache()