import torch
import pytorch_lightning as pl

class GPUMemoryCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self, trainer, pl_module):
        import gc
        gc.collect()
        torch.cuda.empty_cache()