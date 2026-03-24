import argparse
import torch
import random
import numpy as np
import pytorch_lightning as pl
from utils.config import load_config, instantiate_from_config
import time


def main(args):

    config = load_config(args.config)
    all_classes = set(range(config['model']['params'].get('n_classes', 10)))
    print('all_class:', all_classes)
    if args.query_digits:
        if isinstance(args.query_digits, (list, tuple)):
            unlearn_classes = set(args.query_digits)
        else:
            unlearn_classes = {args.query_digits}
    else:
        unlearn_classes = set()
    rest_classes = list(all_classes - unlearn_classes)
    
    # Modify data config to use only rest classes
    config['data']['params']['query_digits'] = rest_classes
    config['data']['params']['support_digits'] = rest_classes


    model = instantiate_from_config(config['model'])
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    datamodule = instantiate_from_config(config['data'])
    datamodule.set_no_train_digits(list(unlearn_classes))
    lightning_config = config.get('lightning', {})
    config['lightning']['callbacks']['checkpoint_callback']['params']['dirpath'] += f"_unl{''.join(map(str, sorted(unlearn_classes)))}"
    logger = instantiate_from_config(lightning_config['logger']) if 'logger' in lightning_config else None
    callbacks = []
    if 'callbacks' in lightning_config and lightning_config['callbacks'] is not None:
        for cb in lightning_config['callbacks'].values():
            if cb is not None:
                callbacks.append(instantiate_from_config(cb))
    
    from pytorch_lightning.loggers import WandbLogger
    if logger is not None and isinstance(logger, WandbLogger):
        logger.log_hyperparams(config)
    
    trainer_config = lightning_config.get('trainer', {})
    trainer_config['logger'] = logger
    trainer_config['callbacks'] = callbacks
    trainer = pl.Trainer(**trainer_config)
    # Set bar refresh rate
    for cb in trainer.callbacks:
        if isinstance(cb, pl.callbacks.progress.ProgressBar):
            old_rate = cb.refresh_rate
            cb._refresh_rate = 100  # or any desired value
            print(f"Set progress bar refresh rate from {old_rate} to {cb.refresh_rate}")
      

    t0 = time.perf_counter()
    trainer.fit(model, datamodule=datamodule)
    t1 = time.perf_counter()
    print(f"Training completed in {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--query_digits", type=int, nargs='+', default=None, help="Digits to be unlearned")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    main(args)