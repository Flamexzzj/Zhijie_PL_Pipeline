import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from st_water_seg.models import build_model
from st_water_seg.datasets import build_dataset
from st_water_seg.datasets.utils import generate_image_slice_object
from st_water_seg.utils.utils_misc import generate_innovation_script


# @hydra.main(version_base="1.1", config_path="conf", config_name="config")
def fit_model(cfg: DictConfig, overwrite_exp_dir: str = None) -> str:

    resume_training = True
    # Get experiment directory.
    if overwrite_exp_dir is None:
        exp_dir = os.getcwd()
    else:
        exp_dir = overwrite_exp_dir

    # Load dataset.
    slice_params = generate_image_slice_object(cfg.crop_height, cfg.crop_width,
                                               cfg.crop_stride)

    if cfg.dataset.dataset_kwargs is None:
        cfg.dataset.dataset_kwargs = {}

    train_dataset = build_dataset(cfg.dataset.name,
                                  'train',
                                  slice_params,
                                  sensor=cfg.dataset.sensor,
                                  channels=cfg.dataset.channels,
                                  norm_mode=cfg.norm_mode,
                                  eval_region=cfg.eval_region,
                                  ignore_index=cfg.ignore_index,
                                  seed_num=cfg.seed_num,
                                  train_split_pct=cfg.train_split_pct,
                                  transforms=cfg.transforms,
                                  **cfg.dataset.dataset_kwargs)
    valid_dataset = build_dataset(cfg.dataset.name,
                                  'valid',
                                  slice_params,
                                  sensor=cfg.dataset.sensor,
                                  channels=cfg.dataset.channels,
                                  norm_mode=cfg.norm_mode,
                                  eval_region=cfg.eval_region,
                                  ignore_index=cfg.ignore_index,
                                  seed_num=cfg.seed_num,
                                  train_split_pct=cfg.train_split_pct,
                                  **cfg.dataset.dataset_kwargs)

    # Create dataloaders.
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.n_workers)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.n_workers)

    # Create model.
    model = build_model(cfg.model.name,
                        train_dataset.n_channels,
                        train_dataset.n_classes,
                        cfg.lr,
                        log_image_iter=cfg.log_image_iter,
                        to_rgb_fcn=train_dataset.to_RGB,
                        ignore_index=train_dataset.ignore_index,
                        **cfg.model.model_kwargs)

    # Create logger.
    logger = pl.loggers.TensorBoardLogger(
        save_dir=os.path.join(exp_dir, 'tensorboard_logs'))

    # Train model.
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(exp_dir, 'checkpoints'),
        save_top_k=cfg.save_topk_models,
        mode='max',
        monitor="val_MulticlassJaccardIndex",
        filename="model-{epoch:02d}-{val_MulticlassJaccardIndex:.4f}")
    trainer = pl.Trainer(max_epochs=cfg.n_epochs,
                         accelerator="gpu",
                         devices=1,
                         default_root_dir=exp_dir,
                         callbacks=[checkpoint_callback],
                         logger=logger,
                         profiler=cfg.profiler,
                         limit_train_batches=cfg.limit_train_batches,
                         limit_val_batches=cfg.limit_val_batches)
    if resume_training:
        # automatically restores model, epoch, step, LR schedulers, etc...
        trainer.fit(model=model,
                    train_dataloaders=train_loader,
                    val_dataloaders=valid_loader,
                    ckpt_path="/home/zhijiezhang/spatial_temporal_water_seg/outputs/Nepal_PS_low+no/checkpoints/model-epoch=08-val_MulticlassJaccardIndex=0.9484.ckpt")
    # Training after loading checkpoints
    
    else:
        trainer.fit(model=model,
                    train_dataloaders=train_loader,
                    val_dataloaders=valid_loader)
    
    # Final evaluation of model on validation set.
    # trainer.test(model, dataloaders=valid_loader, verbose=True)

    # Return best model path.
    return trainer.checkpoint_callback.best_model_path


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def config_fit_func(cfg: DictConfig):
    fit_model(cfg)


if __name__ == '__main__':
    config_fit_func()
