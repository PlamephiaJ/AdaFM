import logging
import os
from omegaconf import DictConfig, OmegaConf
import argparse

import torch
from pathlib import Path
import time as t

from trainers.utils.data_loader import get_data_loader

from optimizer_factory import create_optimizers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> None:
    device = cfg.setup.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is not available but it is the selected device.")
    logger.info(f"Using device: {device}")

    # Load datasets to train and test loaders
    args = argparse.Namespace(
        dataroot=cfg.datasets.dataroot,
        dataset=cfg.datasets.name,
        download=cfg.datasets.download,
        batch_size=cfg.models.training.batch_size,
    )
    train_loader, _ = get_data_loader(
        args
    )  # We're not using test_loader in this function
    Real_Inception_score = []
    logger.info("Data loaders are ready.")

    if cfg.datasets.name == "cifar10":
        results_folder = (
            Path("GAN")
            / "model_factory"
            / cfg.models.backbone.name
            / cfg.optimizers.name
            / t.strftime("%Y%m%d-%H%M%S")
        )
    if cfg.datasets.name == "cifar100":
        results_folder = (
            Path("GAN_CIFAR100")
            / "model_factory"
            / cfg.models.backbone.name
            / cfg.optimizers.name
            / t.strftime("%Y%m%d-%H%M%S")
        )
    results_folder = Path("delta_test") / results_folder
    results_folder.mkdir(parents=True, exist_ok=True)

    # Save configuration snapshot
    config_file = results_folder / "config_snapshot.yaml"
    with open(config_file, "w") as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Configuration snapshot saved to {config_file}")

    if cfg.models.name == "wgan":
        from trainers.wgan_trainer import WGAN_GP_Trainer
        from models.wgan_factory import create_model

        generator = create_model(
            cfg.models.backbone.name,
            cfg.models.backbone.generator
        ).to(device)
        discriminator = create_model(
            cfg.models.backbone.name,
            cfg.models.backbone.discriminator
        ).to(device)

        if cfg.models.use_checkpoint:
            if os.path.isfile(cfg.models.generator.checkpoint_path) and os.path.isfile(
                cfg.models.discriminator.checkpoint_path
            ):
                generator.load_state_dict(
                    torch.load(
                        cfg.models.generator.checkpoint_path, map_location=device
                    )
                )
                logger.info(
                    f"Loaded generator from checkpoints {cfg.models.generator.checkpoint_path}."
                )
                discriminator.load_state_dict(
                    torch.load(
                        cfg.models.discriminator.checkpoint_path, map_location=device
                    )
                )
                logger.info(
                    f"Loaded discriminator from checkpoints {cfg.models.discriminator.checkpoint_path}."
                )
            else:
                raise FileNotFoundError(
                    "Checkpoint files not found for generator or discriminator."
                )
        
        
        # 使用优化器工厂创建优化器
        g_optimizer, d_optimizer = create_optimizers(
            generator=generator,
            discriminator=discriminator,
            cfg=cfg,
            results_folder=results_folder
        )

        trainer = WGAN_GP_Trainer(
            generator=generator,
            discriminator=discriminator,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            generator_iters=cfg.models.generator_iters,
            critic_iters=cfg.optimizers.critic_iters,
            save_interval=cfg.models.training.save_interval,
            z_dim=cfg.models.backbone.generator.in_dim,
            batch_size=cfg.models.training.batch_size,
            cfg=cfg,
            results_folder=results_folder,
            device=device,
        )
        logger.info("Trainer is ready.")

        trainer.train(train_loader, Real_Inception_score)

        logger.info("Training is finished.")
    else:
        raise NotImplementedError(f"Model {cfg.models.name} is not implemented.")
