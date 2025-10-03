import logging
from omegaconf import DictConfig
import argparse

import torch

from trainers.utils.data_loader import get_data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> None:
    device = cfg.setup.device
    if device == "cuda" and not torch.cuda.is_available():
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

    if cfg.models.name == "wgan":
        from trainers.wgan_trainer import WGAN_GP_Trainer
        from models.wgan import Generator, Discriminator

        generator = Generator(
            channels=cfg.models.generator.channels, in_dim=cfg.models.generator.z_dim
        ).to(device)
        discriminator = Discriminator(channels=cfg.models.discriminator.channels).to(
            device
        )

        if cfg.optimizers.name == "adam":
            d_optimizer = torch.optim.Adam(
                discriminator.parameters(),
                lr=cfg.optimizers.lr,
                betas=(cfg.optimizers.b1, cfg.optimizers.b2),
            )
            g_optimizer = torch.optim.Adam(
                generator.parameters(),
                lr=cfg.optimizers.lr,
                betas=(cfg.optimizers.b1, cfg.optimizers.b2),
            )
        elif cfg.optimizers.name == "tiada-adam":
            from optimizers.TiAda import TiAda_Adam

            d_optimizer = TiAda_Adam(
                discriminator.parameters(),
                lr=cfg.optimizers.lr,
                alpha=cfg.optimizers.beta,
                betas=(cfg.optimizers.b1, cfg.optimizers.b2),
            )
            g_optimizer = TiAda_Adam(
                generator.parameters(),
                lr=cfg.optimizers.lr,
                alpha=cfg.optimizers.beta,
                opponent_optim=d_optimizer,
                betas=(cfg.optimizers.b1, cfg.optimizers.b2),
            )
        elif cfg.optimizers.name == "adafm":
            from optimizers.AdaFM import AdaFM

            d_optimizer = AdaFM(
                discriminator.parameters(),
                lr=cfg.optimizers.lr_y,
                beta=cfg.optimizers.beta_for_VRAda,
            )
            g_optimizer = AdaFM(
                generator.parameters(),
                lr=cfg.optimizers.lr_x,
                opponent_optim=d_optimizer,
                beta=cfg.optimizers.beta_for_VRAda,
            )
        elif cfg.optimizers.name == "tiada":
            from optimizers.TiAda import TiAda

            d_optimizer = TiAda(
                discriminator.parameters(),
                beta=cfg.optimizers.beta,
                lr=cfg.optimizers.lr_y,
            )
            g_optimizer = TiAda(
                generator.parameters(),
                beta=cfg.optimizers.beta,
                opponent_optim=d_optimizer,
                lr=cfg.optimizers.lr_x,
            )
        elif cfg.optimizers.name == "RSGDA":
            from optimizers.RSGDA import RSGDA

            d_optimizer = RSGDA(
                discriminator.parameters(),
                beta_y=cfg.optimizers.beta_y,
                lr_y=cfg.optimizers.lr_y,
            )
            g_optimizer = RSGDA(
                generator.parameters(),
                beta_x=cfg.optimizers.beta_x,
                opponent_optim=d_optimizer,
                lr_x=cfg.optimizers.lr_x,
            )
        elif cfg.optimizers.name == "VRAdaGDA":
            from optimizers.VRAdaGDA import VRAdaGDA

            d_optimizer = VRAdaGDA(
                discriminator.parameters(),
                beta_y=cfg.optimizers.beta_y,
                lr_y=cfg.optimizers.lr_y,
            )
            g_optimizer = VRAdaGDA(
                generator.parameters(),
                beta_x=cfg.optimizers.beta_x,
                opponent_optim=d_optimizer,
                lr_x=cfg.optimizers.lr_x,
            )
        else:
            raise NotImplementedError(
                f"Optimizer {cfg.optimizers.name} is not implemented."
            )

        trainer = WGAN_GP_Trainer(
            generator=generator,
            discriminator=discriminator,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            generator_iters=cfg.models.generator_iters,
            critic_iters=cfg.models.critic_iters,
            save_interval=cfg.models.training.save_interval,
            z_dim=cfg.models.generator.z_dim,
            batch_size=cfg.models.training.batch_size,
            device=device,
        )
        logger.info("Trainer is ready.")

        trainer.train(train_loader, Real_Inception_score)

        logger.info("Training is finished.")
    else:
        raise NotImplementedError(f"Model {cfg.models.name} is not implemented.")
