import logging
import os
from omegaconf import DictConfig
import argparse

import torch
from pathlib import Path
import time as t

from trainers.utils.data_loader import get_data_loader

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

    results_folder = Path("GAN") / cfg.optimizers.name / f"{cfg.datasets.name}" / t.strftime("%Y%m%d-%H%M%S")
    results_folder.mkdir(parents=True, exist_ok=True)

    if cfg.models.name == "wgan":
        from trainers.wgan_trainer import WGAN_GP_Trainer
        from models.wgan import Generator, Discriminator

        generator = Generator(
            channels=cfg.models.generator.channels, in_dim=cfg.models.generator.z_dim
        ).to(device)
        discriminator = Discriminator(channels=cfg.models.discriminator.channels).to(
            device
        )

        if cfg.models.use_checkpoint:
            if os.path.isfile(cfg.models.generator.checkpoint_path) and os.path.isfile(
                cfg.models.discriminator.checkpoint_path
            ):
                generator.load_state_dict(
                    torch.load(cfg.models.generator.checkpoint_path, map_location=device)
                )
                logger.info(f"Loaded generator from checkpoints {cfg.models.generator.checkpoint_path}.")
                discriminator.load_state_dict(
                    torch.load(
                        cfg.models.discriminator.checkpoint_path, map_location=device
                    )
                )
                logger.info(f"Loaded discriminator from checkpoints {cfg.models.discriminator.checkpoint_path}.")
            else:
                raise FileNotFoundError(
                    "Checkpoint files not found for generator or discriminator."
                )

        # if cfg.optimizers.name == "adam":
        #     d_optimizer = torch.optim.Adam(
        #         discriminator.parameters(),
        #         lr=cfg.optimizers.lr,
        #         betas=(cfg.optimizers.b1, cfg.optimizers.b2),
        #     )
        #     g_optimizer = torch.optim.Adam(
        #         generator.parameters(),
        #         lr=cfg.optimizers.lr,
        #         betas=(cfg.optimizers.b1, cfg.optimizers.b2),
        #     )
        # elif cfg.optimizers.name == "tiada-adam":
        #     from optimizers.TiAda import TiAda_Adam

        #     d_optimizer = TiAda_Adam(
        #         discriminator.parameters(),
        #         lr=cfg.optimizers.lr,
        #         alpha=cfg.optimizers.beta,
        #         betas=(cfg.optimizers.b1, cfg.optimizers.b2),
        #     )
        #     g_optimizer = TiAda_Adam(
        #         generator.parameters(),
        #         lr=cfg.optimizers.lr,
        #         alpha=cfg.optimizers.beta,
        #         opponent_optim=d_optimizer,
        #         betas=(cfg.optimizers.b1, cfg.optimizers.b2),
        #     )
        if cfg.optimizers.name == "adafm":
            from optimizers.AdaFM import AdaFM

            d_optimizer = AdaFM(
                discriminator.parameters(),
                lr=cfg.optimizers.lr_y,
                beta=cfg.optimizers.beta_for_VRAda,
                results_folder=results_folder,
            )
            g_optimizer = AdaFM(
                generator.parameters(),
                lr=cfg.optimizers.lr_x,
                opponent_optim=d_optimizer,
                beta=cfg.optimizers.beta_for_VRAda,
                results_folder=results_folder,
            )
        elif cfg.optimizers.name == "tiada":
            from optimizers.TiAda import TiAda

            d_optimizer = TiAda(
                discriminator.parameters(),
                beta=cfg.optimizers.beta,
                lr=cfg.optimizers.lr_y,
                results_folder=results_folder,
            )
            g_optimizer = TiAda(
                generator.parameters(),
                beta=cfg.optimizers.beta,
                opponent_optim=d_optimizer,
                lr=cfg.optimizers.lr_x,
                results_folder=results_folder,
            )
        # elif cfg.optimizers.name == "RSGDA":
        #     from optimizers.RSGDA import RSGDA

        #     d_optimizer = RSGDA(
        #         discriminator.parameters(),
        #         beta_y=cfg.optimizers.beta_y,
        #         lr_y=cfg.optimizers.lr_y,
        #     )
        #     g_optimizer = RSGDA(
        #         generator.parameters(),
        #         beta_x=cfg.optimizers.beta_x,
        #         opponent_optim=d_optimizer,
        #         lr_x=cfg.optimizers.lr_x,
        #     )
        # elif cfg.optimizers.name == "VRAdaGDA":
        #     from optimizers.VRAdaGDA import VRAdaGDA

        #     d_optimizer = VRAdaGDA(
        #         discriminator.parameters(),
        #         beta_y=cfg.optimizers.beta_y,
        #         lr_y=cfg.optimizers.lr_y,
        #     )
        #     g_optimizer = VRAdaGDA(
        #         generator.parameters(),
        #         beta_x=cfg.optimizers.beta_x,
        #         opponent_optim=d_optimizer,
        #         lr_x=cfg.optimizers.lr_x,
        #     )
        elif cfg.optimizers.name == "msgda":
            from optimizers.msgda import MSGDA

            d_optimizer = MSGDA(
                discriminator.parameters(),
                lr=cfg.optimizers.lr_discriminator,
                beta=cfg.optimizers.beta_discriminator,
                results_folder=results_folder,
            )
            g_optimizer = MSGDA(
                generator.parameters(),
                lr=cfg.optimizers.lr_generator,
                opponent_optim=d_optimizer,
                beta=cfg.optimizers.beta_generator,
                results_folder=results_folder,
            )
        elif cfg.optimizers.name == "pesg":
            from optimizers.pesg import PESG

            d_optimizer = PESG(
                discriminator.parameters(),
                total_iter=cfg.models.generator_iters * cfg.models.critic_iters,
                lr=cfg.optimizers.lr,
                clip_value=cfg.optimizers.clip_value,
                weight_decay=cfg.optimizers.weight_decay,
                epoch_decay=cfg.optimizers.epoch_decay,
                momentum=cfg.optimizers.momentum,
                decay_iters=cfg.optimizers.decay_iters,
                decay_factor=cfg.optimizers.decay_factor,
                results_folder=results_folder,
            )
            g_optimizer = PESG(
                generator.parameters(),
                total_iter=cfg.models.generator_iters,
                lr=cfg.optimizers.lr,
                clip_value=cfg.optimizers.clip_value,
                weight_decay=cfg.optimizers.weight_decay,
                epoch_decay=cfg.optimizers.epoch_decay,
                momentum=cfg.optimizers.momentum,
                decay_iters=cfg.optimizers.decay_iters,
                decay_factor=cfg.optimizers.decay_factor,
                opponent_optim=d_optimizer,
                results_folder=results_folder,
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
            cfg=cfg,
            results_folder=results_folder,
            device=device,
        )
        logger.info("Trainer is ready.")

        trainer.train(train_loader, Real_Inception_score)

        logger.info("Training is finished.")
    else:
        raise NotImplementedError(f"Model {cfg.models.name} is not implemented.")
