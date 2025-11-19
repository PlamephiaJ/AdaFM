import copy
import logging
import os
import time as t
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from omegaconf import DictConfig
from torch.autograd import Variable
from torchvision.models.inception import Inception_V3_Weights, inception_v3
from torchvision.utils import make_grid
from tqdm import tqdm

from .utils.frechet_inception_distance import FIDCalculator
from .utils.inception_score import get_inception_score

LOGGER = logging.getLogger(__name__)


class WGAN_GP_Trainer:

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        g_optimizer: optim.Optimizer,
        d_optimizer: optim.Optimizer,
        generator_iters: int,
        critic_iters: int,
        eval_interval: int,
        z_dim: int,
        batch_size: int,
        cfg: DictConfig,
        train_loader: torch.utils.data.DataLoader,
        results_folder: Path,
        device=None,
        tb_writer=None,
    ):
        if device is None:
            raise ValueError("Device must be specified for the trainer.")
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = device
        self.number_of_images = 100
        self.generator_iters = generator_iters
        self.critic_iters = critic_iters
        self.lambda_term = cfg.models.lambda_term
        LOGGER.info(f"Using gradient penalty lambda term: {self.lambda_term}")
        self.eval_interval = eval_interval
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.cfg = cfg
        self.results_folder = results_folder
        self.images_folder = results_folder / "images"
        os.makedirs(self.images_folder, exist_ok=True)
        self.writer = tb_writer
        if cfg.worker.gpu_memory_gb <= 8:
            self.inception_model = None
        else:
            self.inception_model = inception_v3(
                weights=Inception_V3_Weights.DEFAULT
            ).to(device)
            self.inception_model.eval()

        if cfg.models.evaluation.use_fid:
            self.fid_calculator = FIDCalculator(device, train_loader)
        else:
            self.fid_calculator = None

    def calculate_gradient_penalty(self, real_images, fake_images, eta):
        # eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        if eta is None:
            eta = (
                torch.FloatTensor(real_images.size(0), 1, 1, 1)
                .uniform_(0, 1)
                .to(self.device)
            )
            eta = eta.expand(
                real_images.size(0),
                real_images.size(1),
                real_images.size(2),
                real_images.size(3),
            )
        else:
            eta = eta.to(self.device)

        interpolated = eta * real_images + ((1 - eta) * fake_images)
        interpolated = interpolated.to(self.device)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated, device=self.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        B = gradients.size(0)
        gradients = gradients.view(B, -1)
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term

        return grad_penalty, eta

    def _save_models_checkpoint(self, iteration):
        checkpoint_dir = self.results_folder / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        g_checkpoint_path = checkpoint_dir / f"generator_iter_{iteration}.pth"
        d_checkpoint_path = checkpoint_dir / f"discriminator_iter_{iteration}.pth"

        # Remove old checkpoints for previous iterations
        for old_checkpoint in checkpoint_dir.glob("generator_iter_*.pth"):
            if old_checkpoint != g_checkpoint_path:
                old_checkpoint.unlink()
                LOGGER.info(f"Removed old generator checkpoint: {old_checkpoint}")

        for old_checkpoint in checkpoint_dir.glob("discriminator_iter_*.pth"):
            if old_checkpoint != d_checkpoint_path:
                old_checkpoint.unlink()
                LOGGER.info(f"Removed old discriminator checkpoint: {old_checkpoint}")

        torch.save(self.G.state_dict(), g_checkpoint_path)
        torch.save(self.D.state_dict(), d_checkpoint_path)

        LOGGER.info(f"Saved generator checkpoint to {g_checkpoint_path}")
        LOGGER.info(f"Saved discriminator checkpoint to {d_checkpoint_path}")

    def train(self, train_loader):
        try:
            self.t_begin = t.time()
            self.data = self.get_infinite_batches(train_loader)
            one = torch.tensor(1, dtype=torch.float).to(self.device)
            minus_one = one * -1

            real_images, _ = next(iter(train_loader))
            real_images = real_images.to(self.device)

            os.makedirs(self.results_folder, exist_ok=True)
            vutils.save_image(
                real_images, self.results_folder / "real_images.png", normalize=True
            )

            total_iter = 0
            if self.cfg.optimizers.use_previous_model:
                D_old, G_old = None, None

            best_real_inception_score = -float("inf")
            inception_scores = []
            fid_scores = []
            time_record = []

            for g_iter in tqdm(
                range(self.generator_iters),
                desc=f"Training: optimizer {self.cfg.optimizers.name}",
            ):
                improvement = False
                # Requires grad, Generator requires_grad = False
                for p in self.D.parameters():
                    p.requires_grad = True

                if (
                    self.cfg.optimizers.use_previous_model
                    and D_old is not None
                ):
                    for p in D_old.parameters():
                        p.requires_grad = True

                d_loss_real = 0
                d_loss_fake = 0
                # Wasserstein_D = 0

                for d_iter in range(self.cfg.optimizers.critic_iters):
                    self.D.zero_grad()
                    if (
                        self.cfg.optimizers.use_previous_model
                        and D_old is not None
                    ):
                        D_old.zero_grad()

                    images = self.data.__next__()
                    images = images.to(self.device)

                    # Train discriminator
                    # WGAN - Training discriminator more iterations than generator
                    # Train with real images
                    d_loss_real = self.D(images)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(minus_one)

                    # Train with fake images
                    z = torch.randn(
                        images.size(0), self.z_dim, 1, 1, device=self.device
                    )

                    fake_images = self.G(z)
                    d_loss_fake = self.D(fake_images)
                    d_loss_fake = d_loss_fake.mean()
                    d_loss_fake.backward(one)

                    # Train with gradient penalty
                    gradient_penalty, eta = self.calculate_gradient_penalty(
                        images.detach(), fake_images.detach(), eta=None
                    )
                    gradient_penalty.backward()

                    d_loss = d_loss_fake - d_loss_real + gradient_penalty
                    self.writer.add_scalar(
                        "Discriminator Loss", d_loss.item(), total_iter
                    )
                    # Wasserstein_D = (d_loss_real - d_loss_fake).item()

                    if self.cfg.optimizers.use_previous_model:
                        if D_old is not None:
                            d_loss_real_old = D_old(images).mean()
                            d_loss_real_old.backward(minus_one)

                            fake_images_ = G_old(z)
                            d_loss_fake_old = D_old(fake_images_).mean()
                            d_loss_fake_old.backward(one)

                            # Train with gradient penalty
                            gradient_penalty_old, _ = self.calculate_gradient_penalty(
                                images.detach(), fake_images_.detach(), eta=eta
                            )
                            gradient_penalty_old.backward()
                            delta_y = [g.grad.data.clone() for g in D_old.parameters()]
                            d_loss_real_old = d_loss_fake_old = gradient_penalty_old = (
                                None
                            )
                        else:
                            delta_y = None

                        D_old = copy.deepcopy(self.D).to(self.device)
                        self.d_optimizer.step(delta=delta_y)

                        if delta_y is not None:
                            delta_y.clear()
                    else:
                        self.d_optimizer.step()

                    total_iter += 1

                # Generator update
                for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation
                if (
                    self.cfg.optimizers.use_previous_model
                    and D_old is not None
                ):
                    for p in D_old.parameters():
                        p.requires_grad = False

                self.G.zero_grad()
                if (
                    self.cfg.optimizers.use_previous_model
                    and G_old is not None
                ):
                    G_old.zero_grad()
                # train generator
                # compute loss with fake images
                z = torch.randn(self.batch_size, self.z_dim, 1, 1, device=self.device)
                fake_images = self.G(z)
                g_loss = self.D(fake_images)
                g_loss = g_loss.mean()
                g_loss.backward(minus_one)
                self.writer.add_scalar("Generator Loss", g_loss.item(), total_iter)
                # g_cost = -g_loss
                if self.cfg.optimizers.use_previous_model:
                    if G_old is not None:
                        fake_images_ = G_old(z)
                        g_loss_old = D_old(fake_images_).mean()
                        g_loss_old.backward(minus_one)
                        delta_x = [g.grad.data.clone() for g in G_old.parameters()]
                        g_loss_old = fake_images_ = None
                    else:
                        delta_x = None
                    # TODO: deepcopy can be optimized
                    G_old = copy.deepcopy(self.G).to(self.device)
                    self.g_optimizer.step(delta=delta_x)
                    if delta_x is not None:
                        delta_x.clear()
                else:
                    self.g_optimizer.step()

                total_iter += 1
                # Saving model and sampling images every 1000th generator iterations
                if (total_iter) % self.eval_interval == 0:
                    with torch.no_grad():
                        z = torch.randn(
                            self.cfg.models.evaluation.number_of_generated_images_for_inception_score_calculation,
                            self.z_dim,
                            1,
                            1,
                            device=self.device,
                        )
                        samples = self.G(z)

                    # for _ in range(10):
                    #     z = Variable(torch.randn(800, self.z_dim, 1, 1)).to(self.device)
                    #     samples = self.G(z)
                    #     sample_list.append(samples.data.cpu().numpy())

                    # # Flattening list of list into one list
                    # new_sample_list = list(chain.from_iterable(sample_list))
                    LOGGER.info(
                        f"Calculating Inception Score or FID Score over {self.cfg.models.evaluation.number_of_generated_images_for_inception_score_calculation} generated images"
                    )

                    if self.cfg.models.evaluation.use_is:
                        is_mean, is_std = get_inception_score(
                            samples,
                            inception_model=self.inception_model,
                            batch_size=64,
                            resize=True,
                            splits=10,
                            device=self.device,
                        )
                    else:
                        is_mean, is_std = float("nan"), float("nan")
                    inception_scores.append(is_mean)
                    if self.cfg.models.evaluation.use_fid:
                        fid_score = self.fid_calculator.calculate(samples)
                    else:
                        fid_score = float("nan")
                    fid_scores.append(fid_score)

                    if is_mean > best_real_inception_score:
                        best_real_inception_score = is_mean
                        self._save_models_checkpoint(total_iter)
                        LOGGER.info(
                            f"New best Inception Score: {best_real_inception_score:.4f}. Checkpoints saved."
                        )
                        improvement = True

                    # Testing
                    elapsed_time = t.time() - self.t_begin
                    time_record.append(elapsed_time)

                    self.writer.add_scalar("Inception Score", is_mean, total_iter)
                    self.writer.add_scalar("FID Score", fid_score, total_iter)

                    LOGGER.info(
                        f"Inception score (mean, std): ({is_mean:.4f}, {is_std:.4f})"
                    )
                    LOGGER.info(f"FID score: {fid_score:.4f}")
                    LOGGER.info(f"Generator iter: {g_iter}")
                    LOGGER.info(f"total_iter_finished: {total_iter}")
                    LOGGER.info(
                        f"Time elapsed: {str(timedelta(seconds=int(elapsed_time)))}"
                    )

                    # Save generated images for visualization
                    if self.cfg.models.evaluation.save_img and improvement:
                        with torch.no_grad():
                            z = torch.randn(
                                self.batch_size, self.z_dim, 1, 1, device=self.device
                            )
                            fake_images = self.G(z).detach().cpu()
                        save_image_path = self.images_folder / f"iter_{total_iter}.png"
                        vutils.save_image(fake_images, save_image_path, normalize=True)

                        # Log to TensorBoard
                        grid = make_grid(
                            fake_images, nrow=8, normalize=True, value_range=(-1, 1)
                        )
                        self.writer.add_image("Generated Images", grid, total_iter)
                        LOGGER.info(
                            f"Saved images at iteration {total_iter} with improvement."
                        )

            self.t_end = t.time()
            LOGGER.info(f"Time of training-{self.t_end - self.t_begin}")
            np_inception_scores = np.array(inception_scores)
            np_fid_scores = np.array(fid_scores)

        except KeyboardInterrupt:
            LOGGER.warning("Training interrupted. Saving Real Inception Scores...")
            np_inception_scores = np.array(inception_scores)
            np_fid_scores = np.array(fid_scores)
        finally:
            if np_inception_scores is not None and np_fid_scores is not None:
                # Also save as text file for easy reading
                csv_save_path = self.results_folder / "training_log.csv"
                with open(csv_save_path, "w") as f:
                    f.write("Step,IS,FID,Time\n")
                    for i, (inception_score, fid_score, elapsed) in enumerate(
                        zip(np_inception_scores, np_fid_scores, time_record)
                    ):
                        f.write(
                            f"{(i+1)*self.eval_interval},{inception_score:.6f},{fid_score:.6f},{elapsed:.6f}\n"
                        )

                best_metrics_save_path = self.results_folder / "best_metrics.csv"
                with open(best_metrics_save_path, "w") as f:
                    f.write("BestIS,AvgIS,BestFID,AvgFID\n")
                    f.write(
                        f"{np_inception_scores.max()},{np_inception_scores.mean()},{np_fid_scores.min()},{np_fid_scores.mean()}\n"
                    )

                LOGGER.info(
                    f"Inception Scores saved to {csv_save_path} and {best_metrics_save_path}"
                )
            else:
                LOGGER.warning("No scores to save.")

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images
