from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import time as t
from datetime import timedelta
from itertools import chain
import copy
from .utils.inception_score import get_inception_score
import torchvision.utils as vutils
import os
import pickle
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

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
        save_interval: int,
        z_dim: int,
        batch_size: int,
        cfg: DictConfig,
        results_folder: Path,
        device=None,
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
        LOGGER.info(f"Using gradient penalty lambda term: {self.lambda_term}")
        LOGGER.info(f"Using gradient penalty lambda term: {self.lambda_term}")
        LOGGER.info(f"Using gradient penalty lambda term: {self.lambda_term}")
        self.save_interval = save_interval
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.cfg = cfg
        self.results_folder = results_folder
        self.images_folder = results_folder / "images"
        os.makedirs(self.images_folder, exist_ok=True)
        tb_log_dir = Path("GP_EXP") / Path(cfg.tensorboard.log_dir_root) / str(results_folder).replace("/", "_")
        self.writer = SummaryWriter(log_dir=tb_log_dir)

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
    
    def train(self, train_loader, Real_Inception_score):
        use_delta = self.cfg.optimizers.get("use_delta", False)
        if use_delta:
            self.train_use_delta(train_loader, Real_Inception_score)
        else:
            self.train_dont_use_delta(train_loader, Real_Inception_score)

    def train_dont_use_delta(self, train_loader, Real_Inception_score):
        try:
            self.t_begin = t.time()
            self.data = self.get_infinite_batches(train_loader)
            one = torch.tensor(1, dtype=torch.float).to(self.device)
            mone = one * -1

            real_images, _ = next(iter(train_loader))
            real_images = real_images.to(self.device)

            os.makedirs(self.results_folder, exist_ok=True)
            vutils.save_image(
                real_images, self.results_folder / "real_images.png", normalize=True
            )

            total_iter = 0

            best_real_inception_score = -float("inf")

            for g_iter in tqdm(
                range(self.generator_iters),
                desc=f"Training: optimizer {self.cfg.optimizers.name}",
            ):
                # Requires grad, Generator requires_grad = False
                for p in self.D.parameters():
                    p.requires_grad = True

                d_loss_real = 0
                d_loss_fake = 0
                Wasserstein_D = 0

                for d_iter in range(self.critic_iters):
                    self.D.zero_grad()

                    images = self.data.__next__()
                    images = self.get_torch_variable(images)

                    # Train discriminator
                    # WGAN - Training discriminator more iterations than generator
                    # Train with real images
                    d_loss_real = self.D(images)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(mone)

                    # Train with fake images
                    z = self.get_torch_variable(
                        torch.randn(images.size(0), self.z_dim, 1, 1)
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
                    self.writer.add_scalar('Discriminator Loss', d_loss.item(), total_iter)
                    Wasserstein_D = (d_loss_real - d_loss_fake).item()

                    self.d_optimizer.step()
                    total_iter += 1

                # Generator update
                for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation

                self.G.zero_grad()
                # train generator
                # compute loss with fake images
                z = self.get_torch_variable(
                    torch.randn(self.batch_size, self.z_dim, 1, 1)
                )
                fake_images = self.G(z)
                g_loss = self.D(fake_images)
                g_loss = g_loss.mean()
                g_loss.backward(mone)
                self.writer.add_scalar('Generator Loss', g_loss.item(), total_iter)
                g_cost = -g_loss
                self.g_optimizer.step()

                # LOGGER.info(f'Generator iteration: {g_iter}/{self.generator_iters}, '
                #       f'loss_real: {d_loss_real:.4f}, '
                #       f'loss_fake: {d_loss_fake:.4f}, '
                #       f'g_loss: {g_loss:.4f}, '
                #       f'lr_x ={self.lr_x},'
                #       f'lr_y={self.lr_y}, '
                #       f'beta = {self.beta_for_VRAda},'
                #       f'dataset={args.dataset}')

                total_iter += 1
                # Saving model and sampling images every 1000th generator iterations
                if (total_iter) % self.save_interval == 0:
                    grad_g = WGAN_GP_Trainer.get_gradient_norm(self.G).item()
                    grad_d = WGAN_GP_Trainer.get_gradient_norm(self.D).item()
                    # self.save_model()
                    # Workaround because graphic card memory can't store more than 830 examples in memory for generating image
                    # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
                    # This way Inception score is more correct since there are different generated examples from every class of Inception model
                    sample_list = []
                    for _ in range(10):
                        # samples  = self.data.__next__()
                        z = Variable(torch.randn(800, self.z_dim, 1, 1)).to(self.device)
                        samples = self.G(z)
                        # samples = samples.mul(0.5).add(0.5)
                        sample_list.append(samples.data.cpu().numpy())

                    # # Flattening list of list into one list
                    new_sample_list = list(chain.from_iterable(sample_list))
                    LOGGER.info("Calculating Inception Score over 8k generated images")
                    # # Feeding list of numpy arrays
                    # inception_score is a tuple (mean, std)
                    # mean IS and std IS
                    inception_score = get_inception_score(
                        new_sample_list,
                        cuda=True,
                        batch_size=64,
                        resize=True,
                        splits=10,
                    )

                    z = self.get_torch_variable(
                        torch.randn(self.number_of_images, self.z_dim, 1, 1)
                    )
                    Real_Inception_score.append(inception_score[0])

                    if inception_score[0] > best_real_inception_score:
                        best_real_inception_score = inception_score[0]
                        self._save_models_checkpoint(total_iter)
                        LOGGER.info(
                            f"New best Inception Score: {best_real_inception_score:.4f}. Checkpoints saved."
                        )

                    # Testing
                    elapsed_time = t.time() - self.t_begin
                    LOGGER.info(
                        "Real Inception score (mean, std): {}".format(inception_score)
                    )
                    LOGGER.info("Generator iter: {}".format(g_iter))
                    LOGGER.info("total_iter_finished: {}".format(total_iter))
                    LOGGER.info(
                        "Time elapsed: {}".format(
                            str(timedelta(seconds=int(elapsed_time)))
                        )
                    )

                    z = self.get_torch_variable(
                        torch.randn(self.batch_size, self.z_dim, 1, 1)
                    )
                    with torch.no_grad():
                        fake_images = self.G(z).detach().cpu()

                    # 保存图片
                    save_image_path = self.images_folder / f"iter_{total_iter}.png"
                    vutils.save_image(fake_images, save_image_path, normalize=True)

                    # Log to TensorBoard
                    grid = make_grid(fake_images, nrow=8, normalize=True, value_range=(-1, 1))
                    self.writer.add_image('Generated Images', grid, total_iter)
                    self.writer.add_scalar('Inception Score', inception_score[0], total_iter)
                    #
                    # # 可选：打印保存图片的消息
                    LOGGER.info(f"Saved images at iteration {total_iter}")

            self.t_end = t.time()
            LOGGER.info("Time of training-{}".format((self.t_end - self.t_begin)))
            # Save Real Inception Score

            # Convert to numpy array if it's a list
            real_inception_scores = np.array(Real_Inception_score)
        except KeyboardInterrupt as e:
            LOGGER.warning("Training interrupted. Saving Real Inception Scores...")
            real_inception_scores = np.array(Real_Inception_score)
        finally:
            if real_inception_scores is not None:
                # Save to pickle file
                score_save_path = self.results_folder / "real_inception_scores.pkl"
                os.makedirs(os.path.dirname(score_save_path), exist_ok=True)
                with open(score_save_path, "wb") as f:
                    pickle.dump(real_inception_scores, f)

                # Also save as text file for easy reading
                txt_save_path = self.results_folder / "real_inception_scores.csv"
                with open(txt_save_path, "w") as f:
                    f.write("Iteration,IS\n")
                    for i, score in enumerate(real_inception_scores):
                        f.write(f"{(i+1)*self.save_interval},{score:.6f}\n")

                best_IS_save_path = (
                    self.results_folder / "best_real_inception_score.csv"
                )
                with open(best_IS_save_path, "w") as f:
                    f.write(f"BestIS,AvgIS\n")
                    f.write(
                        f"{real_inception_scores.max()},{real_inception_scores.mean()}\n"
                    )

                LOGGER.info(
                    f"Real Inception Scores saved to {score_save_path} and {txt_save_path}"
                )
            else:
                LOGGER.warning("No Real Inception Scores to save.")
    
    def train_use_delta(self, train_loader, Real_Inception_score):
        try:
            self.t_begin = t.time()
            self.data = self.get_infinite_batches(train_loader)
            one = torch.tensor(1, dtype=torch.float).to(self.device)
            mone = one * -1

            real_images, _ = next(iter(train_loader))
            real_images = real_images.to(self.device)

            os.makedirs(self.results_folder, exist_ok=True)
            vutils.save_image(
                real_images, self.results_folder / "real_images.png", normalize=True
            )

            total_iter = 0
            D_old, G_old = None, None

            best_real_inception_score = -float("inf")

            for g_iter in tqdm(
                range(self.generator_iters),
                desc=f"Training: optimizer {self.cfg.optimizers.name}",
            ):
                # Requires grad, Generator requires_grad = False
                for p in self.D.parameters():
                    p.requires_grad = True
                if D_old is not None:
                    for p in D_old.parameters():
                        p.requires_grad = True

                d_loss_real = 0
                d_loss_fake = 0
                Wasserstein_D = 0

                for d_iter in range(1):
                    self.D.zero_grad()
                    if D_old is not None:
                        D_old.zero_grad()

                    images = self.data.__next__()
                    images = self.get_torch_variable(images)

                    # Train discriminator
                    # WGAN - Training discriminator more iterations than generator
                    # Train with real images
                    d_loss_real = self.D(images)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(mone)

                    # Train with fake images
                    z = self.get_torch_variable(
                        torch.randn(images.size(0), self.z_dim, 1, 1)
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
                    self.writer.add_scalar('Discriminator Loss', d_loss.item(), total_iter)
                    Wasserstein_D = (d_loss_real - d_loss_fake).item()

                    if D_old is not None:
                        d_loss_real_old = D_old(images).mean()
                        d_loss_real_old.backward(mone)

                        fake_images_ = G_old(z)
                        d_loss_fake_old = D_old(fake_images_).mean()
                        d_loss_fake_old.backward(one)

                        # Train with gradient penalty
                        gradient_penalty_old, _ = self.calculate_gradient_penalty(
                            images.detach(), fake_images_.detach(), eta=eta
                        )
                        gradient_penalty_old.backward()
                        delta_y = [g.grad.data.clone() for g in D_old.parameters()]
                        d_loss_real_old = d_loss_fake_old = gradient_penalty_old = None
                    else:
                        delta_y = None
                    D_old = copy.deepcopy(self.D).to(self.device)
                    self.d_optimizer.step(delta=delta_y)
                    if delta_y is not None:
                        delta_y.clear()
                    total_iter += 1

                # Generator update
                for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation
                if D_old is not None:
                    for p in D_old.parameters():
                        p.requires_grad = False

                self.G.zero_grad()
                if G_old is not None:
                    G_old.zero_grad()
                # train generator
                # compute loss with fake images
                z = self.get_torch_variable(
                    torch.randn(self.batch_size, self.z_dim, 1, 1)
                )
                fake_images = self.G(z)
                g_loss = self.D(fake_images)
                g_loss = g_loss.mean()
                g_loss.backward(mone)
                self.writer.add_scalar('Generator Loss', g_loss.item(), total_iter)
                g_cost = -g_loss
                if G_old is not None:
                    fake_images_ = G_old(z)
                    g_loss_old = D_old(fake_images_).mean()
                    g_loss_old.backward(mone)
                    delta_x = [g.grad.data.clone() for g in G_old.parameters()]
                    g_loss_old = fake_images_ = None
                else:
                    delta_x = None
                # TODO: deepcopy can be optimized
                G_old = copy.deepcopy(self.G).to(self.device)
                self.g_optimizer.step(delta=delta_x)
                if delta_x is not None:
                    delta_x.clear()

                # LOGGER.info(f'Generator iteration: {g_iter}/{self.generator_iters}, '
                #       f'loss_real: {d_loss_real:.4f}, '
                #       f'loss_fake: {d_loss_fake:.4f}, '
                #       f'g_loss: {g_loss:.4f}, '
                #       f'lr_x ={self.lr_x},'
                #       f'lr_y={self.lr_y}, '
                #       f'beta = {self.beta_for_VRAda},'
                #       f'dataset={args.dataset}')

                total_iter += 1
                # Saving model and sampling images every 1000th generator iterations
                if (total_iter) % self.save_interval == 0:
                    grad_g = WGAN_GP_Trainer.get_gradient_norm(self.G).item()
                    grad_d = WGAN_GP_Trainer.get_gradient_norm(self.D).item()
                    # self.save_model()
                    # Workaround because graphic card memory can't store more than 830 examples in memory for generating image
                    # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
                    # This way Inception score is more correct since there are different generated examples from every class of Inception model
                    sample_list = []
                    for _ in range(10):
                        # samples  = self.data.__next__()
                        z = Variable(torch.randn(800, self.z_dim, 1, 1)).to(self.device)
                        samples = self.G(z)
                        # samples = samples.mul(0.5).add(0.5)
                        sample_list.append(samples.data.cpu().numpy())

                    # # Flattening list of list into one list
                    new_sample_list = list(chain.from_iterable(sample_list))
                    LOGGER.info("Calculating Inception Score over 8k generated images")
                    # # Feeding list of numpy arrays
                    # inception_score is a tuple (mean, std)
                    # mean IS and std IS
                    inception_score = get_inception_score(
                        new_sample_list,
                        cuda=True,
                        batch_size=64,
                        resize=True,
                        splits=10,
                    )

                    z = self.get_torch_variable(
                        torch.randn(self.number_of_images, self.z_dim, 1, 1)
                    )
                    Real_Inception_score.append(inception_score[0])

                    if inception_score[0] > best_real_inception_score:
                        best_real_inception_score = inception_score[0]
                        self._save_models_checkpoint(total_iter)
                        LOGGER.info(
                            f"New best Inception Score: {best_real_inception_score:.4f}. Checkpoints saved."
                        )

                    # Testing
                    elapsed_time = t.time() - self.t_begin
                    LOGGER.info(
                        "Real Inception score (mean, std): {}".format(inception_score)
                    )
                    LOGGER.info("Generator iter: {}".format(g_iter))
                    LOGGER.info("total_iter_finished: {}".format(total_iter))
                    LOGGER.info(
                        "Time elapsed: {}".format(
                            str(timedelta(seconds=int(elapsed_time)))
                        )
                    )

                    z = self.get_torch_variable(
                        torch.randn(self.batch_size, self.z_dim, 1, 1)
                    )
                    with torch.no_grad():
                        fake_images = self.G(z).detach().cpu()

                    # 保存图片
                    save_image_path = self.images_folder / f"iter_{total_iter}.png"
                    vutils.save_image(fake_images, save_image_path, normalize=True)

                    # Log to TensorBoard
                    grid = make_grid(fake_images, nrow=8, normalize=True, value_range=(-1, 1))
                    self.writer.add_image('Generated Images', grid, total_iter)
                    self.writer.add_scalar('Inception Score', inception_score[0], total_iter)
                    #
                    # # 可选：打印保存图片的消息
                    LOGGER.info(f"Saved images at iteration {total_iter}")

            self.t_end = t.time()
            LOGGER.info("Time of training-{}".format((self.t_end - self.t_begin)))
            # Save Real Inception Score

            # Convert to numpy array if it's a list
            real_inception_scores = np.array(Real_Inception_score)
        except KeyboardInterrupt as e:
            LOGGER.warning("Training interrupted. Saving Real Inception Scores...")
            real_inception_scores = np.array(Real_Inception_score)
        finally:
            if real_inception_scores is not None:
                # Save to pickle file
                score_save_path = self.results_folder / "real_inception_scores.pkl"
                os.makedirs(os.path.dirname(score_save_path), exist_ok=True)
                with open(score_save_path, "wb") as f:
                    pickle.dump(real_inception_scores, f)

                # Also save as text file for easy reading
                txt_save_path = self.results_folder / "real_inception_scores.csv"
                with open(txt_save_path, "w") as f:
                    f.write("Iteration,IS\n")
                    for i, score in enumerate(real_inception_scores):
                        f.write(f"{(i+1)*self.save_interval},{score:.6f}\n")

                best_IS_save_path = (
                    self.results_folder / "best_real_inception_score.csv"
                )
                with open(best_IS_save_path, "w") as f:
                    f.write(f"BestIS,AvgIS\n")
                    f.write(
                        f"{real_inception_scores.max()},{real_inception_scores.mean()}\n"
                    )

                LOGGER.info(
                    f"Real Inception Scores saved to {score_save_path} and {txt_save_path}"
                )
            else:
                LOGGER.warning("No Real Inception Scores to save.")

    @staticmethod
    def get_gradient_norm(model, norm_type=2.0):
        with torch.no_grad():
            total_norm = torch.norm(
                torch.stack(
                    [torch.norm(p.grad.detach(), norm_type) for p in model.parameters()]
                ),
                norm_type,
            )
        return total_norm

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def get_torch_variable(self, arg):
        return Variable(arg).to(self.device)
