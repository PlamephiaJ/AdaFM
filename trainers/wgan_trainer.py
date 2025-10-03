import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import time as t
from itertools import chain
import copy
from .utils.inception_score import get_inception_score


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
        device=None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = device
        self.number_of_images = 100
        self.generator_iters = generator_iters
        self.critic_iters = critic_iters
        self.lambda_term = 1e-4
        self.save_interval = save_interval

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
            grad_outputs=torch.ones(prob_interpolated.size()).cuda(self.cuda_index)
            if self.cuda
            else torch.ones(prob_interpolated.size()),
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty, eta

    def train(self, train_loader, Real_Inception_score, args):
        args = args
        self.t_begin = t.time()
        self.data = self.get_infinite_batches(train_loader)
        one = torch.tensor(1, dtype=torch.float).to(self.device)
        mone = one * -1

        real_images, _ = next(iter(train_loader))
        real_images = real_images.to(self.device)

        # save_path = './GAN/gan_fake_images_c100/real_images.png'

        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # vutils.save_image(real_images, save_path, normalize=True)

        total_iter = 0
        D_old, G_old = None, None

        for g_iter in range(self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True
            if D_old is not None:
                for p in D_old.parameters():
                    p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0

            for d_iter in range(self.critic_iter):
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
                    images.data, fake_images.data, eta=None
                )
                gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = (d_loss_real - d_loss_fake).item()

                if D_old is not None:
                    d_loss_real_old = D_old(images).mean()
                    d_loss_real_old.backward(mone)

                    fake_images_ = G_old(z)
                    d_loss_fake_old = D_old(fake_images_).mean()
                    d_loss_fake_old.backward(one)

                    # Train with gradient penalty
                    gradient_penalty_old, _ = self.calculate_gradient_penalty(
                        images.data, fake_images_.data, eta=eta
                    )
                    gradient_penalty_old.backward()
                    delta_y = [g.grad.data.clone() for g in D_old.parameters()]
                    d_loss_real_old = d_loss_fake_old = gradient_penalty_old = None
                else:
                    delta_y = None
                D_old = copy.deepcopy(self.D).cuda()
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
            z = self.get_torch_variable(torch.randn(self.batch_size, self.z_dim, 1, 1))
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
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
            G_old = copy.deepcopy(self.G).cuda()
            self.g_optimizer.step(delta=delta_x)
            if delta_x is not None:
                delta_x.clear()

            # print(f'Generator iteration: {g_iter}/{self.generator_iters}, '
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
                    z = Variable(torch.randn(800, self.z_dim, 1, 1)).cuda(
                        self.cuda_index
                    )
                    samples = self.G(z)
                    # samples = samples.mul(0.5).add(0.5)
                    sample_list.append(samples.data.cpu().numpy())

                # # Flattening list of list into one list
                new_sample_list = list(chain.from_iterable(sample_list))
                print("Calculating Inception Score over 8k generated images")
                # # Feeding list of numpy arrays
                inception_score = get_inception_score(
                    new_sample_list, cuda=True, batch_size=64, resize=True, splits=10
                )

                z = self.get_torch_variable(
                    torch.randn(self.number_of_images, self.z_dim, 1, 1)
                )
                Real_Inception_score.append(inception_score[0])
                # Testing
                # time = t.time() - self.t_begin
                print("Real Inception score: {}".format(inception_score))
                print("Generator iter: {}".format(g_iter))
                print("total_iter: {}".format(total_iter))
                # print("Time {}".format(time))
                # z = self.get_torch_variable(torch.randn(self.batch_size, self.z_dim, 1, 1))
                # with torch.no_grad():
                #     fake_images = self.G(z).detach().cpu()

                # 保存图片
                # save_image_path = f'/home/panxiaokang/parameter_free/GAN/gan_fake_images_c100/iter_{total_iter}.png'
                # vutils.save_image(fake_images, save_image_path, normalize=True)
                #
                # # 可选：打印保存图片的消息
                # print(f'Saved images at iteration {total_iter}')

        # self.t_end = t.time()
        # print('Time of training-{}'.format((self.t_end - self.t_begin)))

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
