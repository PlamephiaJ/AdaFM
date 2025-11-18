#!/usr/bin/env python3
import csv

with open("results/grid_search/GAN_cifar10/wgan-gp-in/missing_lr.csv") as f:
    reader = csv.DictReader(f)
    groups = []
    for row in reader:
        lx = row["lr_x"]
        ly = row["lr_y"]
        if lx == "N/A" or ly == "N/A":
            continue
        groups.append(f"'optimizers.lr_x={lx} optimizers.lr_y={ly}'")

cmd = "python main.py -m " + " ".join(groups)
print(cmd)

import torch

model = torch.nn.Linear(10, 1)

loss_fn = torch.nn.MSELoss()

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer=torch.optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.1),
    step_size=10,
    gamma=0.1,
)

optimizer = torch.optim.Adagrad(
    params=[torch.nn.Parameter(torch.randn(1))], lr=0.01
)

dataset = torch.utils.data.TensorDataset(
    torch.randn(100, 10), torch.randn(100, 1)
)

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True
)

num_epochs = 5

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    lr_scheduler.step()

generator_model = torch.nn.Linear(10, 1)
discriminator_model = torch.nn.Linear(1, 1)

optimizer_G = torch.optim.Adam(generator_model.parameters(), lr=0.001)
optimizer_D = torch.optim.Adam(discriminator_model.parameters(), lr=0.001)

num_critic = 5
D_model = torch.nn.Linear(1, 1)
G_model = torch.nn.Linear(10, 1)

GP = True  # Gradient Penalty flag

def calculate_gradient_penalty(D, real_data, fake_data):
    return 0.0

for epoch in range(num_epochs):
    # Train Discriminator
    for _ in range(num_critic):
        D_model.zero_grad()
        images = train_loader.__next__()
        d_loss_real = D_model(images).mean()
        d_loss_real.backward()

        noise = torch.randn(images.size(0), 10)
        fake_images = G_model(noise).detach()
        d_loss_fake = D_model(fake_images).mean()
        d_loss_fake.backward()

        if GP:
            gradient_penalty = calculate_gradient_penalty()
            gradient_penalty.backward()
        
        optimizer_D.step()

    # Train Generator
    for p in D_model.parameters(): p.requires_grad = False
    optimizer_G.zero_grad()
    noise = torch.randn(images.size(0), 10)
    fake_images = G_model(noise)
    g_loss = D_model(fake_images).mean()
    g_loss.backward(torch.tensor(-1, dtype=torch.float))
    optimizer_G.step()

import torch
from torch.optim import Optimizer

class MyOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # SGD, in-place operation
                p.add_(grad, alpha=-lr)

        return loss
    

@torch.no_grad()
def step(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        clip_value = group["clip_value"]
        momentum = group["momentum"]
        lr = group["lr"]

        # updates
        for _, p in enumerate(group["params"]):
            if p.grad is None:
                continue
            d_p = torch.clamp(p.grad.data, -clip_value, clip_value)
            if momentum != 0:
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                else:
                    buf = param_state["momentum_buffer"]
                    buf.mul_(1 - momentum).add_(d_p, alpha=momentum)
                d_p = buf

            p.data = p.data - lr * d_p
    return loss