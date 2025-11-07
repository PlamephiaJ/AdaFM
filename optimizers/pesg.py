import torch
from pathlib import Path
from typing import Optional
import copy
import logging

LOGGER = logging.getLogger(__name__)


class PESG(torch.optim.Optimizer):
    """Proximal Epoch Stochastic Gradient (PESG) for GANs"""

    def __init__(
        self,
        params,
        total_iter: int,
        lr: float = 0.1,
        clip_value: float = 1.0,
        weight_decay: float = 1e-5,
        epoch_decay: float = 2e-3,
        momentum: float = 0,
        decay_iters: Optional[list[int]] = None,
        decay_factor: int = 3,
        opponent_optim: Optional[torch.optim.Optimizer] = None,
        results_folder: Optional[Path] = None,
        *,
        maximize: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if clip_value < 0.0:
            raise ValueError("Invalid clip_value: {}".format(clip_value))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay: {}".format(weight_decay))
        if epoch_decay < 0.0:
            raise ValueError("Invalid epoch_decay: {}".format(epoch_decay))
        if momentum < 0.0:
            raise ValueError("Invalid momentum: {}".format(momentum))

        params = list(params)

        self.model_ref = []
        self.init_model_ref(params=params)
        self.model_acc = []
        self.init_model_acc(params=params)

        if not decay_iters:
            self.decay_iters = [total_iter // 2, (3 * total_iter) // 4]
        else:
            self.decay_iters = decay_iters

        self.decay_factor = decay_factor

        self.T = 0
        self.steps = 0

        self.device = torch.device("cuda:0")

        defaults = dict(
            lr=lr,
            clip_value=clip_value,
            weight_decay=weight_decay,
            epoch_decay=epoch_decay,
            momentum=momentum,
            model_ref=self.model_ref,
            model_acc=self.model_acc,
            opponent_optim=opponent_optim,
            maximize=maximize,
        )

        self.opponent_optim = opponent_optim

        super().__init__(params, defaults)

        self.results_folder = results_folder
        if self.results_folder is None:
            raise ValueError("results_folder must be provided.")
        if opponent_optim is not None:
            self.optimizer_log_path = Path(self.results_folder) / "optimizer_log_x.csv"
        else:
            self.optimizer_log_path = Path(self.results_folder) / "optimizer_log_y.csv"
        self.optimizer_log_file = open(self.optimizer_log_path, "w")
        self.optimizer_log_file.write("step,learning_rate\n")

    def init_model_ref(self, params):
        for var in params:
            if var is not None:
                self.model_ref.append(
                    torch.empty(var.shape).normal_(mean=0, std=0.01).to(var.device)
                )

    def init_model_acc(self, params):
        for var in params:
            if var is not None:
                self.model_acc.append(
                    torch.zeros(
                        var.shape,
                        dtype=torch.float32,
                        device=var.device,
                        requires_grad=False,
                    )
                )

    def __setstate__(self, state):
        super(PESG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @property
    def optim_steps(self):
        return self.steps

    @torch.no_grad()
    def step(self, closure=None, delta=None):
        if self.steps in self.decay_iters:
            self.update_regularizer(self.decay_factor)

        if self.opponent_optim is not None:
            return self.step_for_x(closure, delta)
        else:
            return self.step_for_y(closure, delta)

    @torch.no_grad()
    def step_for_x(self, closure=None, delta=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            clip_value = group["clip_value"]
            momentum = group["momentum"]
            lr = group["lr"]

            epoch_decay = group["epoch_decay"]
            model_ref = group["model_ref"]
            model_acc = group["model_acc"]

            # updates
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                d_p = (
                    torch.clamp(p.grad.data, -clip_value, clip_value)
                    + epoch_decay * (p.data - model_ref[i].data)
                    + weight_decay * p.data
                )
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(1 - momentum).add_(d_p, alpha=momentum)
                    d_p = buf
                    p.data = p.data - lr * d_p
                model_acc[i].data = model_acc[i].data + p.data

        self.T += 1
        self.steps += 1
        return loss

    @torch.no_grad()
    def step_for_y(self, closure=None, delta=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            clip_value = group["clip_value"]
            lr = group["lr"]

            # updates
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                d_p = torch.clamp(p.grad.data, -clip_value, clip_value)
                p.data = p.data - lr * d_p

        self.T += 1
        self.steps += 1
        return loss

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad = None

    def update_regularizer(self, decay_factor=None):
        if decay_factor != None:
            self.param_groups[0]["lr"] = self.param_groups[0]["lr"] / decay_factor
            LOGGER.info(
                "Reducing learning rate to %.5f @ T=%s!",
                self.param_groups[0]["lr"],
                self.steps,
            )

        LOGGER.info("Updating regularizer @ T=%s!", self.steps)
        for i, param in enumerate(self.model_ref):
            self.model_ref[i].data = self.model_acc[i].data / self.T
        for i, param in enumerate(self.model_acc):
            self.model_acc[i].data = torch.zeros(
                param.shape,
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )
        self.T = 0
