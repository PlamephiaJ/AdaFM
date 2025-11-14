import logging
from pathlib import Path

import torch

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
        decay_iters: list[int] | None = None,
        decay_factor: int = 3,
        opponent_optim: torch.optim.Optimizer | None = None,
        results_folder: Path | None = None,
        tb_writer=None,
        *,
        maximize: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if clip_value < 0.0:
            raise ValueError(f"Invalid clip_value: {clip_value}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if epoch_decay < 0.0:
            raise ValueError(f"Invalid epoch_decay: {epoch_decay}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        params = list(params)

        self.model_ref = []
        self.init_model_ref(params=params)
        self.model_acc = []
        self.init_model_acc(params=params)

        self.tb_writer = tb_writer

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
        if self.opponent_optim is not None:
            return self.step_for_x(closure, delta)
        else:
            return self.step_for_y(closure, delta)

    @torch.no_grad()
    def step_for_x(self, closure=None, delta=None):
        # Compute raw gradient norm before any processing
        grad_norm = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm**0.5

        # Log raw gradient norm to TensorBoard
        if self.tb_writer is not None:
            # Use opponent_optim to determine if this is generator or discriminator
            tag = (
                "Raw_Gradient_Norm/generator"
                if self.opponent_optim is not None
                else "Raw_Gradient_Norm/discriminator"
            )
            self.tb_writer.add_scalar(tag, grad_norm, self.steps)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Compute processed gradient norm (d_p)
        processed_grad_norm = 0.0
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

                # Accumulate processed gradient norm
                processed_grad_norm += d_p.norm(2).item() ** 2

                p.data = p.data - lr * d_p
                model_acc[i].data = model_acc[i].data + p.data

        processed_grad_norm = processed_grad_norm**0.5

        # Log processed gradient norm to TensorBoard
        if self.tb_writer is not None:
            tag = (
                "Processed_Gradient_Norm/generator"
                if self.opponent_optim is not None
                else "Processed_Gradient_Norm/discriminator"
            )
            self.tb_writer.add_scalar(tag, processed_grad_norm, self.steps)

        self.T += 1
        self.steps += 1
        return loss

    @torch.no_grad()
    def step_for_y(self, closure=None, delta=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 计算所有参数的原始梯度的二范数
        grad_norm = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm**0.5

        # 写入TensorBoard - Use opponent_optim to determine if this is generator or discriminator
        if self.tb_writer is not None:
            tag = (
                "Raw_Gradient_Norm/generator"
                if self.opponent_optim is not None
                else "Raw_Gradient_Norm/discriminator"
            )
            self.tb_writer.add_scalar(tag, grad_norm, self.steps)

        # Compute processed gradient norm (d_p)
        processed_grad_norm = 0.0
        for group in self.param_groups:
            clip_value = group["clip_value"]
            lr = group["lr"]

            # updates
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                d_p = torch.clamp(p.grad.data, -clip_value, clip_value)

                # Accumulate processed gradient norm
                processed_grad_norm += d_p.norm(2).item() ** 2

                p.data = p.data - lr * d_p

        processed_grad_norm = processed_grad_norm**0.5

        # Log processed gradient norm to TensorBoard
        if self.tb_writer is not None:
            tag = (
                "Processed_Gradient_Norm/generator"
                if self.opponent_optim is not None
                else "Processed_Gradient_Norm/discriminator"
            )
            self.tb_writer.add_scalar(tag, processed_grad_norm, self.steps)

        self.T += 1
        self.steps += 1
        return loss

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad = None

    def update_regularizer(self, decay_factor=None):
        if decay_factor is not None:
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
