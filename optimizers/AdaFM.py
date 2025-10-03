import torch
from torch.optim import Optimizer
from typing import Optional


class AdaFM(Optimizer):
    r"""Implements TiAda algorithm.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        foreach (bool, optional): whether foreach implementation of optimizer is used (default: None)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
        alpha (float): alpha parameter in TiAda
        opponent_optim (optional): If this optimizer is for x, provide the optimizer of y. If
            this optimizer is for y, set it to None.
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        foreach: Optional[bool] = None,
        beta=0.9,
        opponent_optim=None,
        compute_effective_stepsize=False,
        *,
        maximize: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                "Invalid initial_accumulator_value value: {}".format(
                    initial_accumulator_value
                )
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            foreach=foreach,
            maximize=maximize,
        )

        self.beta = beta
        self.opponent_optim = opponent_optim
        # whether to compute effective_stepsize
        self.compute_effective_stepsize = compute_effective_stepsize

        super(AdaFM, self).__init__(params, defaults)

        # store the total_sum in the same device as the first parameter
        self.total_sum = self.param_groups[0]["params"][0].new_zeros(1)

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    state["step"] = torch.tensor(0.0)
                    init_value = (
                        complex(initial_accumulator_value, initial_accumulator_value)
                        if torch.is_complex(p)
                        else initial_accumulator_value
                    )
                    state["sum"] = torch.full_like(
                        p, init_value, memory_format=torch.preserve_format
                    )

                    # Update total_sum
                    self.total_sum.add_(state["sum"].sum())

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)

        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

    @torch.no_grad()
    def step(self, closure=None, delta=None):
        """Performs a single optimization step.

        Args:
            delta (tensor): gradient from old iter model.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历每一个参数组并更新梯度的平方和。
        if delta is None:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        # 如果参数是复数或者梯度是稀疏的，抛出异常。
                        if torch.is_complex(p) or p.grad.is_sparse:
                            raise NotImplementedError
                        grad = p.grad
                        state = self.state[p]
                        # todo:步骤7,8更新梯度
                        d_p = state["est"] = torch.clone(grad).detach()
                        sq_grad = torch.mul(d_p, d_p.conj()) / self.beta  # 梯度的平方
                        state["sum"].add_(sq_grad)
                        self.total_sum.add_(sq_grad.sum())
        else:
            for group in self.param_groups:
                for i, (p, delta_x_i) in enumerate(zip(group["params"], delta)):
                    if p.grad is not None:
                        # 如果参数是复数或者梯度是稀疏的，抛出异常。
                        if torch.is_complex(p) or p.grad.is_sparse:
                            raise NotImplementedError
                        grad = p.grad
                        state = self.state[p]
                        # todo:步骤7,8更新梯度
                        d_p = state["est"]
                        d_p.sub_(delta_x_i).mul_(1 - self.beta).add_(grad)
                        sq_grad = torch.mul(d_p, d_p.conj()) / self.beta  # 梯度的平方
                        state["sum"].add_(sq_grad)
                        self.total_sum.add_(sq_grad.sum())

        # 如果存在对手的优化器，则计算比率。
        if self.opponent_optim is not None:
            ratio = self.total_sum.pow(1 / 3)
            ratio.div_(torch.max(ratio, self.opponent_optim.total_sum.pow(1 / 3)))
        else:
            ratio = 1
        # 遍历每一个参数组进行参数更新。
        for group in self.param_groups:
            lr = group["lr"]
            lr_decay = group["lr_decay"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            maximize = group["maximize"]

            for i, p in enumerate(group["params"]):
                if p.grad is not None:
                    state = self.state[p]
                    # grad = p.grad
                    grad = state["est"]
                    state_sum = state["sum"]

                    step_t = state["step"]
                    step_t += 1
                    step = step_t.item()

                    # if delta is not None:
                    #     d_p = state["lastest_grad"]
                    #     d_p.sub_(delta[i]).mul_(1 - self.beta).add_(grad)
                    #     grad = d_p
                    # else:
                    #     state["lastest_grad"] = grad

                    # 如果maximize为True，取梯度的负值。
                    grad_m = grad if not maximize else -grad

                    # 如果设置了权重衰减并且梯度不是稀疏的，应用权重衰减。
                    if weight_decay != 0:
                        if grad_m.is_sparse:
                            raise RuntimeError(
                                "weight_decay option is not compatible with sparse gradients"
                            )
                        # L2正则项求导
                        grad_m.add_(p.data, alpha=weight_decay)

                    # 计算学习率的衰减
                    clr = lr / (1 + (step - 1) * lr_decay)

                    # 根据之前计算的比率更新参数。
                    ratio_p = state_sum.pow(1 / 3).add_(eps).div_(ratio)
                    p.data.addcdiv_(grad_m, ratio_p, value=-clr)
                    # print(clr / ratio_p)
                    # 如果设置了计算有效的步长大小，计算它。
                    if self.compute_effective_stepsize:
                        self.effective_stepsize = (clr / ratio_p).item()

        return loss
