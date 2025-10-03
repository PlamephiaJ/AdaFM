import torch
from torch.optim import Optimizer
from typing import Optional

class RSGDA(Optimizer):
    def __init__(
        self,
        params,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        foreach: Optional[bool] = None,
        beta_x=0.9,
        beta_y=0.9,
        lr_x=0.1,
        lr_y=0.1,

        opponent_optim = None,
        compute_effective_stepsize=False,
        *,
        maximize: bool = False,
    ):
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
            lr_x=lr_x,
            lr_y=lr_y,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            foreach=foreach,
            maximize=maximize,
        )

        self.beta_x = beta_x
        self.beta_y = beta_y
        self.lr_x = lr_x
        self.lr_y = lr_y
        self.opponent_optim = opponent_optim
        super(RSGDA, self).__init__(params, defaults)

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
    def step(self, closure=None,delta=None):
        """Performs a single optimization step.

        Args:
            delta (tensor): gradient from old iter model.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.opponent_optim is not None:
            lr = self.lr_x
            beta = self.beta_x
        else:
            lr = self.lr_y
            beta = self.beta_y

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
                        d_p = state['est'] = torch.clone(grad).detach()

        else:
            for group in self.param_groups:
                for i, (p, delta_x_i) in enumerate(zip(group['params'], delta)):
                    if p.grad is not None:
                        # 如果参数是复数或者梯度是稀疏的，抛出异常。
                        if torch.is_complex(p) or p.grad.is_sparse:
                            raise NotImplementedError
                        grad = p.grad
                        state = self.state[p]
                        # todo:步骤7,8更新梯度
                        d_p = state['est']
                        d_p.sub_(delta_x_i).mul_(1 - beta).add_(grad)



        # 遍历每一个参数组进行参数更新。
        for group in self.param_groups:
            lr_decay = group["lr_decay"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            maximize = group["maximize"]

            for i,p in enumerate(group["params"]):
                if p.grad is not None:
                    state = self.state[p]
                    # grad = p.grad
                    grad = state['est']
                    state_sum = state["sum"]

                    step_t = state["step"]
                    step_t += 1
                    step = step_t.item()


                    # 如果maximize为True，取梯度的负值。
                    grad_m = grad if not maximize else -grad

                    # 如果设置了权重衰减并且梯度不是稀疏的，应用权重衰减。
                    if weight_decay != 0:
                        if grad_m.is_sparse:
                            raise RuntimeError(
                                "weight_decay option is not compatible with sparse gradients"
                            )
                        # L2正则项求导
                        grad_m.add_(p, alpha=weight_decay)

                    # 计算学习率的衰减
                    clr = lr / (1 + (step - 1) * lr_decay)
                    # 根据之前计算的比率更新参数。
                    p.add_(grad_m, alpha=-clr)

        return loss