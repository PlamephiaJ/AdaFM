import math
from typing import List, Optional
import torch
from torch.optim import Optimizer


class TiAda_wo_max(Optimizer):

    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        foreach: Optional[bool] = None,
        alpha=0.5,
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
        super(TiAda_wo_max, self).__init__(params, defaults)

        self.alpha = alpha

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
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Update sum of norms
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if torch.is_complex(p) or p.grad.is_sparse:
                        raise NotImplementedError
                    grad = p.grad
                    state = self.state[p]
                    sq_grad = torch.mul(grad, grad)
                    state["sum"].add_(sq_grad)

        for group in self.param_groups:
            lr = group["lr"]
            lr_decay = group["lr_decay"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is not None:

                    state = self.state[p]
                    grad = p.grad
                    state_sum = state["sum"]

                    step_t = state["step"]
                    step_t += 1
                    step = step_t.item()

                    grad = grad if not maximize else -grad

                    if weight_decay != 0:
                        if grad.is_sparse:
                            raise RuntimeError(
                                "weight_decay option is not compatible with sparse gradients"
                            )
                        grad = grad.add(p, alpha=weight_decay)

                    clr = lr / (1 + (step - 1) * lr_decay)
                    # already updated sum
                    ratio_p = state_sum.pow(self.alpha).add_(eps)
                    p.addcdiv_(grad, ratio_p, value=-clr)

        return loss


# Adam


class TiAda_Adam(Optimizer):
    r"""Implements TiAda-Adam algorithm.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used (default: None)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
        capturable (bool, optional): whether this instance is safe to capture in a CUDA graph.
            Passing True can impair ungraphed performance, so if you don't intend to
            graph capture this instance, leave it False (default: False)
        alpha (float): alpha parameter in TiAda
        opponent_optim (optional): If this optimizer is for x, provide the optimizer of y. If
            this optimizer is for y, set it to None.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        alpha=0.5,
        opponent_optim=None,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
        )
        super(TiAda_Adam, self).__init__(params, defaults)

        self.alpha = alpha
        self.opponent_optim = opponent_optim

        # store the total_sum in the same device as the first parameter
        self.total_sum = self.param_groups[0]["params"][0].new_zeros(1)

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    state["step"] = (
                        torch.zeros((1,), dtype=torch.float, device=p.device)
                        if self.defaults["capturable"]
                        else torch.tensor(0.0)
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                        # Update total
                        self.total_sum.add_(state["max_exp_avg_sq"].sum())
                    else:
                        self.total_sum.add_(state["exp_avg_sq"].sum())

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # Performs a health check for CUDA graph capture
        self._cuda_graph_capture_health_check()

        # Initialize loss
        loss = None
        if closure is not None:
            # Compute loss if closure is provided
            with torch.enable_grad():
                loss = closure()

        # Get the 'amsgrad' option from defaults
        amsgrad = self.defaults["amsgrad"]

        # Reset the total sum to 0
        self.total_sum.zero_()

        # Update the states and compute the exponential moving averages
        for group in self.param_groups:
            # Extract parameters for the current group
            beta1, beta2 = group["betas"]
            maximize = group["maximize"]
            capturable = group["capturable"]
            weight_decay = group["weight_decay"]
            lr = group["lr"]
            eps = group["eps"]

            for p in group["params"]:
                param = p
                if p.grad is not None:
                    # Check if gradient is sparse
                    if p.grad.is_sparse:
                        # Adam doesn't support sparse gradients
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )

                    # Retrieve the state for current parameter
                    state = self.state[p]

                    # If maximize is true, use negative gradient
                    grad = param.grad if not maximize else -param.grad

                    # Retrieve the exponential moving averages
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    step_t = state["step"]

                    if capturable:
                        # Check for special handling for capturable parameters
                        # Currently, not implemented
                        raise NotImplementedError

                    # Add weight decay if specified
                    if weight_decay != 0:
                        grad = grad.add(param, alpha=weight_decay)

                    # Update the first moment (mean) and second moment (uncentered variance)
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                    if amsgrad:
                        # If AMSGrad variant is used
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        # Maintain the maximum of all 2nd moment running averages till now
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Update the total sum using max_exp_avg_sq
                        self.total_sum.add_(max_exp_avg_sq.sum())
                    else:
                        # Update the total sum using exp_avg_sq
                        self.total_sum.add_(exp_avg_sq.sum())

        # Calculate the ratio based on opponent optimizer if provided
        if self.opponent_optim is not None:
            ratio = self.total_sum.pow(self.alpha)
            ratio.div_(torch.max(ratio, self.opponent_optim.total_sum.pow(self.alpha)))
        else:
            ratio = 1

        # Actual parameters update
        for group in self.param_groups:
            # Extract parameters for the current group
            beta1, beta2 = group["betas"]
            maximize = group["maximize"]
            capturable = group["capturable"]
            weight_decay = group["weight_decay"]
            lr = group["lr"]
            eps = group["eps"]

            for p in group["params"]:
                param = p
                if p.grad is not None:
                    # Check again for sparse gradients
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )

                    # Retrieve state and gradient for current parameter
                    state = self.state[p]
                    grad = param.grad if not maximize else -param.grad
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    step_t = state["step"]

                    # Increment the step counter
                    step_t += 1

                    if capturable:
                        # Check for special handling for capturable parameters
                        raise NotImplementedError
                    else:
                        step = step_t.item()

                    # Compute bias corrections
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    step_size = lr / bias_correction1
                    bias_correction2_sqrt = math.sqrt(bias_correction2)

                    # Compute the adaptive learning rate denominator
                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        denom = (
                            max_exp_avg_sq.pow(self.alpha) / bias_correction2_sqrt
                        ).add_(eps)
                    else:
                        denom = (
                            exp_avg_sq.pow(self.alpha) / bias_correction2_sqrt
                        ).add_(eps)

                    denom.div_(ratio)

                    # Apply the Adam update rule
                    param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# Copy of AdaGrad, customized to compute the effective stepsize


class TiAda(Optimizer):
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
        alpha=0.5,
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

        self.alpha = alpha
        self.opponent_optim = opponent_optim
        # whether to compute effective_stepsize
        self.compute_effective_stepsize = compute_effective_stepsize

        super(TiAda, self).__init__(params, defaults)

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
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Update sum of norms
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if torch.is_complex(p) or p.grad.is_sparse:
                        raise NotImplementedError
                    grad = p.grad
                    state = self.state[p]
                    sq_grad = torch.mul(grad, grad)
                    state["sum"].add_(sq_grad)
                    self.total_sum.add_(sq_grad.sum())

        # calculate the ratio
        if self.opponent_optim is not None:
            ratio = self.total_sum.pow(self.alpha)
            ratio.div_(torch.max(ratio, self.opponent_optim.total_sum.pow(self.alpha)))
        else:
            ratio = 1

        for group in self.param_groups:
            lr = group["lr"]
            lr_decay = group["lr_decay"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is not None:

                    state = self.state[p]
                    grad = p.grad
                    state_sum = state["sum"]

                    step_t = state["step"]
                    step_t += 1
                    step = step_t.item()

                    grad = grad if not maximize else -grad

                    if weight_decay != 0:
                        if grad.is_sparse:
                            raise RuntimeError(
                                "weight_decay option is not compatible with sparse gradients"
                            )
                        grad = grad.add(p, alpha=weight_decay)

                    clr = lr / (1 + (step - 1) * lr_decay)
                    # already updated sum
                    ratio_p = state_sum.pow(self.alpha).add_(eps).div_(ratio)
                    p.addcdiv_(grad, ratio_p, value=-clr)
                    if self.compute_effective_stepsize:
                        self.effective_stepsize = (clr / ratio_p).item()

        return loss
