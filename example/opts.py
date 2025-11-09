# ======= AdaFM 2-variable + visualization =======
import torch
import matplotlib.pyplot as plt
import numpy as np

"""
Single-loop Optimzers
"""
# ------------------------
# Toy AdaFM optimizer
# ------------------------
class AdaFM2Var:
    def __init__(self, x, y, lr_x=0.1, lr_y=0.1, beta=0.9, lr_decay=0.0, weight_decay=0.0, eps=1e-10):
        self.x = x
        self.y = y
        self.lr_x = float(lr_x)
        self.lr_y = float(lr_y)
        self.beta = float(beta)
        self.lr_decay = float(lr_decay)
        self.weight_decay = float(weight_decay)
        self.eps = float(eps)

        self.step_t = 0
        self.est_x = torch.zeros_like(x)
        self.est_y = torch.zeros_like(y)
        self.sum_x = torch.zeros_like(x)
        self.sum_y = torch.zeros_like(y)

    @torch.no_grad()
    def _accumulate(self, gx, gy):
        self.est_x = gx
        self.est_y = gy
        self.sum_x.add_((self.est_x * self.est_x) / self.beta)
        self.sum_y.add_((self.est_y * self.est_y) / self.beta)

    def step(self, loss_fn):
        self.step_t += 1

        loss = loss_fn(self.x, self.y)
        (d_fx,) = torch.autograd.grad(loss, self.x, retain_graph=True)
        (d_fy,) = torch.autograd.grad(loss, self.y)

        if self.weight_decay != 0.0:
            d_fx = d_fx + self.weight_decay * self.x
            d_fy = d_fy + self.weight_decay * self.y

        self._accumulate(d_fx, d_fy)

        clr_x = self.lr_x / (1.0 + (self.step_t - 1) * self.lr_decay)
        clr_y = self.lr_y / (1.0 + (self.step_t - 1) * self.lr_decay)

        root_x = torch.pow(self.sum_x, 1.0 / 3.0 + 0.1)
        root_y = torch.pow(self.sum_y, 1.0 / 3.0 - 0.1)
        common = torch.maximum(root_x, root_y)

        ratio_px = (root_x + self.eps) * (common / (root_x + 1e-12))
        ratio_py = (root_y + self.eps) * (root_y / (root_y + 1e-12))

        with torch.no_grad():
            self.x.add_(-clr_x * (d_fx / ratio_px))
            self.y.add_(+clr_y * (d_fy / ratio_py))

        return loss.detach()

# ------------------------
# Toy MSGDA optimizer
# ------------------------
class MSGDA2Var:
    def __init__(self, x, y, lr_x=0.1, lr_y=0.1, beta=0.9, lr_decay=0.0, weight_decay=0.0, eps=1e-10):
        self.x = x
        self.y = y
        self.lr_x = float(lr_x)
        self.lr_y = float(lr_y)
        self.beta = float(beta)
        self.lr_decay = float(lr_decay)
        self.weight_decay = float(weight_decay)
        self.eps = float(eps)

        # algorithm hyper-parameters from full MSGDA
        # time factor parameters (k,m) replicate k/(m+step)^(1/3)
        self.k = 5.0
        self.m = 125.0

        # state
        self.step_t = 0
        # momentum buffers (damping with (1-beta))
        self.mom_x = torch.zeros_like(x)
        self.mom_y = torch.zeros_like(y)

    @torch.no_grad()
    def _update_momentum(self, gx, gy):
        """MSGDA momentum update: m <- (1-beta)*m + grad (or pure grad first iter)."""
        if self.step_t == 1:
            # first step just take gradient
            self.mom_x.copy_(gx)
            self.mom_y.copy_(gy)
        else:
            self.mom_x.mul_(1 - self.beta).add_(gx)
            self.mom_y.mul_(1 - self.beta).add_(gy)

    def step(self, loss_fn):
        """
        Perform one MSGDA style update on two scalar/low-dim variables x (descent) and y (ascent).

        loss_fn: callable(x,y) -> scalar loss (we minimize w.r.t x and maximize w.r.t y)
        """
        self.step_t += 1

        loss = loss_fn(self.x, self.y)
        (d_fx,) = torch.autograd.grad(loss, self.x, retain_graph=True)
        (d_fy,) = torch.autograd.grad(loss, self.y)

        # apply weight decay like L2 regularization
        if self.weight_decay != 0.0:
            d_fx = d_fx + self.weight_decay * self.x
            d_fy = d_fy + self.weight_decay * self.y

        # update momentum buffers
        self._update_momentum(d_fx, d_fy)

        # compute decayed base lr
        clr_x = self.lr_x / (1.0 + (self.step_t - 1) * self.lr_decay)
        clr_y = self.lr_y / (1.0 + (self.step_t - 1) * self.lr_decay)

        # time scaling factor
        time_factor = self.k / (self.m + (self.step_t - 1)) ** (1.0 / 3.0)

        # effective step size for x (descent) and y (ascent) use momentum
        # In original MSGDA both sides share same scalar time_factor.
        eff_lr_x = clr_x * time_factor
        eff_lr_y = clr_y * time_factor

        with torch.no_grad():
            # descent on x
            self.x.add_(self.mom_x, alpha=-eff_lr_x)
            # ascent on y (reverse sign)
            self.y.add_(self.mom_y, alpha=+eff_lr_y)

        return loss.detach()
    
# -------------------
# Toy TiAda optimizer for 2 variables
# -------------------
class TiAda2Var:
    def __init__(
        self,
        x,
        y,
        lr_x=0.1,
        lr_y=0.1,
        lr_decay=0.0,
        weight_decay=0.0,
        eps=1e-10,
        alpha=0.5,
        compute_effective_stepsize=False,
    ):
        """
        Two-variable toy TiAda optimizer (min over x, max over y).

        Mirrors core ideas from optimizers/TiAda.py:
        - Accumulate squared gradients per-variable (sum_x, sum_y)
        - Use exponent alpha to form adaptive preconditioners
        - Balance x/y via a scalar ratio based on total accumulators
        - Optional lr decay and weight decay
        """
        self.x = x
        self.y = y
        self.lr_x = float(lr_x)
        self.lr_y = float(lr_y)
        self.lr_decay = float(lr_decay)
        self.weight_decay = float(weight_decay)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.compute_effective_stepsize = bool(compute_effective_stepsize)

        self.step_t = 0
        # Per-variable accumulators of squared gradients
        self.sum_x = torch.zeros_like(x)
        self.sum_y = torch.zeros_like(y)

        # Diagnostics (optional)
        self.effective_stepsize_x = None
        self.effective_stepsize_y = None

    def _calc_ratio_xy(self):
        """Compute scalar balance ratios for x and y based on total accumulators.

        ratio_x = total_x^alpha / max(total_x^alpha, total_y^alpha)
        ratio_y = total_y^alpha / max(total_x^alpha, total_y^alpha)
        If both totals are zero, return (1.0, 1.0).
        """
        with torch.no_grad():
            total_x = self.sum_x.sum()
            total_y = self.sum_y.sum()
            # Convert totals to alpha-scaled magnitudes
            txa = total_x.pow(self.alpha - 0.1)
            tya = total_y.pow(self.alpha + 0.1)
            denom = torch.maximum(txa, tya)
            if denom.item() == 0.0:
                return 1.0, 1.0
            rx = (txa / denom).item()
            ry = (tya / denom).item()
            return rx, ry

    def step(self, loss_fn):
        """Perform one TiAda-style update for min_x max_y.

        loss_fn: callable(x, y) -> scalar loss
        """
        self.step_t += 1

        loss = loss_fn(self.x, self.y)
        (d_fx,) = torch.autograd.grad(loss, self.x, retain_graph=True)
        (d_fy,) = torch.autograd.grad(loss, self.y)

        # Optional L2 weight decay
        if self.weight_decay != 0.0:
            d_fx = d_fx + self.weight_decay * self.x
            d_fy = d_fy + self.weight_decay * self.y

        # Update accumulators with squared gradients
        with torch.no_grad():
            self.sum_x.add_(d_fx * d_fx)
            self.sum_y.add_(d_fy * d_fy)

        # Learning rate with decay
        clr_x = self.lr_x / (1.0 + (self.step_t - 1) * self.lr_decay)
        clr_y = self.lr_y / (1.0 + (self.step_t - 1) * self.lr_decay)

        # Compute scalar ratios for x and y
        ratio_x, ratio_y = self._calc_ratio_xy()

        # Per-variable denominators: |g| accumulation to the power alpha, eps for stability,
        # and divided by the cross-player ratio to balance the two sides
        denom_x = self.sum_x.pow(self.alpha).add(self.eps).div(ratio_x)
        denom_y = self.sum_y.pow(self.alpha).add(self.eps).div(ratio_y)

        # Parameter updates: descent for x, ascent for y
        with torch.no_grad():
            # x <- x - clr * d_fx / denom_x
            self.x.addcdiv_(d_fx, denom_x, value=-clr_x)
            # y <- y + clr * d_fy / denom_y
            self.y.addcdiv_(d_fy, denom_y, value=+clr_y)

            if self.compute_effective_stepsize:
                # Use L2 norm as a scalar proxy for effective stepsizes
                eff_x = (clr_x / denom_x).norm(p=2)
                eff_y = (clr_y / denom_y).norm(p=2)
                self.effective_stepsize_x = eff_x.item()
                self.effective_stepsize_y = eff_y.item()

        return loss.detach()
    
# -------------------------
# Toy PESG optimizer for 2 variables
# -------------------------
class PESG2Var:
    def __init__(
        self,
        x,
        y,
        lr_x=0.1,
        lr_y=0.1,
        clip_value=1.0,
        weight_decay=1e-5,
        epoch_decay=2e-3,
        momentum=0.0,
        total_iter=10000,
        decay_iters=None,
        decay_factor=3,
    ):
        """
        Two-variable toy PESG optimizer (min over x, max over y).

        Mirrors core ideas from optimizers/pesg.py:
        - x update uses clipped gradient + proximal epoch regularizer + weight decay (+ optional momentum)
        - y update uses clipped gradient (ascent in our toy to match max_y)
        - periodic regularizer update averages model_acc into model_ref and optionally decays lr
        """
        self.x = x
        self.y = y

        self.lr_x = float(lr_x)
        self.lr_y = float(lr_y)
        self.clip_value = float(clip_value)
        self.weight_decay = float(weight_decay)
        self.epoch_decay = float(epoch_decay)
        self.momentum = float(momentum)
        self.decay_factor = int(decay_factor)

        self.total_iter = int(total_iter)
        if decay_iters is None:
            self.decay_iters = [self.total_iter // 2, (3 * self.total_iter) // 4]
        else:
            self.decay_iters = list(decay_iters)

        # Internal state
        self.steps = 0
        self.T = 0  # epoch counter for averaging

        # Model references and accumulators for x only (as in PESG)
        with torch.no_grad():
            self.model_ref_x = torch.empty_like(x).normal_(mean=0.0, std=0.01)
            self.model_acc_x = torch.zeros_like(x)

        # Optional momentum buffer for x
        self.mom_x = None

    def _maybe_update_regularizer(self):
        # Update regularizer and optionally decay lr at specified steps
        if self.steps in self.decay_iters:
            # Decay learning rate
            self.lr_x = self.lr_x / float(self.decay_factor)
            self.lr_y = self.lr_y / float(self.decay_factor)
        # Average accumulated model over the epoch T
        if self.T > 0:
            with torch.no_grad():
                self.model_ref_x.copy_(self.model_acc_x / float(self.T))
                self.model_acc_x.zero_()
            self.T = 0

    def step(self, loss_fn):
        """Perform one PESG-style update for min_x max_y."""
        # compute gradients
        loss = loss_fn(self.x, self.y)
        (d_fx,) = torch.autograd.grad(loss, self.x, retain_graph=True)
        (d_fy,) = torch.autograd.grad(loss, self.y)

        cv = self.clip_value

        # X update (descent) with proximal and weight decay, plus optional momentum
        with torch.no_grad():
            dpx = torch.clamp(d_fx, -cv, cv)
            dpx = dpx + self.epoch_decay * (self.x - self.model_ref_x) + self.weight_decay * self.x

            if self.momentum != 0.0:
                if self.mom_x is None:
                    self.mom_x = dpx.clone().detach()
                else:
                    # buf <- (1-m) * buf + m * dpx (match repo pattern)
                    self.mom_x.mul_(1.0 - self.momentum).add_(dpx, alpha=self.momentum)
                step_dir_x = self.mom_x
            else:
                step_dir_x = dpx

            # x descent
            self.x.add_(step_dir_x, alpha=-self.lr_x)

            # accumulate for epoch averaging
            self.model_acc_x.add_(self.x)

        # Y update (ascent) with clipped gradient only (toy simplification)
        with torch.no_grad():
            dpy = torch.clamp(d_fy, -cv, cv)
            # y ascent to reflect max_y in toy problem
            self.y.add_(dpy, alpha=+self.lr_y)

        # advance counters
        self.T += 1
        self.steps += 1

        # apply scheduled regularizer updates and lr decay
        self._maybe_update_regularizer()

        return loss.detach()

# -------------------------
# Toy Adam optimizer for 2 variables
# -------------------------
class Adam2Var:
    def __init__(
        self,
        x,
        y,
        lr_x=1e-3,
        lr_y=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        maximize_y=True,
    ):
        """Minimal wrapper around torch.optim.Adam for two scalar variables.

        We maintain separate Adam optimizers for x (descent) and y (ascent if maximize_y=True).
        This demonstrates how to leverage the built-in implementation without re-writing logic.
        """
        self.x = x
        self.y = y
        # Create parameter tensors (need to be leaf nodes with requires_grad=True)
        self.param_x = self.x
        self.param_y = self.y
        self.maximize_y = bool(maximize_y)

        # Separate Adam instances (so we can choose different lrs if desired)
        self.opt_x = torch.optim.Adam(
            [self.param_x], lr=lr_x, betas=betas, eps=eps, weight_decay=weight_decay
        )
        self.opt_y = torch.optim.Adam(
            [self.param_y], lr=lr_y, betas=betas, eps=eps, weight_decay=weight_decay
        )

    def step(self, loss_fn):
        # Zero grads
        self.opt_x.zero_grad()
        self.opt_y.zero_grad()

        # Forward + backward
        loss = loss_fn(self.param_x, self.param_y)
        # Compute grads for x and y separately (retain_graph not needed if we re-run)
        (g_x,) = torch.autograd.grad(loss, self.param_x, retain_graph=True)
        (g_y,) = torch.autograd.grad(loss, self.param_y)

        # Apply gradients to x (descent)
        self.param_x.grad = g_x
        self.opt_x.step()

        # For y ascent, flip sign of gradient if maximize_y
        if self.maximize_y:
            self.param_y.grad = -g_y  # Adam minimizes, so negate for ascent
        else:
            self.param_y.grad = g_y
        self.opt_y.step()

        return loss.detach()

# -------------------------
# Toy RMSProp optimizer (wrapper) for 2 variables
# -------------------------
class RMSProp2Var:
    def __init__(
        self,
        x,
        y,
        lr_x=1e-2,
        lr_y=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        centered=False,
        maximize_y=True,
    ):
        self.x = x
        self.y = y
        self.param_x = self.x
        self.param_y = self.y
        self.maximize_y = bool(maximize_y)

        self.opt_x = torch.optim.RMSprop(
            [self.param_x], lr=lr_x, alpha=alpha, eps=eps,
            weight_decay=weight_decay, momentum=momentum, centered=centered
        )
        self.opt_y = torch.optim.RMSprop(
            [self.param_y], lr=lr_y, alpha=alpha, eps=eps,
            weight_decay=weight_decay, momentum=momentum, centered=centered
        )

    def step(self, loss_fn):
        self.opt_x.zero_grad()
        self.opt_y.zero_grad()

        loss = loss_fn(self.param_x, self.param_y)
        (g_x,) = torch.autograd.grad(loss, self.param_x, retain_graph=True)
        (g_y,) = torch.autograd.grad(loss, self.param_y)

        self.param_x.grad = g_x
        self.opt_x.step()

        self.param_y.grad = -g_y if self.maximize_y else g_y
        self.opt_y.step()

        return loss.detach()

# -------------------------
# Toy Adagrad optimizer (wrapper) for 2 variables
# -------------------------
class AdaGrad2Var:
    def __init__(
        self,
        x,
        y,
        lr_x=5e-2,
        lr_y=5e-2,
        lr_decay=0.0,
        weight_decay=0.0,
        initial_accumulator_value=0.0,
        eps=1e-10,
        maximize_y=True,
    ):
        self.x = x
        self.y = y
        self.param_x = self.x
        self.param_y = self.y
        self.maximize_y = bool(maximize_y)

        self.opt_x = torch.optim.Adagrad(
            [self.param_x], lr=lr_x, lr_decay=lr_decay, weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value, eps=eps
        )
        self.opt_y = torch.optim.Adagrad(
            [self.param_y], lr=lr_y, lr_decay=lr_decay, weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value, eps=eps
        )

    def step(self, loss_fn):
        self.opt_x.zero_grad()
        self.opt_y.zero_grad()

        loss = loss_fn(self.param_x, self.param_y)
        (g_x,) = torch.autograd.grad(loss, self.param_x, retain_graph=True)
        (g_y,) = torch.autograd.grad(loss, self.param_y)

        self.param_x.grad = g_x
        self.opt_x.step()

        self.param_y.grad = -g_y if self.maximize_y else g_y
        self.opt_y.step()

        return loss.detach()
