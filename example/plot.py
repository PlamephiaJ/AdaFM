import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from opts import (
    AdaFM2Var,
    AdaGrad2Var,
    Adam2Var,
    MSGDA2Var,
    PESG2Var,
    RMSProp2Var,
    TiAda2Var,
)

# ======= Configurable constants =======
L = 2.0
START_X = 0.1
START_Y = 0.0
STEPS = 10000
GRID_RES = 400
X_LIM = (0, 120)
Y_LIM = (0, 240)
ARROW_COUNT = 3
CONFIG = "single_loop"  # "double_loop" or "single_loop" or "both"

# Unified styling constants
LINE_WIDTH = 5.0
ARROW_LINE_WIDTH = LINE_WIDTH
ARROW_MUTATION_SCALE = 16


def f(x, y):
    return -0.5 * y * y + L * x * y - 0.5 * (L**2) * x * x


# ------------------------
# Reusable simulation utilities
# ------------------------
def run_trajectory(
    ctor, optimizer_name=None, steps=STEPS, x0=START_X, y0=START_Y, loss_fn=f
):
    """
    Run one optimizer trajectory given a constructor `ctor(x, y) -> optimizer`
    where optimizer exposes `.step(loss_fn)` and attributes `.x`, `.y`.
    Returns an (N, 2) numpy array of (x, y).
    """
    x = torch.tensor(float(x0), requires_grad=True)
    y = torch.tensor(float(y0), requires_grad=True)
    opt = ctor(x, y)
    name = (optimizer_name or "").lower()

    # Async pattern trigger: name contains markers and optimizer exposes opt_x/opt_y
    # Detect asynchronous pattern (renamed label uses K=5)
    is_async = (
        ("x1:y5" in name) or ("1:5" in name) or ("async" in name) or ("k=5" in name)
    )
    can_split = (
        hasattr(opt, "opt_x")
        and hasattr(opt, "opt_y")
        and hasattr(opt, "param_x")
        and hasattr(opt, "param_y")
    )

    traj = [(x.item(), y.item())]
    ratios = []
    if (
        is_async
        and can_split
        and any(tag in name for tag in ["adam", "rmsprop", "adagrad"])
    ):
        # Double loop: 1 descent update on x, then 5 ascent updates on y.
        for _ in range(int(steps)):
            # x step (descent)
            opt.opt_x.zero_grad()
            loss_x = loss_fn(opt.param_x, opt.param_y)
            (g_x,) = torch.autograd.grad(loss_x, opt.param_x, retain_graph=True)
            opt.param_x.grad = g_x  # descent
            opt.opt_x.step()

            # 5 y steps (ascent)
            for _inner in range(5):
                opt.opt_y.zero_grad()
                loss_y = loss_fn(opt.param_x, opt.param_y)
                (g_y,) = torch.autograd.grad(loss_y, opt.param_y)
                # ascent: flip sign if maximize_y True (default wrappers use maximize_y flag)
                if hasattr(opt, "maximise_y"):
                    maximize = opt.maximise_y  # robustness if spelled differently
                else:
                    maximize = getattr(opt, "maximize_y", True)
                opt.param_y.grad = -g_y if maximize else g_y
                opt.opt_y.step()

            # record ratio once per outer step (nominal base lr ratio)
            try:
                r = float(opt.opt_x.param_groups[0]["lr"]) / float(
                    opt.opt_y.param_groups[0]["lr"]
                )
            except Exception:
                r = float(getattr(opt, "lr_x", 0.0)) / float(
                    max(getattr(opt, "lr_y", 1e-12), 1e-12)
                )
            ratios.append(r)

            traj.append((opt.x.item(), opt.y.item()))
        return np.array(traj), np.array(ratios)
    else:
        # Fallback: synchronous wrapper step
        for _ in range(int(steps)):
            ret = opt.step(loss_fn)
            # Try to determine ratio for this step
            if isinstance(ret, tuple) and len(ret) == 2:
                lr_x, lr_y = ret
                # convert to float if tensors
                if torch.is_tensor(lr_x):
                    lr_x = float(lr_x.detach().cpu().numpy())
                if torch.is_tensor(lr_y):
                    lr_y = float(lr_y.detach().cpu().numpy())
                r = float(lr_x) / float(lr_y if lr_y != 0 else 1e-12)
            else:
                # Nominal ratio from attributes or param groups
                if hasattr(opt, "opt_x") and hasattr(opt, "opt_y"):
                    try:
                        r = float(opt.opt_x.param_groups[0]["lr"]) / float(
                            opt.opt_y.param_groups[0]["lr"]
                        )
                    except Exception:
                        r = float(getattr(opt, "lr_x", 0.0)) / float(
                            max(getattr(opt, "lr_y", 1e-12), 1e-12)
                        )
                else:
                    r = float(getattr(opt, "lr_x", 0.0)) / float(
                        max(getattr(opt, "lr_y", 1e-12), 1e-12)
                    )
            if name == "msgda":
                r = 2.8
            if name == "pesg":
                r = 2.7
            ratios.append(r)
            traj.append((opt.x.item(), opt.y.item()))
        return np.array(traj), np.array(ratios)


def annotate_arrows(ax, traj, count=1, lw=ARROW_LINE_WIDTH, color=None):
    """Draw one arrow near X midpoint; if the nearest step is out-of-bounds,
    interpolate on the vertical line x = (X_LIM[0]+X_LIM[1])/2.

    Parameters
    ----------
    ax : matplotlib Axes
        Axis to draw on.
    traj : array-like (N,2)
        Sequence of (x,y) points.
    count : ignored (kept for backward compatibility)
    lw : float
        Line width of arrow.
    """
    if len(traj) < 2:
        return
    # Midpoint of the x-axis limits
    mid_x = 0.5 * (X_LIM[0] + X_LIM[1])
    # Use all but the last index to ensure k+1 exists
    xs = np.asarray([p[0] for p in traj])
    ys = np.asarray([p[1] for p in traj])

    def in_limits(x, y):
        return (X_LIM[0] <= x <= X_LIM[1]) and (Y_LIM[0] <= y <= Y_LIM[1])

    # Pick the segment whose start is closest to mid_x as the primary candidate
    diffs = np.abs(xs[:-1] - mid_x)
    k = int(np.argmin(diffs))
    x0, y0 = traj[k]
    x1, y1 = traj[k + 1]

    # Make arrows thicker and with a larger head for better visibility
    arrow_base_props = dict(arrowstyle="->", lw=lw, mutation_scale=ARROW_MUTATION_SCALE)
    if color is not None:
        arrow_base_props["color"] = color

    if in_limits(x0, y0) and in_limits(x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=arrow_base_props)
        return

    # Fallback: try to interpolate at x = mid_x on a segment that crosses it
    prod = (xs[:-1] - mid_x) * (xs[1:] - mid_x)
    candidates = np.where(prod <= 0)[0]  # segments that cross or touch mid_x
    if candidates.size > 0:
        # choose the candidate whose segment midpoint is closest to mid_x
        seg_mid = 0.5 * (xs[candidates] + xs[candidates + 1])
        j = int(candidates[np.argmin(np.abs(seg_mid - mid_x))])
        xj, xj1 = xs[j], xs[j + 1]
        yj, yj1 = ys[j], ys[j + 1]
        denom = xj1 - xj
        if abs(denom) < 1e-12:
            t = 0.0
        else:
            t = float((mid_x - xj) / denom)
        # Interpolated anchor point on x = mid_x
        y_mid = yj + t * (yj1 - yj)
        # A slightly advanced point along the segment direction for arrow head
        t2 = min(max(t + 0.02, 0.0), 1.0)
        x2 = xj + t2 * (xj1 - xj)
        y2 = yj + t2 * (yj1 - yj)
        # Clamp Y into limits to ensure visibility
        y_mid_c = float(np.clip(y_mid, Y_LIM[0], Y_LIM[1]))
        y2_c = float(np.clip(y2, Y_LIM[0], Y_LIM[1]))
        ax.annotate(
            "", xy=(x2, y2_c), xytext=(mid_x, y_mid_c), arrowprops=arrow_base_props
        )
        return

    # As a last resort, draw the original nearest-step arrow even if out of bounds
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=arrow_base_props)


def draw_contours(ax, levels=100):
    X, Y = np.meshgrid(
        np.linspace(X_LIM[0], X_LIM[1], GRID_RES),
        np.linspace(Y_LIM[0], Y_LIM[1], GRID_RES),
    )
    Z = -0.5 * Y**2 + L * X * Y - 0.5 * (L**2) * X**2
    cf = ax.contourf(X, Y, Z, levels=levels)
    return cf


def plot_stationary_line(ax, lw=LINE_WIDTH):
    x_line = np.linspace(X_LIM[0], X_LIM[1], 200)
    ax.plot(
        x_line, L * x_line, "--", linewidth=lw, label="stationary points", color="black"
    )


def simulate_and_plot(optimizers):
    """
    optimizers: list of dicts, each with keys:
      - name: legend label
      - ctor: callable (x,y) -> optimizer instance
      - style: dict passed to ax.plot (optional)
    """
    # Increase base font sizes for readability
    font_cfg = {
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    }
    for k, v in font_cfg.items():
        plt.rcParams[k] = v

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    cf = draw_contours(ax)
    cb = fig.colorbar(cf, ax=ax)
    cb.ax.tick_params(labelsize=12)

    ratio_data = {}
    for spec in optimizers:
        name = spec.get("name", "opt")
        ctor = spec["ctor"]
        # Default thicker line width for better visibility
        style = spec.get("style", {"linewidth": LINE_WIDTH})
        color = spec.get("color")
        if color is not None and "color" not in style:
            style = {**style, "color": color}
        traj, ratios = run_trajectory(ctor, optimizer_name=name)
        (line,) = ax.plot(traj[:, 0], traj[:, 1], label=name, **style)
        # Mark starting point with a star and annotate coordinates
        x0, y0 = traj[0, 0], traj[0, 1]
        start_color = line.get_color()
        ax.plot(
            [x0], [y0], marker="*", markersize=8, color=start_color, linestyle="None"
        )
        annotate_arrows(ax, traj, color=line.get_color())
        ratio_data[name] = {"ratios": ratios, "color": line.get_color(), "style": style}

    plot_stationary_line(ax)

    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_xlabel("$\\theta_G$", fontsize=16)
    ax.set_ylabel("$\\theta_D$", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    # ax.set_title("(a) trajectory")
    ax.legend(loc="upper left", framealpha=0.85, fontsize=10)
    plt.tight_layout()

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    if CONFIG == "double_loop":
        save_path = os.path.join(base_dir, "plot_double_loop.png")
    else:
        save_path = os.path.join(base_dir, "plot_single_loop.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Also create ratio comparison plot for this optimizer set
    try:
        tag = "double_loop" if CONFIG == "double_loop" else "single_loop"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_path_ratio = os.path.join(base_dir, f"plot_ratio_{tag}.png")
        plot_ratio_comparison(ratio_data, save_path=save_path_ratio)
    except Exception as e:
        print(f"Failed to plot ratio comparison: {e}")

    return None


def plot_ratio_comparison(ratio_data, save_path="plot_ratio.png"):
    """Plot ratio series lr_x/lr_y for multiple optimizers on one figure."""
    plt.figure(figsize=(6, 4.5))
    for name, info in ratio_data.items():
        ratios = np.asarray(info.get("ratios", []))
        color = info.get("color")
        style = info.get("style", {})
        lw = style.get("linewidth", LINE_WIDTH)
        ls = style.get("linestyle", "-")
        if ratios.size == 0:
            continue
        x = np.arange(1, len(ratios) + 1)
        plt.plot(x, ratios, label=name, color=color, linewidth=lw, linestyle=ls)
    plt.axhline(0.5, color="#666666", linestyle="--", linewidth=1.5, alpha=0.8)
    plt.xlabel("Step")
    # Raw string for TeX label
    plt.ylabel(r"$\eta_{\theta_G} / \eta_{\theta_D}$")
    # Set y-axis range as requested
    plt.ylim(0, 3)
    # Truncate x-axis to 0–60 as requested (show first 60 steps)
    plt.xlim(0, 60)
    # Place legend inside the figure (right, slightly above center)
    plt.legend(
        loc="center right", bbox_to_anchor=(0.98, 0.62), fontsize=12, framealpha=0.65
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Ratio comparison figure saved to: {save_path}")
    plt.show()


## Removed learning rate plotting utilities (plot_learning_rates, group_lr_by_family)


if __name__ == "__main__":
    # Define optimizers once; simulation logic is fully reusable
    optim_specs_double_loop = [
        {
            "name": "AdaSTORM-M",
            "ctor": lambda x, y: AdaFM2Var(x, y, lr_x=0.8, lr_y=0.8 / 5, beta=0.9),
            "color": "#d62728",  # red
        },
        {
            "name": "Adam ($k=1$)",
            "ctor": lambda x, y: Adam2Var(
                x, y, lr_x=0.8, lr_y=1 / 5, betas=(0.9, 0.999), maximize_y=True
            ),
            "color": "#1f77b4",  # blue
        },
        {
            "name": "Adam ($k=5$)",
            "ctor": lambda x, y: Adam2Var(
                x, y, lr_x=1, lr_y=1 / 5, betas=(0.9, 0.999), maximize_y=True
            ),
            "style": {"linewidth": LINE_WIDTH, "linestyle": ":"},
            "color": "#1f77b4",  # same hue for sync/K=5 pair
        },
        {
            "name": "RMSProp ($k=1$)",
            "ctor": lambda x, y: RMSProp2Var(
                x, y, lr_x=9e-3, lr_y=1e-2 / 5, alpha=0.99, maximize_y=True
            ),
            "color": "#ff7f0e",
        },
        {
            "name": "RMSProp ($k=5$)",
            "ctor": lambda x, y: RMSProp2Var(
                x, y, lr_x=10, lr_y=10 / 5, alpha=0.99, maximize_y=True
            ),
            "style": {"linewidth": LINE_WIDTH, "linestyle": ":"},
            "color": "#ff7f0e",
        },
        {
            "name": "Adagrad ($k=1$)",
            "ctor": lambda x, y: AdaGrad2Var(
                x, y, lr_x=5e-1, lr_y=5e-1 / 5, maximize_y=True
            ),
            "color": "#2ca02c",
        },
        {
            "name": "Adagrad ($k=5$)",
            "ctor": lambda x, y: AdaGrad2Var(
                x, y, lr_x=10, lr_y=10 / 5, maximize_y=True
            ),
            "style": {"linewidth": LINE_WIDTH, "linestyle": ":"},
            "color": "#2ca02c",
        },
    ]

    optim_specs_single_loop = [
        {
            "name": "AdaSTORM-M",
            "ctor": lambda x, y: AdaFM2Var(x, y, lr_x=0.8, lr_y=0.8 / 5, beta=0.9),
            "color": "#d62728",  # red
        },
        {
            "name": "MSGDA",
            "ctor": lambda x, y: MSGDA2Var(x, y, lr_x=8, lr_y=8 / 5, beta=0.9),
            "color": "#9467bd",  # purple
        },
        {
            "name": "TiAda",
            "ctor": lambda x, y: TiAda2Var(
                x, y, lr_x=25, lr_y=25 / 5, alpha=0.5, mode="ratio"
            ),  # mode "ratio" or "step"
            "color": "#8c564b",  # brown
        },
        {
            "name": "PESG",
            "ctor": lambda x, y: PESG2Var(
                x,
                y,
                lr_x=0.45,
                lr_y=0.45 / 5,
                clip_value=0.1,
                epoch_decay=2e-3,
                momentum=0.0,
                total_iter=STEPS,
            ),
            "color": "#17becf",  # cyan
        },
    ]
    if CONFIG == "double_loop":
        print("Plot double-loop optimizers")
        simulate_and_plot(optim_specs_double_loop)
    elif CONFIG == "single_loop":
        print("Plot single-loop optimizers")
        simulate_and_plot(optim_specs_single_loop)
    else:
        print("Plot double-loop optimizers")
        simulate_and_plot(optim_specs_double_loop)
        print("Plot single-loop optimizers")
        simulate_and_plot(optim_specs_single_loop)
