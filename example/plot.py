import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from opts import (
    AdaFM2Var,
    MSGDA2Var,
    TiAda2Var,
    PESG2Var,
    Adam2Var,
    RMSProp2Var,
    AdaGrad2Var,
    Adam2VarStepRatio,
    RMSProp2VarStepRatio,
    AdaGrad2VarStepRatio,
)


# ======= Configurable constants =======
L = 2.0
START_X = 0.1
START_Y = 0.0
STEPS = 10000
GRID_RES = 400
X_LIM = (0, 25)
Y_LIM = (0, 50)
ARROW_COUNT = 3


def f(x, y):
    return -0.5 * y * y + L * x * y - 0.5 * (L**2) * x * x


# ------------------------
# Reusable simulation utilities
# ------------------------
def run_trajectory(ctor, steps=STEPS, x0=START_X, y0=START_Y, loss_fn=f):
    """
    Run one optimizer trajectory given a constructor `ctor(x, y) -> optimizer`
    where optimizer exposes `.step(loss_fn)` and attributes `.x`, `.y`.
    Returns an (N, 2) numpy array of (x, y).
    """
    x = torch.tensor(float(x0), requires_grad=True)
    y = torch.tensor(float(y0), requires_grad=True)
    opt = ctor(x, y)

    traj = [(x.item(), y.item())]
    for _ in range(int(steps)):
        opt.step(loss_fn)
        traj.append((opt.x.item(), opt.y.item()))
    return np.array(traj)


def annotate_arrows(ax, traj, count=ARROW_COUNT, lw=2.0):
    if len(traj) < 5:
        return
    N = len(traj)
    for k in np.linspace(5, N - 3, int(count)).astype(int):
        x0, y0 = traj[k]
        x1, y1 = traj[k + 1]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=lw))


def draw_contours(ax, levels=100):
    X, Y = np.meshgrid(
        np.linspace(X_LIM[0], X_LIM[1], GRID_RES),
        np.linspace(Y_LIM[0], Y_LIM[1], GRID_RES),
    )
    Z = -0.5 * Y**2 + L * X * Y - 0.5 * (L**2) * X**2
    cf = ax.contourf(X, Y, Z, levels=levels)
    return cf


def plot_stationary_line(ax, lw=2.5):
    x_line = np.linspace(X_LIM[0], X_LIM[1], 200)
    ax.plot(x_line, L * x_line, "--", linewidth=lw, label="stationary points")


def simulate_and_plot(optimizers):
    """
    optimizers: list of dicts, each with keys:
      - name: legend label
      - ctor: callable (x,y) -> optimizer instance
      - style: dict passed to ax.plot (optional)
    """
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    cf = draw_contours(ax)
    fig.colorbar(cf, ax=ax)

    for spec in optimizers:
        name = spec.get("name", "opt")
        ctor = spec["ctor"]
        style = spec.get("style", {"linewidth": 2.5})
        traj = run_trajectory(ctor)
        ax.plot(traj[:, 0], traj[:, 1], "-", label=name, **style)
        annotate_arrows(ax, traj)

    plot_stationary_line(ax)

    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("(a) trajectory")
    ax.legend(loc="upper left", framealpha=0.85)
    plt.tight_layout()

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    save_path = os.path.join(base_dir, "plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Define optimizers once; simulation logic is fully reusable
    optim_specs = [
        {
            "name": "AdaFM",
            "ctor": lambda x, y: AdaFM2Var(x, y, lr=0.8, beta=0.9),
        },
        {
            "name": "MSGDA",
            "ctor": lambda x, y: MSGDA2Var(x, y, lr=47e-5, beta=0.9),
        },
        {
            "name": "TiAda",
            "ctor": lambda x, y: TiAda2Var(x, y, lr=2.5, alpha=0.5),
        },
        {
            "name": "PESG",
            "ctor": lambda x, y: PESG2Var(x, y, lr=0.045, clip_value=0.1, epoch_decay=2e-3, momentum=0.0, total_iter=STEPS),
        },
        {
            "name": "Adam",
            "ctor": lambda x, y: Adam2Var(x, y, lr_x=1e-3, lr_y=1e-3, betas=(0.9,0.999), maximize_y=True),
        },
        {
            "name": "RMSProp",
            "ctor": lambda x, y: RMSProp2Var(x, y, lr_x=1e-2, lr_y=1e-2, alpha=0.99, maximize_y=True),
        },
        {
            "name": "Adagrad",
            "ctor": lambda x, y: AdaGrad2Var(x, y, lr_x=5e-2, lr_y=5e-2, maximize_y=True),
        },
    ]

    simulate_and_plot(optim_specs)
