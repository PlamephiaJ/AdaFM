import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from train import run
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_grid():
    # 参数范围（对数空间）
    lower_exp = -5   # 10^-5
    upper_exp = -3   # 10^-3
    num_points = 8   # 每轴8个点 → 8×8 = 64个组合

    # 在 log10 空间均匀采样
    x_vals = np.logspace(lower_exp, upper_exp, num_points)
    y_vals = np.logspace(lower_exp, upper_exp, num_points)

    # 生成笛卡尔积网格
    X, Y = np.meshgrid(x_vals, y_vals)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

    return grid_points



@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    logger.info("Starting training...")
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    grid_points = generate_grid()
    if cfg.worker_id == 0:
        grid_points = grid_points[:len(grid_points)//2]
    else:
        grid_points = grid_points[len(grid_points)//2:]
    for lr_x, lr_y in grid_points:
        logger.info(f"Running with lr_x: {lr_x}, lr_y: {lr_y}")
        cfg.optimizers.lr_discriminator = float(lr_x)
        cfg.optimizers.lr_generator = float(lr_y)
        run(cfg)
        logger.info("Training completed.")


if __name__ == "__main__":
    main()
