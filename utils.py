import logging

import torch
from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


def delta_test_checker(cfg: DictConfig) -> bool:
    if cfg.experiment_type == "delta_test":
        LOGGER.info("Experiment type 'delta_test'")

        if cfg.optimizers.name != "adafm":
            LOGGER.error("Delta test only supports AdaFM optimizer.")
            return False
    return True


def FID_checker(cfg: DictConfig) -> bool:
    if cfg.models.evaluation.use_fid:
        LOGGER.info("FID evaluation is enabled.")
        g_memory = get_graphics_memory()
        if g_memory < 32.0:
            LOGGER.error(
                f"FID evaluation requires at least 32GB of GPU memory. Detected GPU memory: {g_memory:.2f} GB"
            )
            return False
    return True


def get_graphics_memory() -> float:
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (
            1024**3
        )  # in GB
        return total_memory
    return 0.0


def experiment_setting_checker(cfg: DictConfig) -> bool:
    check_result = True
    check_result &= delta_test_checker(cfg)
    check_result &= FID_checker(cfg)
    return check_result
