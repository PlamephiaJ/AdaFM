from omegaconf import DictConfig

import logging

LOGGER = logging.getLogger(__name__)


def delta_test_checker(cfg: DictConfig) -> bool:
    if cfg.experiment_type == "delta_test":
        LOGGER.info("Experiment type 'delta_test'")

        if cfg.optimizers.name != "adafm":
            LOGGER.error("Delta test only supports AdaFM optimizer.")
            return False
    return True


def experiment_setting_checker(cfg: DictConfig) -> bool:
    check_result = True
    check_result &= delta_test_checker(cfg)
    return check_result
