import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from train import run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    logger.info("Starting training...")
    logger.info(OmegaConf.to_yaml(cfg))
    run(cfg)
    logger.info("Training completed.")


if __name__ == "__main__":
    main()
