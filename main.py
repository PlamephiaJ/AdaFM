import os

import hydra
from omegaconf import DictConfig, OmegaConf

from train import run

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run()


if __name__ == "__main__":
    main()