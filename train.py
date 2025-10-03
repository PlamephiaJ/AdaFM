import logging
from omegaconf import DictConfig

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> None:
    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Switching to CPU.")
        device = "cpu"
