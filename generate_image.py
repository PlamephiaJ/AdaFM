import os
import datetime
import torch
import random
import sys
import numpy as np
from pathlib import Path
import torchvision

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from models.wgan_factory import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GENERATOR_CKPT_PATH = Path("generator_iter_157000.pth")
DEVICE = torch.device("cuda:0")
SEED = 8
OUTPUT_DIR = Path("generated_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_BATCH_SIZE = 128  # DO NOT CHANGE THIS VALUE


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        logger.warning(
            "Deterministic algorithms not supported on this PyTorch version/hardware."
        )


def load_generator_ckpt(cfg: DictConfig, ckpt_path: Path) -> torch.nn.Module:
    generator = create_model(
        cfg.models.backbone.name, cfg.models.backbone.generator
    ).to(DEVICE)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
    generator.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return generator


def generate_image(
    generator: torch.nn.Module, noise_dim: int, device: torch.device
) -> torch.Tensor:
    noise = torch.randn(FIXED_BATCH_SIZE, noise_dim, 1, 1, device=device)
    with torch.no_grad():
        generated_image = generator(noise)
    return generated_image


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    # seed_everything(SEED)
    generator = load_generator_ckpt(cfg, GENERATOR_CKPT_PATH)
    for i in range(10):
        now = datetime.datetime.now()
        img_tensor = generate_image(
            generator, cfg.models.backbone.generator.in_dim, DEVICE
        )
        img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2.0  # Rescale to [0, 1]
        img_path = OUTPUT_DIR / f"generated_{now.strftime('%Y-%m-%d-%H-%M-%S')}_{i}.png"
        print(f"Generated image tensor shape: {img_tensor.shape}")
        torchvision.utils.save_image(img_tensor, img_path)
        logger.info(f"Saved generated image to {img_path}")


if __name__ == "__main__":
    main()
