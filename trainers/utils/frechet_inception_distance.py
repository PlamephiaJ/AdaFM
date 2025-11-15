import logging

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

LOGGER = logging.getLogger(__name__)


class FIDCalculator:

    def __init__(self, device: torch.device, dataloader: torch.utils.data.DataLoader):
        self.device = device
        self.fid = FrechetInceptionDistance(
            feature=2048, reset_real_features=False, normalize=True
        ).to(device)

        LOGGER.info("Calculating FID real features...")
        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            if real_images.min() < 0:
                real_images = (real_images + 1) / 2.0
            real_images = real_images.clamp(0, 1)
            self.fid.update(real_images, real=True)
        LOGGER.info("FID real features calculation completed.")

    def calculate(self, fake_images: torch.Tensor) -> float:
        fake_images = fake_images.to(self.device)
        if fake_images.min() < 0:
            fake_images = (fake_images + 1) / 2.0
        fake_images = fake_images.clamp(0, 1)

        self.fid.update(fake_images, real=False)

        fid_value = self.fid.compute().item()

        self.fid.reset()

        return fid_value
