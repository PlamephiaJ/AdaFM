from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_fidelity
import torch.serialization as ts
from scipy import linalg
from torchvision.models.inception import Inception_V3_Weights, inception_v3

ts.add_safe_globals([np._core.multiarray._reconstruct])

INCEPTION_MEAN = (0.485, 0.456, 0.406)
INCEPTION_STD = (0.229, 0.224, 0.225)

class FidelityGeneratorWrapper(torch.nn.Module):
    def __init__(self, generator, z_dim):
        super().__init__()
        self.generator = generator
        self.z_dim = z_dim

    def forward(self, z):
        # z: [N, z_dim]
        # The generator expects input of shape [N, z_dim, 1, 1]
        z = z.view(z.size(0), self.z_dim, 1, 1)
        x = self.generator(z)

        # The output of the original generator is in the range [-1, 1] (Tanh).
        x = (x.clamp(-1, 1) + 1) * 127.5
        x = x.clamp(0, 255).to(torch.uint8)
        return x


def _get_inception_model(device: torch.device) -> torch.nn.Module:
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
    model.fc = nn.Identity()
    model.eval().to(device)
    return model


def _update_running_stats(mean: np.ndarray, m2: np.ndarray, n: int, batch: np.ndarray):
    if batch.size == 0:
        return mean, m2, n

    batch_n = batch.shape[0]
    batch_mean = batch.mean(axis=0)
    batch_centered = batch - batch_mean
    batch_m2 = batch_centered.T @ batch_centered

    if n == 0:
        return batch_mean, batch_m2, batch_n

    delta = batch_mean - mean
    total_n = n + batch_n
    new_mean = mean + delta * (batch_n / total_n)
    m2 = m2 + batch_m2 + np.outer(delta, delta) * (n * batch_n / total_n)
    return new_mean, m2, total_n


def _compute_generator_stats(
    generator: torch.nn.Module,
    z_dim: int,
    device: torch.device,
    num_samples: int,
    batch_size: int = 64,
    image_size: int = 299,
) -> tuple[np.ndarray, np.ndarray]:
    model = _get_inception_model(device)
    mean = None
    m2 = None
    n = 0

    remaining = num_samples
    with torch.no_grad():
        while remaining > 0:
            cur_bs = min(batch_size, remaining)
            z = torch.randn(cur_bs, z_dim, 1, 1, device=device)
            images = generator(z)
            images = (images.clamp(-1, 1) + 1) / 2.0
            images = F.interpolate(
                images, size=(image_size, image_size), mode="bilinear", align_corners=False
            )
            images = (images - torch.tensor(INCEPTION_MEAN, device=device).view(1, 3, 1, 1))
            images = images / torch.tensor(INCEPTION_STD, device=device).view(1, 3, 1, 1)
            feats = model(images).detach().cpu().numpy()

            if mean is None:
                mean = np.zeros(feats.shape[1], dtype=np.float64)
                m2 = np.zeros((feats.shape[1], feats.shape[1]), dtype=np.float64)
            mean, m2, n = _update_running_stats(mean, m2, n, feats)
            remaining -= cur_bs

    if n < 2:
        raise ValueError("Not enough samples to compute covariance.")

    cov = m2 / (n - 1)
    return mean, cov


def _calculate_fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def get_fid_score(
    generator: torch.nn.Module,
    z_dim: int,
    device: torch.device,
    num_samples: int = 50000,
    dataset_name: str = "cifar10",
    fid_stats_path: str | Path | None = None,
    fid_batch_size: int = 64,
) -> float:
    if dataset_name in {"cifar10", "cifar100"} and fid_stats_path is None:
        input2 = "cifar10-train" if dataset_name == "cifar10" else "cifar100-train"

        gen_for_fid = FidelityGeneratorWrapper(generator, z_dim).to(device)
        wrapped_generator = torch_fidelity.GenerativeModelModuleWrapper(
            gen_for_fid, z_dim, "normal", 0
        )

        with torch.no_grad():
            fid_score = torch_fidelity.calculate_metrics(
                input1=wrapped_generator,
                input1_model_num_samples=num_samples,
                input2=input2,
                cuda=True,
                fid=True,
                isc=False,
                kid=False,
                prc=False,
                verbose=False,
            )

        return fid_score["frechet_inception_distance"]

    if dataset_name != "imagenet":
        raise ValueError(
            f"Unsupported dataset for FID calculation: {dataset_name}. Provide fid_stats_path for ImageNet."
        )

    if fid_stats_path is None:
        raise ValueError("fid_stats_path is required for ImageNet FID calculation.")

    stats = np.load(str(fid_stats_path))
    mu_real = stats["mu"]
    sigma_real = stats["sigma"]

    mu_gen, sigma_gen = _compute_generator_stats(
        generator,
        z_dim,
        device,
        num_samples=num_samples,
        batch_size=fid_batch_size,
    )

    return _calculate_fid(mu_gen, sigma_gen, mu_real, sigma_real)