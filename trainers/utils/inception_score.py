import numpy as np
import torch
from scipy.stats import entropy
from torch import nn
from torch.nn import functional as F
from torchvision.models.inception import Inception_V3_Weights, inception_v3


def get_inception_score(
    imgs: torch.Tensor,
    inception_model=None,
    device=None,
    batch_size=32,
    resize=False,
    splits=1,
):
    """Modified to Tensor edition"""
    if imgs.dim() != 4:
        raise ValueError("Input imgs should be a 4D tensor (N, C, H, W)]")
    N = imgs.size(0)

    assert batch_size > 0
    assert N > batch_size

    if inception_model is None:
        inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
        inception_model.eval()

    up = None
    if resize:
        up = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=False).to(
            device
        )

    preds = np.zeros((N, 1000), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = imgs[i : i + batch_size].to(device, non_blocking=True)

            # If previous scale is [-1, 1], to convert to [0, 1] before feeding to Inception, enable the following line:
            # batch = (batch + 1) / 2

            if resize and up is not None:
                batch = up(batch)

            logits = inception_model(batch)
            probs = F.softmax(logits, dim=1)

            batch_size_i = probs.size(0)
            preds[i : i + batch_size_i] = probs.cpu().numpy()

    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits) : (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores), np.std(split_scores)
