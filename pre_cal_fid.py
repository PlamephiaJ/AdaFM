# fid_real_stats.py
import torch
import torch.nn as nn
import numpy as np
from torchvision.models.inception import inception_v3
from torch.nn.functional import adaptive_avg_pool2d


class InceptionFID(nn.Module):
    def __init__(self):
        super().__init__()
        net = inception_v3(pretrained=True, transform_input=False)
        net.fc = nn.Identity()
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        self.net = net

    @torch.no_grad()
    def forward(self, x):
        # x: [N,3,H,W], 范围 [0,1]
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = nn.functional.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        feat = self.net(x)
        if feat.ndim == 4:
            feat = adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
        return feat


@torch.no_grad()
def compute_real_stats(real_loader, device="cuda", max_samples=50000):
    model = InceptionFID().to(device)
    feats = []

    for imgs, *_ in real_loader:
        imgs = imgs.to(device).float()
        # 假设 dataloader 输出已是 [0,1]，否则请在这里归一化
        feats.append(model(imgs).cpu())
        if len(torch.cat(feats)) >= max_samples:
            break

    feats = torch.cat(feats).numpy().astype(np.float64)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)

    return mu, sigma


if __name__ == "__main__":
    # 伪代码示意：
    # from your_dataset import real_loader
    device = "cuda"
    mu_real, sigma_real = compute_real_stats(real_loader, device=device)
    np.savez("fid_real_stats.npz", mu=mu_real, sigma=sigma_real)
    print("Saved real stats to fid_real_stats.npz")
