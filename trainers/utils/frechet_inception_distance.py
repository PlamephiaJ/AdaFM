import torch
import numpy as np
from cleanfid import fid

def tensor_to_uint8(imgs):
    """
    imgs: tensor (N,3,H,W) in [-1,1] or [0,1]
    """

    if imgs.min() < 0:  # [-1,1]
        imgs = (imgs + 1) / 2

    imgs = imgs.clamp(0, 1)
    imgs = (imgs * 255).byte()
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    return imgs

def get_fid_score(fake_imgs, dataset_name="cifar10", dataset_split="train"):
    """
    fake_imgs: tensor (N,3,H,W) in [-1,1] or [0,1]
    """
    fake_array = tensor_to_uint8(fake_imgs)
    score = fid.compute_fid_from_array(
        fake_array, dataset_name=dataset_name, dataset_split=dataset_split
    )
    return score