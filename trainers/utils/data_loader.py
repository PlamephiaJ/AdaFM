import os
import random

import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DatasetRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(fn):
            if name in cls._registry:
                raise ValueError(f"Dataset '{name}' is already registered.")
            cls._registry[name] = fn
            return fn

        return decorator

    @classmethod
    def build(cls, name, dataroot, args):
        if name not in cls._registry:
            raise KeyError(
                f"Unknown dataset '{name}'. " f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](dataroot, args)


# @DatasetRegistry.register("mnist")
# def build_mnist(dataroot, args):
#     trans = transforms.Compose(
#         [
#             transforms.Resize(32),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,)),
#         ]
#     )
#     train_dataset = MNIST(
#         root=dataroot, train=True, download=args.download, transform=trans
#     )
#     test_dataset = MNIST(
#         root=dataroot, train=False, download=args.download, transform=trans
#     )
#     return train_dataset, test_dataset


# @DatasetRegistry.register("fashion-mnist")
# def build_fashion_mnist(dataroot, args):
#     trans = transforms.Compose(
#         [
#             transforms.Resize(32),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,)),
#         ]
#     )
#     train_dataset = FashionMNIST(
#         root=dataroot, train=True, download=args.download, transform=trans
#     )
#     test_dataset = FashionMNIST(
#         root=dataroot, train=False, download=args.download, transform=trans
#     )
#     return train_dataset, test_dataset


@DatasetRegistry.register("cifar10")
def build_cifar10(dataroot, args):
    trans = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = dset.CIFAR10(
        root=dataroot, train=True, download=args.download, transform=trans
    )
    test_dataset = dset.CIFAR10(
        root=dataroot, train=False, download=args.download, transform=trans
    )
    return train_dataset, test_dataset


@DatasetRegistry.register("cifar100")
def build_cifar100(dataroot, args):
    trans = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = dset.CIFAR100(
        root=dataroot, train=True, download=args.download, transform=trans
    )
    test_dataset = dset.CIFAR100(
        root=dataroot, train=False, download=args.download, transform=trans
    )
    return train_dataset, test_dataset


# @DatasetRegistry.register("stl10")
# def build_stl10(dataroot, args):
#     trans = transforms.Compose(
#         [
#             transforms.Resize(32),
#             transforms.ToTensor(),
#         ]
#     )
#     train_dataset = dset.STL10(
#         root=dataroot, split="train", download=args.download, transform=trans
#     )
#     test_dataset = dset.STL10(
#         root=dataroot, split="test", download=args.download, transform=trans
#     )
#     return train_dataset, test_dataset


def get_data_loader(args):
    dataroot = os.path.join(args.dataroot, args.dataset)

    train_dataset, test_dataset = DatasetRegistry.build(args.dataset, dataroot, args)

    train_dataloader = data_utils.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn,
    )
    test_dataloader = data_utils.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=worker_init_fn,
    )

    return train_dataloader, test_dataloader
