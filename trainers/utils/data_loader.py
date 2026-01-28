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
    image_size = getattr(args, "image_size", 32)
    trans = transforms.Compose(
        [
            transforms.Resize(image_size),
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
    image_size = getattr(args, "image_size", 32)
    trans = transforms.Compose(
        [
            transforms.Resize(image_size),
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

@DatasetRegistry.register("imagenet")
def build_imagenet(dataroot, args):
    image_size = getattr(args, "image_size", 128)
    trans = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    from datasets import load_dataset

    hf_dataset = getattr(args, "hf_dataset", "ILSVRC/imagenet-1k")
    hf_train_split = getattr(args, "hf_train_split", "train")
    hf_val_split = getattr(args, "hf_val_split", "validation")
    hf_cache_dir = getattr(args, "hf_cache_dir", None)

    cache_dir = hf_cache_dir if hf_cache_dir else (dataroot if dataroot else None)
    train_hf = load_dataset(hf_dataset, split=hf_train_split, cache_dir=cache_dir)
    val_hf = load_dataset(hf_dataset, split=hf_val_split, cache_dir=cache_dir)

    class HFDataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds, transform):
            self.hf_ds = hf_ds
            self.transform = transform

        def __len__(self):
            return len(self.hf_ds)

        def __getitem__(self, idx):
            item = self.hf_ds[idx]
            image = item["image"].convert("RGB")
            label = int(item["label"]) if "label" in item else 0
            if self.transform is not None:
                image = self.transform(image)
            return image, label

    return HFDataset(train_hf, trans), HFDataset(val_hf, trans)


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

    train_sampler = None
    test_sampler = None

    # Support for using a subset of the dataset
    if args.dataset.use_ratio != 1.0:
        from torch.utils.data import SubsetRandomSampler
        use_ratio = args.dataset.use_ratio
        num_train = len(train_dataset)
        indices = np.random.choice(num_train, int(num_train * use_ratio), replace=False)
        train_sampler = SubsetRandomSampler(indices)

        num_test = len(test_dataset)
        indices = np.random.choice(num_test, int(num_test * use_ratio), replace=False)  
        test_sampler = SubsetRandomSampler(indices)

        train_dataloader = data_utils.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            sampler=train_sampler,
            worker_init_fn=worker_init_fn,
        )
        test_dataloader = data_utils.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            sampler=test_sampler,
            worker_init_fn=worker_init_fn,
        )
        return train_dataloader, test_dataloader

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
