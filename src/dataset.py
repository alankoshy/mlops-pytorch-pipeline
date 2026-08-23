import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(
    data_dir="./data",
    batch_size=64,
    num_workers=4
):
    """ Sets up transforms, downloads datasets, and returns train and validation DataLoaders. """
    # Standard normalization values for CIFAR-10 (https://stackoverflow.com/questions/50710493/cifar-10-meaningless-normalization-values)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261) # https://github.com/kuangliu/pytorch-cifar/issues/19 https://github.com/gpleiss/efficient_densenet_pytorch/issues/68
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        ),
    ])

    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )

    val_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
