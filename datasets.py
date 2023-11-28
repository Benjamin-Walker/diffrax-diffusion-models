import abc
import dataclasses
import pathlib
from typing import Tuple, Union

import jax.numpy as jnp
import jax.random as jr
import torch
import torchvision


_data_dir = pathlib.Path(__file__).resolve().parent / ".." / "data"


class _DropLabel(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        feature, label = self.dataset[item]
        return feature

    def __len__(self):
        return len(self.dataset)


class _AbstractDataLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, dataset, *, key):
        pass

    def __iter__(self):
        raise RuntimeError("Use `.loop` to iterate over the data loader.")

    @abc.abstractmethod
    def loop(self, batch_size):
        pass


class _TorchDataLoader(_AbstractDataLoader):
    def __init__(self, dataset, *, key):
        self.dataset = dataset
        min = torch.iinfo(torch.int32).min
        max = torch.iinfo(torch.int32).max
        self.seed = jr.randint(key, (), min, max).item()

    def loop(self, batch_size):
        generator = torch.Generator().manual_seed(self.seed)
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=6,
            shuffle=True,
            drop_last=True,
            generator=generator,
        )
        while True:
            for tensor in dataloader:
                yield jnp.asarray(tensor)


class _InMemoryDataLoader(_AbstractDataLoader):
    def __init__(self, array, *, key):
        self.array = array
        self.key = key

    def loop(self, batch_size):
        dataset_size = self.array.shape[0]
        if batch_size > dataset_size:
            raise ValueError("Batch size larger than dataset size")

        key = self.key
        indices = jnp.arange(dataset_size)
        while True:
            key, subkey = jr.split(key)
            perm = jr.permutation(subkey, indices)
            start = 0
            end = batch_size
            while end < dataset_size:
                batch_perm = perm[start:end]
                yield self.array[batch_perm]
                start = end
                end = start + batch_size


@dataclasses.dataclass
class Dataset:
    train_dataloader: Union[_TorchDataLoader, _InMemoryDataLoader]
    test_dataloader: Union[_TorchDataLoader, _InMemoryDataLoader]
    data_shape: Tuple[int]
    mean: jnp.ndarray
    std: jnp.ndarray
    max: jnp.ndarray
    min: jnp.ndarray


def diamond(key):
    key0, key1, trainkey, testkey = jr.split(key, 4)

    WIDTH = 3
    BOUND = 0.5
    NOISE = 0.04
    DATASET_SIZE = 8192
    rotation_matrix = jnp.array([[1.0, -1.0], [1.0, 1.0]]) / jnp.sqrt(2.0)

    means = jnp.array(
        [
            (x, y)
            for x in jnp.linspace(-BOUND, BOUND, WIDTH)
            for y in jnp.linspace(-BOUND, BOUND, WIDTH)
        ]
    )
    means = means @ rotation_matrix
    covariance_factor = NOISE * jnp.eye(2)

    index = jr.choice(key0, WIDTH**2, shape=(DATASET_SIZE,), replace=True)
    noise = jr.normal(key1, (DATASET_SIZE, 2))
    data = means[index] + noise @ covariance_factor
    train_data = test_data = data

    mean = jnp.mean(train_data, axis=0)
    std = jnp.std(train_data, axis=0)
    max = jnp.inf
    min = -jnp.inf
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    data_shape = train_data.shape[1:]

    train_dataloader = _InMemoryDataLoader(train_data, key=trainkey)
    test_dataloader = _InMemoryDataLoader(test_data, key=testkey)
    return Dataset(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        data_shape=data_shape,
        mean=mean,
        std=std,
        max=max,
        min=min,
    )


def mnist(key):
    trainkey, testkey = jr.split(key)
    data_shape = (1, 28, 28)
    mean = 0.1307
    std = 0.3081
    max = 1
    min = 0

    train_dataset = torchvision.datasets.MNIST(
        _data_dir / "mnist", train=True, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        _data_dir / "mnist", train=False, download=True
    )

    # MNIST is small enough that the whole dataset can be placed in memory, so
    # we can actually use a faster method of data loading.

    # (We do need to handle normalisation ourselves though.)
    train_data = jnp.asarray(train_dataset.data[:, None]) / 255
    test_data = jnp.asarray(test_dataset.data[:, None]) / 255
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    train_dataloader = _InMemoryDataLoader(train_data, key=trainkey)
    test_dataloader = _InMemoryDataLoader(test_data, key=testkey)
    return Dataset(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        data_shape=data_shape,
        mean=mean,
        std=std,
        max=max,
        min=min,
    )


def cifar10(key):
    trainkey, testkey = jr.split(key)
    data_shape = (3, 32, 32)
    # mean = (0.4914, 0.4822, 0.4465)
    # std = (0.2023, 0.1994, 0.2010)
    # Scale data to be in range [-1, 1]
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    max = 1
    min = 0

    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
            torchvision.transforms.RandomHorizontalFlip(0.5),
        ]
    )
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        _data_dir / "cifar10", train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        _data_dir / "cifar10", train=False, download=True, transform=test_transform
    )

    train_dataset = _DropLabel(train_dataset)
    test_dataset = _DropLabel(test_dataset)
    train_dataloader = _TorchDataLoader(train_dataset, key=trainkey)
    test_dataloader = _TorchDataLoader(test_dataset, key=testkey)
    return Dataset(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        data_shape=data_shape,
        mean=jnp.array(mean)[:, None, None],
        std=jnp.array(std)[:, None, None],
        max=max,
        min=min,
    )
