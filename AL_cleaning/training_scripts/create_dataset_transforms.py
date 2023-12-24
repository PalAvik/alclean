from typing import Any, Callable, List, Tuple

import numpy as np
import torchvision
from torchvision.transforms import ToTensor, Normalize

from AL_cleaning.configs.config_node import ConfigNode
from AL_cleaning.training_scripts.transforms import AddGaussianNoise, CenterCrop, ElasticTransform, \
    ExpandChannels, RandomAffine, RandomColorJitter, RandomErasing, RandomGamma, \
    RandomHorizontalFlip, RandomResizeCrop, Resize, RandomCrop


def _get_dataset_stats(
        config: ConfigNode) -> Tuple[np.ndarray, np.ndarray]:
    name = config.dataset.name
    if name == 'CIFAR10':
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
    else:
        raise ValueError()
    return mean, std


def create_transform(config: ConfigNode, is_train: bool) -> Callable:
    if config.dataset.name in ["IMAGENETDOGS"]:
        return create_imagenetdogs_transform(config, is_train)
    elif config.dataset.name in ["CIFAR10", "CIFAR10H", "CIFAR10IDN", "CIFAR10H_TRAIN_VAL", "CIFAR10SYM"]:
        return create_cifar_transform(config, is_train)
    else:
        raise ValueError


def create_cifar_transform(config: ConfigNode,
                           is_train: bool) -> Callable:
    transforms: List[Any] = list()
    if is_train:
        if config.augmentation.use_random_affine:
            transforms.append(RandomAffine(config))
        if config.augmentation.use_random_crop:
            transforms.append(RandomResizeCrop(config))
        if config.augmentation.use_random_horizontal_flip:
            transforms.append(RandomHorizontalFlip(config))
        if config.augmentation.use_random_color:
            transforms.append(RandomColorJitter(config))
    transforms += [ToTensor()]
    return torchvision.transforms.Compose(transforms)


def create_imagenetdogs_transform(config: ConfigNode,
                                  is_train: bool) -> Callable:
    imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transforms: List[Any] = list()
    transforms += [Resize(config)]
    if is_train:
        if config.augmentation.use_random_crop:
            transforms.append(RandomCrop(config))
        if config.augmentation.use_random_horizontal_flip:
            transforms.append(RandomHorizontalFlip(config))
        if config.augmentation.use_random_affine:
            transforms.append(RandomAffine(config))
    transforms += [ToTensor()]
    transforms += [Normalize(*imagenet_stats, inplace=True)]
    return torchvision.transforms.Compose(transforms)
