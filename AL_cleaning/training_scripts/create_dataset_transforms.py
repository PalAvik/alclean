#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, Callable, List, Tuple

import numpy as np
import torchvision
from torchvision.transforms import ToTensor, Normalize

from AL_cleaning.configs.config_node import ConfigNode
from AL_cleaning.training_scripts.transforms import RandomAffine, RandomHorizontalFlip, Resize, RandomCrop


def create_transform(config: ConfigNode, is_train: bool) -> Callable:
    if config.dataset.name in ["IMAGENETDOGS", "CUB"]:
        return create_imagenetdogs_transform(config, is_train)
    else:
        raise ValueError


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
