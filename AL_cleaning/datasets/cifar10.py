import logging
from typing import Callable, Optional

import numpy as np
import torchvision

from AL_cleaning.datasets.label_distribution import LabelDistribution


TOTAL_CIFAR10H_DATASET_SIZE = 10000

class CIFAR10Original(torchvision.datasets.CIFAR10):
    """
    Dataset class for the CIFAR10 dataset.
    """

    def __init__(self,
                 root: str,
                 train: bool,
                 transform: Optional[Callable] = None,
                 seed: int = 1234,
                 ) -> None:
        """
        :param root: The directory in which the CIFAR10 images will be stored
        :param transform: Transform to apply to the images
        :param seed: The random seed that defines which samples are train/test and which labels are sampled
        """
        super().__init__(root, train=train, transform=transform, target_transform=None, download=True)

        self.seed = seed
        self.targets = np.array(self.targets, dtype=np.int64)  # type: ignore
        self.num_classes = np.unique(self.targets, return_counts=False).size
        self.num_samples = len(self.data)
        self.label_counts = np.eye(self.num_classes, dtype=np.int64)[self.targets]
        self.clean_targets = np.argmax(self.label_counts, axis=1)
        self.indices = np.array(range(self.num_samples))
        logging.info(f"Preparing dataset: CIFAR10-Original (N={self.num_samples})")

        # Create label distribution for simulation of label adjudication
        self.label_distribution = LabelDistribution(seed, self.label_counts, temperature=1.0)

