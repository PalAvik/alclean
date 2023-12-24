import wget
import logging
import tarfile
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from torchvision.datasets import ImageFolder
import PIL.Image
import numpy as np
from tqdm import tqdm

from AL_cleaning.datasets.imagenetdogs_utils import create_imagenetdogs_semantic_graph
from AL_cleaning.datasets.label_distribution import LabelDistribution
from AL_cleaning.evaluation.metrics import compute_label_entropy
from AL_cleaning.selection.simulation_statistics import SimulationStats, get_ambiguous_sample_ids
from AL_cleaning.utils.generics import convert_labels_to_one_hot


TOTAL_IMAGENETDOGS_DATASET_SIZE = 20580

class IMAGENETDOGS_BASE(ImageFolder):
    """
    Base class for Imagenet Dog Breeds dataset.
    """

    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 ) -> None:
        
        local_path_to_store_data = Path(root) / 'raw'

        if not local_path_to_store_data.exists():
            logging.info("Downloading ImagenetDogs dataset...")
            local_path_to_store_data.mkdir(parents=True, exist_ok=True)
            url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
            path_to_tar = local_path_to_store_data / 'stanford_dogs_images.tar'
            wget.download(url, str(path_to_tar))
            tf = tarfile.open(str(path_to_tar))
            tf.extractall(local_path_to_store_data / 'extracted')
            os.remove(path_to_tar)
        imagefolder_path = str(local_path_to_store_data / 'extracted' / 'Images')
        
        super().__init__(imagefolder_path, transform=transform)
        self.class_to_idx = {k.split('-')[0]:v for k,v in self.class_to_idx.items()}
        

class IMAGENETDOGS(IMAGENETDOGS_BASE):
    """
    Dataset class for Imagenet Dog Breeds dataset.
    """

    def __init__(self,
                 root: str,
                 train: bool,
                 noise_rate: float,
                 transform: Optional[Callable] = None,
                 noise_temperature: float = 1.0,
                 seed: int = 1,
                 ) -> None:
        """
        TODO: Add detailed param description.
        """
        super().__init__(root, transform=transform)
        self.seed = seed
        self.clean_targets = np.array(self.targets, dtype=np.int64)
        self.breeds_synset = [n.split('-')[0] for n in self.classes]

        imagenetdogs_labels, indices = self.load_imagenetdogs_label_counts(root, noise_rate, train)
        
        self.indices = indices
        self.num_samples = indices.shape[0]
        self.num_classes = imagenetdogs_labels.shape[1]
        self.label_counts = imagenetdogs_labels[self.indices]
        self.true_label_entropy = compute_label_entropy(label_counts=self.label_counts)

        self.label_distribution = LabelDistribution(seed, self.label_counts, noise_temperature)
        self.targets = self.label_distribution.sample_initial_labels_for_all()
        
        # Check the class distribution
        _, class_counts = np.unique(self.targets, return_counts=True)
        class_distribution = np.array([_c/self.num_samples for _c in class_counts])
        logging.info(f"Preparing dataset: IMAGENETDOGS (N={self.num_samples})")
        logging.info(f"Class distribution (%) (true labels): {class_distribution * 100.0}")
        self.clean_targets = self.clean_targets[self.indices]

        # Identify true ambiguous and clear label noise cases
        self._identify_sample_types()

    def _identify_sample_types(self) -> None:
        """
        Stores and logs clear label noise and ambiguous case types.
        """
        label_stats = SimulationStats(name="imagenetdogs", true_label_counts=self.label_counts,
                                      initial_labels=convert_labels_to_one_hot(self.targets, self.num_classes))
        self.ambiguous_mislabelled_cases = label_stats.mislabelled_ambiguous_sample_ids[0]
        self.clear_mislabeled_cases = label_stats.mislabelled_not_ambiguous_sample_ids[0]
        self.ambiguity_metric_args = {"ambiguous_mislabelled_ids": self.ambiguous_mislabelled_cases,
                                      "clear_mislabelled_ids": self.clear_mislabeled_cases,
                                      "true_label_entropy": self.true_label_entropy}

        # Log dataset details
        logging.info(f"Ambiguous mislabeled cases: {100 * len(self.ambiguous_mislabelled_cases) / self.num_samples}%")
        logging.info(f"Clear mislabeled cases: {100 * len(self.clear_mislabeled_cases) / self.num_samples}%\n")

    def load_imagenetdogs_label_counts(self, root, noise_rate, train, seed=1):
        label_counts_path = Path(root) / 'simulated_label_counts.npy'
        train_indices_save_path = Path(root) / 'train_indices.npy'
        val_indices_save_path = Path(root) / 'val_indices.npy'
        _, sibling_labels, _ = create_imagenetdogs_semantic_graph(root, self.breeds_synset)
    
        if not label_counts_path.exists() or not train_indices_save_path.exists() or not val_indices_save_path.exists():
            logging.info("Simulating label counts for ImageNetDogs dataset...")
            random_state = np.random.RandomState(seed)
            label_counts = self.simulate_label_counts(sibling_labels, random_state, noise_rate)
            np.save(open(label_counts_path, 'wb'), label_counts)
            
            indices = random_state.permutation(label_counts.shape[0])
            train_split_amount = int(TOTAL_IMAGENETDOGS_DATASET_SIZE*0.7)   # FIXED TO 0.7 SPLIT RATIO
            train_indices = indices[:train_split_amount]
            val_indices = indices[train_split_amount:]

            logging.info(f"Saving generated label counts data to {root}")
            np.save(open(train_indices_save_path, 'wb'), train_indices)
            np.save(open(val_indices_save_path, 'wb'), val_indices)
        
        label_counts = np.load(open(label_counts_path, 'rb'))
        indices = np.load(open(train_indices_save_path, 'rb')) if train else np.load(open(val_indices_save_path, 'rb'))
        return label_counts, indices
    
    def simulate_label_counts(self, sibling_labels, random_state, noise_rate=0.2, annotators=50):
        idx_to_class = {v:k for k,v in self.class_to_idx.items()}
        noise_choice = [False, True]
        _p = [1-noise_rate, noise_rate]

        all_label_counts = []

        for i in tqdm(range(len(self.imgs))):

            _, label = self.imgs[i]
            l_counts = np.zeros(120)
            syn = idx_to_class[label]
            graph_siblings = sibling_labels[syn]

            for _ in range(annotators):
                add_noise = random_state.choice(noise_choice, 1, p=_p)
                if add_noise and len(graph_siblings) != 0:
                    _label = str(random_state.choice(graph_siblings, 1)[0])
                else:
                    _label = syn
                _idx = self.class_to_idx[_label]
                l_counts[_idx] += 1

            all_label_counts.append(l_counts)

        all_label_counts = np.array(all_label_counts, dtype=np.int64)

        return all_label_counts
    
    def __getitem__(self, index: int) -> Tuple[PIL.Image.Image, int]:
        """
        :param index: The index of the sample to be fetched
        :return: The image and label tensors
        """
        img_path, _ = self.samples[self.indices[index]]
        img = self.loader(img_path)
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(target)

    def __len__(self) -> int:
        """

        :return: The size of the dataset
        """
        return len(self.indices)
        
    def get_label_names(self) -> List[str]:
        label_names = [k for k,_ in self.class_to_idx.items()]
        return label_names
        