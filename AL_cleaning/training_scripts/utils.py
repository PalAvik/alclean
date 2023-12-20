import collections
import logging
from importlib import import_module
from pathlib import Path
from typing import Optional, Union

import torch
from yacs.config import CfgNode

from default_paths import PROJECT_ROOT_DIR
from AL_cleaning.configs.config_node import ConfigNode
from AL_cleaning.configs import model_config
from AL_cleaning.configs.selector_config import get_default_selector_config
from AL_cleaning.utils.generics import setup_cudnn, get_train_output_dir


def get_run_config(config: ConfigNode, run_seed: int) -> ConfigNode:
    config_run = config.clone()
    config_run.defrost()
    config_run.train.seed = run_seed
    config_run.train.output_dir = get_train_output_dir(config)
    config_run.freeze()
    return config_run


def load_model_config(config_path: Path) -> ConfigNode:
    '''
    Loads configs required for model training and inference.
    '''
    config = model_config.get_default_model_config()
    config.merge_from_file(config_path)
    update_model_config(config)
    setup_cudnn(config)

    # Freeze config entries
    config.freeze()

    return config


def update_model_config(config: ConfigNode) -> ConfigNode:
    '''
    Adds dataset specific parameters in model config
    '''
    if config.dataset.name in ['CIFAR10', 'CIFAR100']:
        dataset_dir = f'~/.torch/datasets/{config.dataset.name}'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 32
        config.dataset.n_channels = 3
        config.dataset.n_classes = int(config.dataset.name[5:])
    elif config.dataset.name in ['MNIST']:
        dataset_dir = '~/.torch/datasets'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 28
        config.dataset.n_channels = 1
        config.dataset.n_classes = 10

    if not torch.cuda.is_available():
        config.device = 'cpu'

    return config


def override_config(source: ConfigNode, overrides: ConfigNode) -> ConfigNode:
    '''
    Overrides the keys and values present in node `overrides` into source object recursively.
    '''
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = override_config(source.get(key, {}), value)  # type: ignore
            returned = CfgNode(returned)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


def load_selector_config(config_path: str) -> ConfigNode:
    """
    Loads a selector config and merges with its model config
    """
    selector_config = get_default_selector_config()
    selector_config.merge_from_file(config_path)
    model_config = load_model_config(PROJECT_ROOT_DIR / selector_config.selector.model_config_path)
    merged_config = override_config(source=model_config, overrides=selector_config)
    merged_config.freeze()

    return merged_config


def create_logger(output_dir: Union[str, Path]) -> None:
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    log_path = output_dir.absolute() / 'training.log'
    logging.basicConfig(filename=log_path,
                        filemode='w',
                        format='%(asctime)s %(name)-4s %(levelname)-6s %(message)s',
                        datefmt='%m-%d %H:%M',
                        level=logging.DEBUG)

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-4s: %(levelname)-6s %(message)s')

    # tell the handler to use this format
    console.setFormatter(formatter)

    # add the handler to the root logger
    logging.getLogger().addHandler(console)


def create_model(config: ConfigNode, model_id: Optional[int]) -> torch.nn.Module:
    device = torch.device(config.device)
    try:
        module = import_module('AL_cleaning.training_scripts.PyTorchImageClassification.models'f'.{config.model.type}.{config.model.name}')
    except ModuleNotFoundError:
        module = import_module(
            f'AL_cleaning.training_scripts.architectures.{config.model.type}.{config.model.name}')
    model = getattr(module, 'Network')(config)
    return model.to(device)
