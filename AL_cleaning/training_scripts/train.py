import argparse
import logging
import os

from AL_cleaning.configs.config_node import ConfigNode
from AL_cleaning.training_scripts.utils import create_logger, get_run_config, load_model_config
from AL_cleaning.training_scripts.trainers.co_teaching_trainer import CoTeachingTrainer
from AL_cleaning.training_scripts.trainers.vanilla_trainer import VanillaTrainer
from AL_cleaning.utils.generics import set_seed
import wandb

# Start a wandb run with `sync_tensorboard=True`
wandb.init(project="AL_cleaning", sync_tensorboard=True)


def train(config: ConfigNode) -> None:
    create_logger(config.train.output_dir)
    logging.info('Starting training...')
    if config.train.use_co_teaching:
        model_trainer_class = CoTeachingTrainer
    else:
        model_trainer_class = VanillaTrainer  # type: ignore
    model_trainer_class(config).run_training()


def train_ensemble(config: ConfigNode, num_runs: int) -> None:
    for i, _ in enumerate(range(num_runs)):
        config_run = get_run_config(config, config.train.seed + i)
        set_seed(config_run.train.seed)
        os.makedirs(config_run.train.output_dir, exist_ok=True)
        train(config_run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for model training.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file characterising trained CNN model/s')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs (ensemble)')
    args, unknown_args = parser.parse_known_args()

    # Load config
    config = load_model_config(args.config)

    # Launch training
    train_ensemble(config, args.num_runs)
