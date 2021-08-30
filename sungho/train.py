import argparse
import pandas as pd
import numpy as np
import os
from functools import partial
from datetime import datetime

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
import torch

from utils import generate_csv

import config

from trainer import feature_train

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import random
import torch.backends.cudnn as cudnn


os.environ['WANDB_API_KEY'] = config.wandb_api_key

ray_config = {
    "batch_size": tune.choice([2, 4, 8, 16]),
    "loss": tune.choice(config.loss),
}

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train_worker(train_df, test_df):
    date = datetime.now().isoformat().replace(':', '-')
    model_dir = os.path.join(config.model_dir, date)
    os.makedirs(model_dir)

    if config.merge_feature:
        feature_train(train_df, test_df, 'Merged feature', config.model_name, model_dir)
    else:
        for feature in config.features:
            feature_train(train_df, test_df, feature, config.model_name, model_dir)


def main():
    train_df = pd.read_csv(config.with_system_path_csv)
    test_df = pd.read_csv(config.test_csv)

    if config.ray_tune:
        # set scheduler
        scheduler = ASHAScheduler(
            metric="f1_score",  # statics for selecting model
            mode="max",  # selecting method, what is good metric
            max_t=config.NUM_EPOCH,  # ray tune can't try more than {max_t} times
        )

        # set reporter
        reporter = CLIReporter(
            # parameter_columns=["l1", "l2", "lr", "batch_size"],
            metric_columns=[
                "training_iteration",
                "loss",
                "accuracy",
                "f1_scroe",
            ]
        )

        # run ray
        result = tune.run(
            partial(train_worker, train_df=train_df, test_df=test_df),
            resources_per_trial={"cpu": 4, "gpu": 1},
            config=ray_config,
            num_samples=10,
            scheduler=scheduler,
            progress_reporter=reporter,
        )
        best_trial = result.get_best_trial("loss", "min", "last")
    else:
        train_worker(train_df, test_df)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-g-path",
        dest="generate_path",
        action="store_true",
        default=False,
        required=False,
        help="Generate csv file with system path",
    )

    parser.add_argument(
        "-split-train",
        dest="split_train",
        action="store_true",
        default=True,
        required=False,
        help="Train with split features",
    )

    args = parser.parse_args()

    if args.generate_path:
        generate_csv(config.train_csv, config.train_dir, config.with_system_path_csv)
    if args.split_train:
        main()
        print('End Train!')
