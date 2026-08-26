import argparse
import os

import torch
from omegaconf import OmegaConf

from model.ResShift_model import ResShiftTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EMR-Diff under the shared UFGNet comparison protocol."
    )
    parser.add_argument(
        "--dataset",
        default="PaviaU",
        choices=["PaviaU", "Houston13", "Chikusei"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--test_frequency", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")

    root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root, "config", "5_step_EMRDiff.yaml")
    configs = OmegaConf.load(config_path)
    configs.data.dataset = args.dataset
    configs.train.device = args.device
    if args.epochs is not None:
        configs.train.epochs = args.epochs
    if args.test_frequency is not None:
        configs.train.test_frequency = args.test_frequency

    trainer = ResShiftTrainer(configs=configs)
    trainer.train(configs.train.epochs, configs.train.test_frequency)
