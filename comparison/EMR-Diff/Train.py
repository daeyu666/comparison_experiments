import argparse
import os

import torch
from omegaconf import OmegaConf

from model.ResShift_model import ResShiftTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EMR-Diff under the shared comparison protocol."
    )
    parser.add_argument(
        "--dataset",
        default="PaviaU",
        choices=["PaviaU", "Houston13", "Chikusei"],
    )
    parser.add_argument(
        "--degradation_mode",
        default="gaussian_bicubic",
        choices=["gaussian_bicubic", "physical"],
        help="Switch LR-HSI observation between ordinary and physical degradation.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--test_frequency", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=None)
    parser.add_argument("--early_stop_metric", type=str, default=None)
    parser.add_argument("--eval_seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")

    root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root, "config", "5_step_EMRDiff.yaml")
    configs = OmegaConf.load(config_path)
    configs.data.dataset = args.dataset
    configs.data.degradation_mode = args.degradation_mode
    configs.train.device = args.device

    if args.epochs is not None:
        configs.train.epochs = args.epochs
    if args.test_frequency is not None:
        configs.train.test_frequency = args.test_frequency
    if args.early_stop_patience is not None:
        configs.train.early_stop_patience = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        configs.train.early_stop_min_delta = args.early_stop_min_delta
    if args.early_stop_metric is not None:
        configs.train.early_stop_metric = args.early_stop_metric
    if args.eval_seed is not None:
        configs.train.eval_seed = args.eval_seed

    trainer = ResShiftTrainer(configs=configs)
    trainer.train(configs.train.epochs, configs.train.test_frequency)
