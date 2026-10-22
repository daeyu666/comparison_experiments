import argparse
import os

import torch
from omegaconf import OmegaConf

from model.ResShift_model import ResShiftTrainer
from emr_configuration import add_hidden_channels_argument, checkpoint_width, validate_hidden_channels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test EMR-Diff under the shared comparison protocol."
    )
    parser.add_argument(
        "--dataset",
        default="PaviaU",
        choices=["PaviaU", "Houston13", "Chikusei", "CAVE", "Botswana", "Augsburg"],
    )
    parser.add_argument(
        "--degradation_mode",
        default="physical",
        choices=["gaussian_bicubic", "physical"],
        help="Must match the degradation mode used to train the checkpoint.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default=None)
    add_hidden_channels_argument(parser)
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
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        configs.train.hidden_channels = checkpoint_width(state, args.hidden_channels)
    else:
        configs.train.hidden_channels = validate_hidden_channels(args.hidden_channels)

    trainer = ResShiftTrainer(configs=configs)
    trainer.test(checkpoint_path=args.checkpoint)
