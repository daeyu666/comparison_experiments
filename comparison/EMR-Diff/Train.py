import argparse
import os

import torch
from omegaconf import OmegaConf

from model.ResShift_model import ResShiftTrainer


DEFAULT_VALIDATION_INTERVALS = {
    "PaviaU": 20,
    "Houston13": 10,
    "Chikusei": 5,
}


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
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=None,
        help=(
            "Epochs between validation runs. Defaults are dataset-aware: "
            "PaviaU=20, Houston13=10, Chikusei=5."
        ),
    )
    # Backward-compatible alias for old commands. New scripts/readmes use
    # --validation_interval because this is validation, not final testing.
    parser.add_argument("--test_frequency", type=int, default=None, help=argparse.SUPPRESS)
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

    if args.validation_interval is not None:
        validation_interval = int(args.validation_interval)
    elif args.test_frequency is not None:
        validation_interval = int(args.test_frequency)
    else:
        configured = configs.train.get("validation_interval_by_dataset", {})
        validation_interval = int(
            configured.get(
                args.dataset,
                DEFAULT_VALIDATION_INTERVALS[args.dataset],
            )
        )
    if validation_interval < 1:
        raise ValueError("validation_interval must be >= 1")
    configs.train.validation_interval = validation_interval

    if args.early_stop_patience is not None:
        configs.train.early_stop_patience = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        configs.train.early_stop_min_delta = args.early_stop_min_delta
    if args.early_stop_metric is not None:
        configs.train.early_stop_metric = args.early_stop_metric
    if args.eval_seed is not None:
        configs.train.eval_seed = args.eval_seed

    print(
        f"[request] dataset={args.dataset}, degradation_mode={args.degradation_mode}"
    )
    print(
        f"Validation interval for {args.dataset}: "
        f"every {validation_interval} epoch(s)"
    )

    trainer = ResShiftTrainer(configs=configs)
    if trainer.degradation_mode != args.degradation_mode:
        raise RuntimeError(
            "Degradation-mode mismatch before training: "
            f"requested={args.degradation_mode}, resolved={trainer.degradation_mode}. "
            "Training is aborted to prevent checkpoints from being written to the wrong protocol directory."
        )
    print(
        f"[resolved] degradation_mode={trainer.degradation_mode}, "
        f"checkpoint_dir={os.path.join(root, 'checkpoints', trainer.degradation_mode, trainer.dataset)}"
    )

    trainer.train(configs.train.epochs, validation_interval)
