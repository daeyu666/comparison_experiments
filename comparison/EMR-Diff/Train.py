import argparse
import os

import torch
from omegaconf import OmegaConf

from model.ResShift_model import ResShiftTrainer


DEFAULT_VALIDATION_INTERVALS = {
    "PaviaU": 20,
    "Houston13": 10,
    "Chikusei": 5,
    "CAVE": 20,
    "Botswana": 20,
    "Augsburg": 20,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EMR-Diff under the shared comparison protocol."
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
        help="Switch LR-HSI observation between ordinary and physical degradation.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--scale_ratio", type=int, default=4)
    parser.add_argument("--mtf_nyquist", type=float, default=0.2)
    parser.add_argument("--psf_truncate", type=float, default=3.0)
    parser.add_argument(
        "--train_misalignment_mode",
        default="registered",
        choices=["registered"],
        help="Stage-1 EMR-Diff is registered-only; deformation is stage 2.",
    )
    parser.add_argument("--optimizer", choices=["Adam", "AdamW"], default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--log_dir", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=None,
        help=(
            "Epochs between validation runs. Defaults are dataset-aware: "
            "PaviaU=20, Houston13=10, Chikusei=5."
        ),
    )
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
    configs.data.patch_size = args.patch_size
    configs.data.stride = args.stride
    configs.data.validation_size = args.image_size
    configs.data.test_size = args.image_size
    configs.data.mtf_nyquist = args.mtf_nyquist
    configs.data.psf_truncate = args.psf_truncate
    configs.diffusion.params.sf = args.scale_ratio
    configs.train.device = args.device
    configs.train.seed = args.seed
    configs.train.batch = [args.batch_size, 1]
    configs.train.num_workers = args.num_workers
    if args.optimizer is not None:
        configs.train.optimizer = args.optimizer
    if args.lr is not None:
        configs.train.lr = args.lr
    if args.weight_decay is not None:
        configs.train.weight_decay = args.weight_decay
    if args.checkpoint_dir:
        configs.train.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        configs.train.log_dir = args.log_dir
    if args.output_dir:
        configs.train.output_dir = args.output_dir

    if args.epochs is not None:
        configs.train.epochs = args.epochs

    if args.validation_interval is not None:
        validation_interval = int(args.validation_interval)
    elif args.test_frequency is not None:
        validation_interval = int(args.test_frequency)
    else:
        configured = configs.train.get("validation_interval_by_dataset", {})
        validation_interval = int(
            configured.get(args.dataset, DEFAULT_VALIDATION_INTERVALS[args.dataset])
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

    print(f"[request] dataset={args.dataset}, degradation_mode={args.degradation_mode}")
    print(
        f"Validation interval for {args.dataset}: "
        f"every {validation_interval} epoch(s)"
    )

    trainer = ResShiftTrainer(configs=configs)
    if trainer.dataset != args.dataset:
        raise RuntimeError(
            "Dataset mismatch before training: "
            f"requested={args.dataset}, resolved={trainer.dataset}."
        )
    if trainer.degradation_mode != args.degradation_mode:
        raise RuntimeError(
            "Degradation-mode mismatch before training: "
            f"requested={args.degradation_mode}, resolved={trainer.degradation_mode}."
        )

    checkpoint_dir = trainer.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    protocol_path = os.path.join(checkpoint_dir, "run_protocol.txt")
    with open(protocol_path, "w", encoding="utf-8") as f:
        f.write(f"requested_dataset: {args.dataset}\n")
        f.write(f"resolved_dataset: {trainer.dataset}\n")
        f.write(f"requested_degradation_mode: {args.degradation_mode}\n")
        f.write(f"resolved_degradation_mode: {trainer.degradation_mode}\n")
        f.write("training_stage: stage1_registered\n")
        f.write(f"train_misalignment_mode: {args.train_misalignment_mode}\n")
        f.write(f"image_size: {args.image_size}\n")
        f.write(f"patch_size: {args.patch_size}\n")
        f.write(f"stride: {args.stride}\n")
        f.write(f"scale_ratio: {args.scale_ratio}\n")
        f.write(f"mtf_nyquist: {args.mtf_nyquist}\n")
        f.write(f"psf_truncate: {args.psf_truncate}\n")
        f.write(f"optimizer: {trainer.optimizer_name}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"num_workers: {args.num_workers}\n")
        f.write(f"lr: {trainer.optimizer.param_groups[0]['lr']}\n")
        f.write(f"weight_decay: {trainer.weight_decay}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"validation_interval: {validation_interval}\n")
        f.write(f"early_stop_metric: {trainer.early_stop_metric}\n")
        f.write(f"early_stop_min_delta: {trainer.early_stop_min_delta}\n")
        f.write(f"early_stop_patience: {trainer.early_stop_patience}\n")
        f.write(f"eval_seed: {trainer.eval_seed}\n")

    print(
        f"[resolved] dataset={trainer.dataset}, "
        f"degradation_mode={trainer.degradation_mode}, "
        f"checkpoint_dir={checkpoint_dir}"
    )
    print(f"[protocol] {protocol_path}")

    trainer.train(configs.train.epochs, validation_interval)
