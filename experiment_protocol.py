"""Frozen two-stage comparison protocol and model adapter registry.

User-facing entry points are root-level train.py and test.py. They intentionally
expose only model/dataset/mode (plus device/dry-run) so method comparisons do
not drift through model-specific CLI knobs.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATASETS = (
    "PaviaU", "Houston13", "Chikusei", "CAVE", "Botswana", "Augsburg"
)
MODES = ("registered", "mixed")

# Frozen synthetic x4 comparison protocol.
IMAGE_SIZE = 128
PATCH_SIZE = 64
STRIDE = 32
SCALE_RATIO = 4
DEGRADATION_MODE = "physical"
MTF_NYQUIST = 0.2
PSF_TRUNCATE = 3.0

REGISTERED_EPOCHS = 800
MIXED_EPOCHS = 600
BATCH_SIZE = 1
LR = 1e-5
WEIGHT_DECAY = 5e-5
REGISTERED_SEED = 1001
MIXED_SEED = 10
EVAL_SEED = 1234

# Formal stage-2 mixture.
IDENTITY_PROBABILITY = 0.10
DEFORMED_PROBABILITY = 0.90
MAX_TRANSLATION = 4.0
MAX_ROTATION_DEG = 2.0
MAX_LOCAL_PX = 4.0
CONTROL_GRID = 5
MIN_JACOBIAN = 0.5

VALIDATION_CASES = 5
TEST_CASES = 10
MIXED_VALIDATION_INTERVAL = 20
EARLY_STOP_PATIENCE_FIXED_BUDGET = 999999
EARLY_STOP_MIN_DELTA = 0.02
REGISTERED_VALIDATION_INTERVAL = {
    "PaviaU": 20,
    "Houston13": 10,
    "Chikusei": 5,
    "CAVE": 20,
    "Botswana": 20,
    "Augsburg": 20,
}


@dataclass(frozen=True)
class ModelAdapter:
    canonical: str
    folder: str
    registered_train: str
    mixed_train: str
    test_script: str
    real_train: str
    real_infer: str


_ADAPTERS = {
    "uafl": ModelAdapter(
        canonical="UAFL",
        folder="UAFL",
        registered_train="comparison/UAFL/train.py",
        mixed_train="comparison/UAFL/train_hsi_deformed.py",
        test_script="comparison/UAFL/test_hsi_deformed.py",
        real_train="comparison/UAFL/train_augsburg2_wald.py",
        real_infer="comparison/UAFL/infer_augsburg2_wald.py",
    ),
    "emr-diff": ModelAdapter(
        canonical="EMR-Diff",
        folder="EMR-Diff",
        registered_train="comparison/EMR-Diff/Train.py",
        mixed_train="comparison/EMR-Diff/train_hsi_deformed.py",
        test_script="comparison/EMR-Diff/test_hsi_deformed.py",
        real_train="comparison/EMR-Diff/train_augsburg2_wald.py",
        real_infer="comparison/EMR-Diff/infer_augsburg2_wald.py",
    ),
}
_ALIASES = {
    "ua": "uafl",
    "uafl": "uafl",
    "emr": "emr-diff",
    "emrdiff": "emr-diff",
    "emr_diff": "emr-diff",
    "emr-diff": "emr-diff",
}


def resolve_model(name: str) -> ModelAdapter:
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _ADAPTERS:
        supported = ", ".join(adapter.canonical for adapter in _ADAPTERS.values())
        raise ValueError(
            f"Unsupported model={name!r}. Registered unified adapters: {supported}"
        )
    return _ADAPTERS[key]


def checkpoint_path(adapter: ModelAdapter, dataset: str, mode: str) -> Path:
    if mode not in MODES:
        raise ValueError(mode)
    family = "physical" if mode == "registered" else "hsi_warp_final"
    return ROOT / "comparison" / adapter.folder / "checkpoints" / family / dataset / "best.pth.tar"


def log_dir(adapter: ModelAdapter, dataset: str, mode: str) -> Path:
    family = "physical" if mode == "registered" else "hsi_warp_final"
    return ROOT / "comparison" / adapter.folder / "logs" / family / dataset


def checkpoint_dir(adapter: ModelAdapter, dataset: str, mode: str) -> Path:
    family = "physical" if mode == "registered" else "hsi_warp_final"
    return ROOT / "comparison" / adapter.folder / "checkpoints" / family / dataset


def output_dir(adapter: ModelAdapter, dataset: str, mode: str) -> Path:
    family = "physical" if mode == "registered" else "hsi_warp_final"
    return ROOT / "comparison" / adapter.folder / "outputs" / family / dataset


def _shared_geometry_args():
    return [
        "--max_translation", str(MAX_TRANSLATION),
        "--max_rotation_deg", str(MAX_ROTATION_DEG),
        "--max_local_px", str(MAX_LOCAL_PX),
        "--control_grid", str(CONTROL_GRID),
        "--min_jacobian", str(MIN_JACOBIAN),
    ]


def build_train_command(adapter: ModelAdapter, dataset: str, mode: str, device: str):
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset={dataset!r}")
    if mode not in MODES:
        raise ValueError(f"Unsupported mode={mode!r}")

    script = adapter.registered_train if mode == "registered" else adapter.mixed_train
    cmd = [sys.executable, str(ROOT / script)]

    if mode == "registered":
        common = [
            "--dataset", dataset,
            "--image_size", str(IMAGE_SIZE),
            "--patch_size", str(PATCH_SIZE),
            "--stride", str(STRIDE),
            "--scale_ratio", str(SCALE_RATIO),
            "--degradation_mode", DEGRADATION_MODE,
            "--mtf_nyquist", str(MTF_NYQUIST),
            "--psf_truncate", str(PSF_TRUNCATE),
            "--train_misalignment_mode", "registered",
            "--epochs", str(REGISTERED_EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--lr", str(LR),
            "--weight_decay", str(WEIGHT_DECAY),
            "--seed", str(REGISTERED_SEED),
            "--device", device,
            "--validation_interval", str(REGISTERED_VALIDATION_INTERVAL[dataset]),
            "--early_stop_patience", str(EARLY_STOP_PATIENCE_FIXED_BUDGET),
            "--early_stop_min_delta", str(EARLY_STOP_MIN_DELTA),
            "--checkpoint_dir", str(checkpoint_dir(adapter, dataset, mode).relative_to(ROOT)),
            "--log_dir", str(log_dir(adapter, dataset, mode).relative_to(ROOT)),
        ]
        if adapter.canonical == "EMR-Diff":
            common += [
                "--optimizer", "AdamW",
                "--eval_seed", str(EVAL_SEED),
                "--output_dir", str(output_dir(adapter, dataset, mode).relative_to(ROOT)),
            ]
        cmd += common
        return cmd

    stage1 = checkpoint_path(adapter, dataset, "registered")
    if not stage1.is_file():
        raise FileNotFoundError(
            f"Stage-1 best checkpoint is required before mixed training: {stage1}"
        )
    cmd += [
        "--dataset", dataset,
        "--image_size", str(IMAGE_SIZE),
        "--patch_size", str(PATCH_SIZE),
        "--stride", str(STRIDE),
        "--scale_ratio", str(SCALE_RATIO),
        "--degradation_mode", DEGRADATION_MODE,
        "--mtf_nyquist", str(MTF_NYQUIST),
        "--psf_truncate", str(PSF_TRUNCATE),
        *_shared_geometry_args(),
        "--registered_probability", str(IDENTITY_PROBABILITY),
        "--epochs", str(MIXED_EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--weight_decay", str(WEIGHT_DECAY),
        "--seed", str(MIXED_SEED),
        "--device", device,
        "--validation_interval", str(MIXED_VALIDATION_INTERVAL),
        "--validation_cases", str(VALIDATION_CASES),
        "--early_stop_patience", str(EARLY_STOP_PATIENCE_FIXED_BUDGET),
        "--early_stop_min_delta", str(EARLY_STOP_MIN_DELTA),
        "--checkpoint_dir", str(checkpoint_dir(adapter, dataset, mode).relative_to(ROOT)),
        "--log_dir", str(log_dir(adapter, dataset, mode).relative_to(ROOT)),
        "--init_checkpoint", str(stage1.relative_to(ROOT)),
    ]
    if adapter.canonical == "EMR-Diff":
        cmd += ["--eval_seed", str(EVAL_SEED)]
    return cmd


def build_test_command(adapter: ModelAdapter, dataset: str, mode: str, device: str):
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset={dataset!r}")
    checkpoint = checkpoint_path(adapter, dataset, mode)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output = (
        ROOT / "comparison" / adapter.folder / "outputs" / "unified_two_stage"
        / dataset / f"{adapter.canonical.replace('-', '')}_{mode}_seed{MIXED_SEED}.json"
    )
    cmd = [
        sys.executable,
        str(ROOT / adapter.test_script),
        "--dataset", dataset,
        "--image_size", str(IMAGE_SIZE),
        "--patch_size", str(PATCH_SIZE),
        "--stride", str(STRIDE),
        "--scale_ratio", str(SCALE_RATIO),
        "--degradation_mode", DEGRADATION_MODE,
        "--mtf_nyquist", str(MTF_NYQUIST),
        "--psf_truncate", str(PSF_TRUNCATE),
        *_shared_geometry_args(),
        "--seed", str(MIXED_SEED),
        "--cases", str(TEST_CASES),
        "--device", device,
        "--test_mode", (
            "registered_only" if mode == "registered" else "registered_and_warp"
        ),
        "--checkpoint", str(checkpoint.relative_to(ROOT)),
        "--output_json", str(output.relative_to(ROOT)),
    ]
    if adapter.canonical == "EMR-Diff":
        cmd += ["--eval_seed", str(EVAL_SEED)]
    return cmd



# Frozen real-world Augsburg-2 center-heldout Wald x3 protocol.
REAL_WALD_ROOT = Path("../S2Diff-MH/data/augsburg2_wald_center_holdout")
REAL_RADIOMETRY = Path(
    "../S2Diff-MH/data/calibration/Augsburg2_Wald_center_holdout_radiometry.json"
)
REAL_SPLIT_PROTOCOL = "Augsburg2-Wald-center-holdout-v1"
REAL_TEST_BBOX_30M = (24, 36, 72, 84)
REAL_TEST_BBOX_10M = (72, 108, 216, 252)
REAL_FORBIDDEN_BBOX_30M = (18, 30, 78, 90)
REAL_TRAIN_PATCH = 24
REAL_TRAIN_STRIDE = 6
REAL_EVAL_PATCH = 48
REAL_MIN_VALID_FRACTION = 0.80
REAL_BATCH_SIZE = 1
REAL_NUM_WORKERS = 0
REAL_EPOCHS = 100
REAL_LR = 1e-5
REAL_WEIGHT_DECAY = 5e-5
REAL_EVAL_INTERVAL = 5
REAL_SAVE_INTERVAL = 5
REAL_MONITOR = "ref_sam"
REAL_SEED = 10
REAL_EVAL_SEED = 1234
REAL_TILE_SIZE = 96
REAL_TILE_STRIDE = 48
REAL_QNR_WINDOW_HR = 48
REAL_QNR_MIN_VALID_FRACTION = 0.80
REAL_QNR_SUPPORT_FRACTION = 0.01


def real_checkpoint_dir(adapter: ModelAdapter) -> Path:
    return (
        ROOT / "comparison" / adapter.folder
        / "checkpoints" / "augsburg2_wald_center_holdout"
    )


def real_log_dir(adapter: ModelAdapter) -> Path:
    return (
        ROOT / "comparison" / adapter.folder
        / "logs" / "augsburg2_wald_center_holdout"
    )


def real_output_dir(adapter: ModelAdapter) -> Path:
    return (
        ROOT / "comparison" / adapter.folder
        / "outputs" / "augsburg2_wald_center_holdout"
    )


def real_checkpoint_path(adapter: ModelAdapter) -> Path:
    return real_checkpoint_dir(adapter) / "best.pth.tar"


def build_real_train_command(adapter: ModelAdapter, device: str):
    cmd = [
        sys.executable,
        str(ROOT / adapter.real_train),
        "--stage", "train",
        "--wald_root", str(REAL_WALD_ROOT),
        "--radiometry_json", str(REAL_RADIOMETRY),
        "--checkpoint_dir", str(real_checkpoint_dir(adapter).relative_to(ROOT)),
        "--log_dir", str(real_log_dir(adapter).relative_to(ROOT)),
        "--train_patch_size", str(REAL_TRAIN_PATCH),
        "--train_stride", str(REAL_TRAIN_STRIDE),
        "--eval_patch_size", str(REAL_EVAL_PATCH),
        "--min_valid_fraction", str(REAL_MIN_VALID_FRACTION),
        "--batch_size", str(REAL_BATCH_SIZE),
        "--num_workers", str(REAL_NUM_WORKERS),
        "--epochs", str(REAL_EPOCHS),
        "--lr", str(REAL_LR),
        "--weight_decay", str(REAL_WEIGHT_DECAY),
        "--eval_interval", str(REAL_EVAL_INTERVAL),
        "--save_interval", str(REAL_SAVE_INTERVAL),
        "--monitor", REAL_MONITOR,
        "--seed", str(REAL_SEED),
        "--device", device,
    ]
    if adapter.canonical == "EMR-Diff":
        cmd += ["--eval_seed", str(REAL_EVAL_SEED), "--model_width", "64"]
    return cmd


def build_real_reference_test_command(adapter: ModelAdapter, device: str):
    checkpoint = real_checkpoint_path(adapter)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Real Augsburg checkpoint not found: {checkpoint}")
    cmd = [
        sys.executable,
        str(ROOT / adapter.real_train),
        "--stage", "test",
        "--wald_root", str(REAL_WALD_ROOT),
        "--radiometry_json", str(REAL_RADIOMETRY),
        "--checkpoint_dir", str(real_checkpoint_dir(adapter).relative_to(ROOT)),
        "--log_dir", str(real_log_dir(adapter).relative_to(ROOT)),
        "--checkpoint", str(checkpoint.relative_to(ROOT)),
        "--train_patch_size", str(REAL_TRAIN_PATCH),
        "--train_stride", str(REAL_TRAIN_STRIDE),
        "--eval_patch_size", str(REAL_EVAL_PATCH),
        "--min_valid_fraction", str(REAL_MIN_VALID_FRACTION),
        "--batch_size", "1",
        "--num_workers", "0",
        "--epochs", str(REAL_EPOCHS),
        "--lr", str(REAL_LR),
        "--weight_decay", str(REAL_WEIGHT_DECAY),
        "--eval_interval", str(REAL_EVAL_INTERVAL),
        "--save_interval", str(REAL_SAVE_INTERVAL),
        "--monitor", REAL_MONITOR,
        "--seed", str(REAL_SEED),
        "--device", device,
    ]
    if adapter.canonical == "EMR-Diff":
        cmd += ["--eval_seed", str(REAL_EVAL_SEED), "--model_width", "64"]
    return cmd


def build_real_native_test_command(adapter: ModelAdapter, device: str):
    checkpoint = real_checkpoint_path(adapter)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Real Augsburg checkpoint not found: {checkpoint}")
    cmd = [
        sys.executable,
        str(ROOT / adapter.real_infer),
        "--center_holdout",
        "--wald_root", str(REAL_WALD_ROOT),
        "--radiometry_json", str(REAL_RADIOMETRY),
        "--checkpoint", str(checkpoint.relative_to(ROOT)),
        "--save_root", str(real_output_dir(adapter).relative_to(ROOT)),
        "--tile_size", str(REAL_TILE_SIZE),
        "--tile_stride", str(REAL_TILE_STRIDE),
        "--device", device,
        "--write_tif",
        "--qnr_window_hr", str(REAL_QNR_WINDOW_HR),
        "--qnr_min_valid_fraction", str(REAL_QNR_MIN_VALID_FRACTION),
        "--qnr_support_fraction", str(REAL_QNR_SUPPORT_FRACTION),
    ]
    if adapter.canonical == "EMR-Diff":
        cmd += ["--eval_seed", str(REAL_EVAL_SEED)]
    return cmd


def real_protocol_summary() -> str:
    return (
        f"{REAL_SPLIT_PROTOCOL}; x3 Wald; test30={REAL_TEST_BBOX_30M}, "
        f"test10={REAL_TEST_BBOX_10M}, forbidden30={REAL_FORBIDDEN_BBOX_30M}; "
        f"train={REAL_TRAIN_PATCH}/stride{REAL_TRAIN_STRIDE}, "
        f"eval={REAL_EVAL_PATCH}; epochs={REAL_EPOCHS}; "
        f"AdamW lr={REAL_LR:g}, wd={REAL_WEIGHT_DECAY:g}; "
        f"best={REAL_MONITOR}; native tile={REAL_TILE_SIZE}/"
        f"{REAL_TILE_STRIDE}; QNR window={REAL_QNR_WINDOW_HR}"
    )


def protocol_summary() -> str:
    return (
        f"x{SCALE_RATIO}, physical MTF={MTF_NYQUIST}, truncate={PSF_TRUNCATE}; "
        f"train={PATCH_SIZE}/stride{STRIDE}, eval={IMAGE_SIZE}; "
        f"stage1={REGISTERED_EPOCHS} epochs registered; "
        f"stage2={MIXED_EPOCHS} epochs: {IDENTITY_PROBABILITY:.0%} identity + "
        f"{DEFORMED_PROBABILITY:.0%} deformed, "
        f"dx/dy~U(-{MAX_TRANSLATION:g},{MAX_TRANSLATION:g}), "
        f"rot~U(-{MAX_ROTATION_DEG:g},{MAX_ROTATION_DEG:g}), "
        f"local amplitude~U(0,{MAX_LOCAL_PX:g}), minJac={MIN_JACOBIAN}; "
        f"AdamW lr={LR:g}, wd={WEIGHT_DECAY:g}, batch={BATCH_SIZE}"
    )
