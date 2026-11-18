import csv
import json
from pathlib import Path

import torch
from torch import nn

import benchmark_paviau as benchmark


def test_repeated_inference_work_counts_mac_twice_but_shares_parameters():
    layer = nn.Conv2d(3, 3, 1, bias=False)
    layer.requires_grad_(False)
    x = torch.ones(1, 3, 4, 5)

    def forward():
        for _ in range(3):
            layer(x)

    assert benchmark.parameter_count([layer, layer]) == 9
    assert benchmark.dense_flops(forward)[0] == 3 * 4 * 5 * 9 * 2


def test_benchmark_plan_includes_all_methods_without_data(capsys):
    assert benchmark.main(["--dry_run", "--emr_width", "128"]) == 0
    plan = capsys.readouterr().out
    assert plan.count("PLAN ") == 13
    assert "best_base128.pth.tar" in plan
    assert "hsi_warp_final/PaviaU" in plan


def test_initialized_plan_keeps_emr_width_without_resolving_checkpoints(monkeypatch, capsys):
    def unexpected(*args):
        raise AssertionError("No checkpoint should be resolved")

    monkeypatch.setattr(benchmark, "resolve_checkpoint", unexpected)
    assert benchmark.main(["--dry_run", "--no_checkpoints", "--emr_width", "128"]) == 0
    plan = capsys.readouterr().out
    assert plan.count("initialized (no checkpoint)") == 11
    assert "backbone_width=128" in plan


def test_retry_ours_preserves_other_rows(tmp_path, monkeypatch):
    ours = tmp_path / "S2Diff-MH"
    ours.mkdir()
    (ours / "benchmark_paviau.py").touch()
    baseline = {"method": "EMR-Diff", "status": "ok", "dataset": "PaviaU",
                "params_M": "1.5", "input_sha256": "test-pair", "seed": "1234",
                "weight_source": "initialized"}
    benchmark.write_summary(tmp_path, [baseline, {"method": "ours", "status": "failed"}])
    monkeypatch.setattr(benchmark, "save_shared_pair", lambda *args: "test-pair")

    def run_ours(command, **kwargs):
        assert Path(command[1]).parent == ours
        result = Path(command[command.index("--_result") + 1])
        result.write_text(json.dumps({"params_M": 2.0}), encoding="utf-8")
        return benchmark.SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", run_ours)
    assert benchmark.main(["--methods", "ours", "--no_checkpoints", "--keep_other_results",
                           "--s2diff_root", str(ours), "--output_dir", str(tmp_path)]) == 0
    with (tmp_path / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        old, new = csv.DictReader(handle)
    assert all(old[key] == value for key, value in baseline.items())
    assert (new["method"], new["status"], new["params_M"]) == ("ours", "ok", "2.0")
