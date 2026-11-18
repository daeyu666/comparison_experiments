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
