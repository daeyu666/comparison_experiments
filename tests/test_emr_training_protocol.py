"""Guard shared validation cadence and EMR runtime overrides without scene data."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment_protocol as protocol

EMR = protocol.resolve_model("EMR-Diff")
UAFL = protocol.resolve_model("UAFL")
INTERVALS = {"PaviaU": 20, "Houston13": 10, "Chikusei": 5,
             "CAVE": 5, "Botswana": 20, "Augsburg": 10}


def option(command, flag):
    return command[command.index(flag) + 1]


@pytest.mark.parametrize("dataset,interval", INTERVALS.items())
@pytest.mark.parametrize("mode", ["registered", "mixed"])
@pytest.mark.parametrize("model", ["EMR-Diff", "UAFL", "FutureMethod"])
def test_all_methods_inherit_dataset_intervals(tmp_path, monkeypatch, dataset, interval, mode, model):
    if model == "FutureMethod":
        # Register a new method without defining any validation-specific logic.
        adapter = protocol.ModelAdapter(
            canonical=model, folder=model,
            registered_train="comparison/FutureMethod/train.py",
            mixed_train="comparison/FutureMethod/train_mixed.py",
            test_script="comparison/FutureMethod/test.py", real_train="", real_infer="",
        )
        monkeypatch.setitem(protocol._ADAPTERS, "futuremethod", adapter)
    adapter = protocol.resolve_model(model)
    monkeypatch.setattr(protocol, "ROOT", tmp_path)
    checkpoint = tmp_path / "best.pth.tar"
    checkpoint.touch()
    monkeypatch.setattr(protocol, "checkpoint_path", lambda *args: checkpoint)
    command = protocol.build_train_command(adapter, dataset, mode, "cuda:0")
    assert option(command, "--batch_size") == ("2" if model == "EMR-Diff" else "1")
    assert option(command, "--lr") == "1e-05"
    assert option(command, "--validation_interval") == str(interval)
    assert option(command, "--epochs") == ("800" if mode == "registered" else "600")


def test_batch_override_and_other_method_defaults():
    command = protocol.build_train_command(EMR, "CAVE", "registered", "cuda:0", batch_size=4)
    assert option(command, "--batch_size") == "4"
    assert option(command, "--lr") == "1e-05"
    assert "batch=4" in protocol.protocol_summary(EMR, 4)
    command = protocol.build_train_command(UAFL, "PaviaU", "registered", "cuda:0")
    assert option(command, "--batch_size") == "1"
    command = protocol.build_train_command(UAFL, "CAVE", "registered", "cuda:0")
    assert option(command, "--validation_interval") == "5"
    with pytest.raises(ValueError, match="only to EMR-Diff"):
        protocol.build_train_command(UAFL, "CAVE", "registered", "cuda:0", batch_size=4)
    with pytest.raises(ValueError, match=">=1"):
        protocol.build_train_command(EMR, "CAVE", "registered", "cuda:0", batch_size=0)


def test_wald_batch_override_does_not_change_reference_eval_protocol():
    for batch, expected in [(None, "2"), (4, "4")]:
        command = protocol.build_real_train_command(EMR, "cuda:0", batch_size=batch)
        assert option(command, "--batch_size") == expected
        assert option(command, "--lr") == "1e-05"
        assert option(command, "--eval_interval") == "5"
        assert option(command, "--train_patch_size") == "24"
        assert option(command, "--model_width") == "64"
        assert option(command, "--monitor") == "ref_sam"
    assert "batch=4" in protocol.real_protocol_summary(EMR, 4)
    command = protocol.build_real_train_command(UAFL, "cuda:0")
    assert option(command, "--batch_size") == "1"
