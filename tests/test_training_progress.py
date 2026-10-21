"""Exercise training progress rendering without models, datasets or GPU work."""
import ast
import importlib
import io
import os
import sys
from pathlib import Path

import pytest
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
TRAINERS = (
    'comparison/CLSNet/train.py',
    'comparison/DRTNet/train.py',
    'comparison/EMR-Diff/model/ResShift_model.py',
    'comparison/EMR-Diff/train_hsi_deformed.py',
)


def calls(path, name):
    return [node for node in ast.walk(ast.parse((ROOT/path).read_text()))
            if isinstance(node, ast.Call)
            and ((isinstance(node.func, ast.Name) and node.func.id == name)
                 or (isinstance(node.func, ast.Attribute) and node.func.attr == name))]


def rendering_options(path):
    options = {kw.arg: kw.value for kw in calls(path, 'tqdm')[0].keywords}
    return {key: ast.literal_eval(options[key])
            for key in ('dynamic_ncols', 'mininterval', 'disable')}


@pytest.mark.parametrize('path', TRAINERS)
def test_trainers_share_terminal_safe_rendering(path):
    # Check the actual constructors, including the EMR evaluation status line.
    for call in calls(path, 'tqdm'):
        options = {kw.arg: kw.value for kw in call.keywords}
        assert ast.literal_eval(options['dynamic_ncols']) is True
        assert ast.literal_eval(options['mininterval']) >= 1.0
        assert ast.literal_eval(options['disable']) is None
    for call in calls(path, 'set_postfix'):
        options = {kw.arg: kw.value for kw in call.keywords}
        assert ast.literal_eval(options['refresh']) is False


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Capture(io.StringIO):
    def __init__(self, tty, fd=None):
        super().__init__()
        self.tty, self.fd, self.columns = tty, fd, 180
        self.frames = []

    def isatty(self):
        return self.tty

    def fileno(self):
        if self.fd is None:
            raise io.UnsupportedOperation('no terminal')
        return self.fd

    def write(self, text):
        # Each status_printer write contains a carriage return and one line.
        if text.startswith('\r'):
            self.frames.append((self.columns, text.strip('\r\n')))
        return super().write(text)


def simulate(monkeypatch, options, *, tty=True, refresh=False):
    if tty and os.name != 'posix':
        pytest.skip('PTY resize regression requires a POSIX terminal')
    master = slave = None
    if tty:
        import pty
        import fcntl
        import struct
        import termios
        master, slave = pty.openpty()
    out = Capture(tty, slave)
    clock = Clock()

    def resize(columns):
        out.columns = columns
        if tty:
            fcntl.ioctl(slave, termios.TIOCSWINSZ,
                        struct.pack('HHHH', 30, columns, 0, 0))

    try:
        resize(180)
        with monkeypatch.context() as patch:
            patch.setattr(importlib.import_module('tqdm.std'), 'time', clock)
            patch.setattr(tqdm, 'monitor_interval', 0)
            patch.setattr(sys, 'stderr', out)
            seen = []
            progress = tqdm(range(615), desc='CLSNet registered 46/800', **options)
            for batch in progress:
                if batch == 600:
                    resize(100)  # SSH/IDE split pane becomes narrower mid-epoch.
                clock.now += 0.11
                seen.append(batch)
                progress.set_postfix(l1='0.012345', identity='1.000', refresh=refresh)
            final_columns = getattr(progress, 'ncols', None)
    finally:
        if master is not None:
            os.close(master)
            os.close(slave)
    return out, seen, final_columns


@pytest.mark.parametrize('path', TRAINERS)
def test_progress_refits_after_resize_without_skipping_batches(monkeypatch, path):
    out, seen, columns = simulate(monkeypatch, rendering_options(path))
    assert seen == list(range(615))
    assert columns <= 100
    assert any(width == 100 for width, _ in out.frames)
    # The first shorter frame can include trailing spaces to erase the old line.
    assert all(len(line.rstrip()) < width for width, line in out.frames)
    # 615 batches take 67.65 simulated seconds; no second forced refresh per batch.
    assert len(out.frames) <= 70
    assert '615/615' in out.getvalue()


@pytest.mark.parametrize('path', TRAINERS)
def test_redirected_logs_omit_carriage_return_status_frames(monkeypatch, path):
    out, seen, _ = simulate(monkeypatch, rendering_options(path), tty=False)
    assert seen == list(range(615))
    assert out.getvalue() == ''


def test_old_defaults_reproduce_overflow_and_double_refresh(monkeypatch):
    out, seen, columns = simulate(monkeypatch, {}, refresh=True)
    assert seen == list(range(615))
    assert columns == 179  # Cached initial terminal width, even after resize.
    assert any(len(line) >= width for width, line in out.frames if width == 100)
    assert len(out.frames) > 2 * 600
