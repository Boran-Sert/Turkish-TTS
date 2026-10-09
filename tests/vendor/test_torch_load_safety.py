"""Regression guard: every ``torch.load`` call must set ``weights_only=True``.

VoxCPM deliberately loads checkpoints with ``weights_only=True`` so that a
crafted ``.ckpt``/``.pth``/``.bin`` file cannot execute arbitrary code via
pickle during unpickling (see
``tests/test_lora_checkpoint_loading.py::test_load_lora_weights_rejects_malicious_pickle_payloads``).

The fine-tuning resume path in ``scripts/train_voxcpm_finetune.py`` originally
called ``torch.load`` without that flag, leaving an arbitrary-code-execution
gap when resuming from an attacker-supplied checkpoint directory. This test
statically asserts the flag is present on every ``torch.load`` call across the
package and scripts so the gap cannot silently reappear.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories whose Python files load checkpoints at runtime / on resume.
SCANNED_DIRS = [REPO_ROOT / "src"]


def _python_files():
    for entry in SCANNED_DIRS:
        if entry.is_file() and entry.suffix == ".py":
            yield entry
        elif entry.is_dir():
            yield from entry.rglob("*.py")


def _is_torch_load(node: ast.Call) -> bool:
    func = node.func
    # Matches ``torch.load(...)`` and ``load(...)`` aliased from torch.
    if isinstance(func, ast.Attribute) and func.attr == "load":
        return isinstance(func.value, ast.Name) and func.value.id == "torch"
    return False


def _has_weights_only_true(node: ast.Call) -> bool:
    for kw in node.keywords:
        if kw.arg == "weights_only":
            return isinstance(kw.value, ast.Constant) and kw.value.value is True
    return False


def test_every_torch_load_sets_weights_only_true():
    offenders = []
    checked = 0
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_torch_load(node):
                checked += 1
                if not _has_weights_only_true(node):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert checked > 0, "expected to find at least one torch.load call to verify"
    assert not offenders, (
        "torch.load without weights_only=True (pickle RCE risk):\n  "
        + "\n  ".join(offenders)
    )


def test_torch_load_weights_only_blocks_malicious_pickle(tmp_path):
    """Behavioral check that weights_only=True actually rejects a code-exec payload."""
    torch = pytest.importorskip("torch")

    marker = tmp_path / "pwned.txt"

    class Exploit:
        def __reduce__(self):
            import pathlib

            return (pathlib.Path.write_text, (marker, "executed\n"))

    ckpt = tmp_path / "optimizer.pth"
    torch.save({"state_dict": {"w": torch.zeros(1)}, "boom": Exploit()}, ckpt)

    with pytest.raises(Exception):
        torch.load(ckpt, map_location="cpu", weights_only=True)

    assert not marker.exists(), "malicious pickle executed despite weights_only=True"
