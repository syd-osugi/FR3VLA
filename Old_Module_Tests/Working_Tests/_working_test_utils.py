"""
Shared helpers for the lightweight Working module test scripts.

These tests are intentionally plain Python scripts instead of pytest tests so
they can be run one at a time on the robot workstation without extra tooling.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
PROJECT_ROOT = REPO_ROOT.parent
WORKING_DIR = PROJECT_ROOT / "Working"
TEST_OUTPUTS_DIR = REPO_ROOT / "Test_Outputs"


def add_working_to_path() -> None:
    """Make the top-level Working package importable from Module_Tests."""
    working_path = str(WORKING_DIR)
    if working_path not in sys.path:
        sys.path.insert(0, working_path)


add_working_to_path()


def unique_output_path(path: Path) -> Path:
    """
    Return a non-existing path by adding a numeric suffix when needed.

    Test output images are evidence from a specific run, so overwriting them can
    hide useful debugging history. If d435_frame_00.png already exists, this
    returns d435_frame_00_001.png, then _002, and so on.
    """
    if not path.exists():
        return path

    for index in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find an unused output path for {path}")


def require(condition: bool, message: str) -> None:
    """Raise AssertionError with a readable message when condition is false."""
    if not condition:
        raise AssertionError(message)


def require_close(actual, expected, message: str, atol: float = 1e-9) -> None:
    """Assert numeric arrays are close."""
    if not np.allclose(actual, expected, atol=atol):
        raise AssertionError(f"{message}\nactual={actual}\nexpected={expected}")


def require_raises(error_type, fn, message: str) -> Exception:
    """Assert that fn raises error_type and return the exception."""
    try:
        fn()
    except error_type as exc:
        return exc
    raise AssertionError(message)


@contextmanager
def patched_attr(obj, name: str, value):
    """Temporarily replace an object attribute."""
    sentinel = object()
    old_value = getattr(obj, name, sentinel)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if old_value is sentinel:
            delattr(obj, name)
        else:
            setattr(obj, name, old_value)


@contextmanager
def temp_environ(overrides):
    """Temporarily apply environment variable overrides."""
    old_values = {}
    for key, value in overrides.items():
        old_values[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)

    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def homogeneous(translation=(0.0, 0.0, 0.0), rotation=None):
    """Create a 4x4 homogeneous transform."""
    matrix = np.eye(4, dtype=float)
    if rotation is not None:
        matrix[:3, :3] = np.array(rotation, dtype=float)
    matrix[:3, 3] = np.array(translation, dtype=float)
    return matrix


def write_transform_json(path: Path, matrix) -> Path:
    """Write a calibration transform JSON file in the Working format."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"matrix": np.array(matrix, dtype=float).tolist()}, indent=2),
        encoding="utf-8",
    )
    return path


def run_tests(tests) -> int:
    """Run a list of (name, callable) tests and print a compact report."""
    failures = []
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"PASS {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"FAIL {name}: {exc}")

    print("-" * 60)
    print(f"Ran {len(tests)} checks: {len(tests) - len(failures)} passed, {len(failures)} failed")

    if failures:
        print("\nFailures:")
        for name, exc in failures:
            print(f"  - {name}: {type(exc).__name__}: {exc}")
        return 1
    return 0
