"""
Run the no-hardware Working unit scripts.

Usage:
    python3 Module_Tests/Working_Tests/run_unit_tests.py
    python3 Module_Tests/Working_Tests/run_unit_tests.py --pattern "unit_trajectory*"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        default="unit_*.py",
        help="Glob pattern relative to this folder. Default: unit_*.py",
    )
    args = parser.parse_args()

    test_dir = Path(__file__).resolve().parent
    repo_root = test_dir.parents[1]
    scripts = sorted(path for path in test_dir.glob(args.pattern) if path.name != Path(__file__).name)

    if not scripts:
        print(f"No scripts matched {args.pattern!r} in {test_dir}")
        return 1

    failures = []
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    for script in scripts:
        print("=" * 72)
        print(f"Running {script.relative_to(repo_root)}")
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode != 0:
            failures.append((script.name, result.returncode))

    print("=" * 72)
    print(f"Finished {len(scripts)} scripts: {len(scripts) - len(failures)} passed, {len(failures)} failed")
    if failures:
        for name, code in failures:
            print(f"  - {name}: exit code {code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
