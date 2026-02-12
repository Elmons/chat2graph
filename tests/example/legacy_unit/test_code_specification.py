import os
import shutil
import subprocess

import pytest


def run_command(command: str):
    """Run a command and assert it exits with 0."""
    exe = command.strip().split(" ", 1)[0]
    if shutil.which(exe) is None:
        pytest.skip(f"Command {exe!r} not available")
    try:
        subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"Command '{command}' failed with exit code {e.returncode}.\n"
            f"Stdout:\n{e.stdout}\n"
            f"Stderr:\n{e.stderr}",
            pytrace=False,
        )


def test_mypy_checks():
    """Run mypy checks on the codebase."""
    if os.getenv("CHAT2GRAPH_RUN_LINT_TESTS", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set CHAT2GRAPH_RUN_LINT_TESTS=1 to enable lint/typecheck tests")
    run_command("mypy app")
    run_command("mypy tests/example")


def test_ruff_checks():
    """Run ruff checks on the codebase."""
    if os.getenv("CHAT2GRAPH_RUN_LINT_TESTS", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set CHAT2GRAPH_RUN_LINT_TESTS=1 to enable lint/typecheck tests")
    run_command("ruff check app")
    run_command("ruff check tests/example")
