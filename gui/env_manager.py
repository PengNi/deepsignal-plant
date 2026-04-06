"""
env_manager.py — Conda environment discovery, tool-path resolution, and validation.

Supports three modes per tool:
  "system"  — use whatever is on the current $PATH (default)
  "conda"   — activate a named conda env via ``conda run -n <env>``
  "path"    — use an explicit executable path supplied by the user
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from typing import List, Tuple


# ─── Tool → verification command ─────────────────────────────────────────────
# Ordered: we try each arg in turn; success = non-empty output
_VERIFY_ARGS: dict[str, list[str]] = {
    "deepsignal_plant":       ["--version", "--help"],
    "guppy_basecaller":       ["--version", "--help"],
    "dorado":                 ["--version", "--help"],
    "tombo":                  ["--version", "--help"],
    "samtools":               ["--version"],
    "minimap2":               ["--version"],
    "multi_to_single_fast5":  ["--version", "--help"],
    "python":                 ["--version"],
    "conda":                  ["--version"],
}


# ─── Conda environment discovery ─────────────────────────────────────────────

def list_conda_envs() -> List[str]:
    """
    Return the names of all conda environments found on this machine.
    Tries ``conda``, ``mamba``, and ``micromamba`` in order.
    Returns an empty list if conda is not available.
    """
    for exe in ("conda", "mamba", "micromamba"):
        if not shutil.which(exe):
            continue
        try:
            r = subprocess.run(
                [exe, "env", "list", "--json"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                data = json.loads(r.stdout)
                names: list[str] = []
                for path in data.get("envs", []):
                    name = os.path.basename(path)
                    if name:
                        names.append(name)
                return names
        except Exception:
            continue
    return []


def conda_available() -> bool:
    return any(shutil.which(e) for e in ("conda", "mamba", "micromamba"))


def _conda_exe() -> str:
    for e in ("conda", "mamba", "micromamba"):
        if shutil.which(e):
            return e
    return "conda"


# ─── Command wrapping ─────────────────────────────────────────────────────────

def wrap_cmd(cmd: str, tool_name: str, mode: str, value: str) -> str:
    """
    Return *cmd* optionally wrapped to run inside a particular environment.

    Parameters
    ----------
    cmd       : The full shell command string (may include env-var prefixes).
    tool_name : The bare executable name inside *cmd* (e.g. ``"deepsignal_plant"``).
    mode      : ``"system"`` | ``"conda"`` | ``"path"``
    value     : Conda env name or full exe path (ignored when mode == "system").
    """
    value = (value or "").strip()

    if mode == "conda" and value:
        conda = _conda_exe()
        # Wrap the entire command in `conda run … bash -c '…'`
        # Escape single-quotes inside cmd for safe embedding
        safe = cmd.replace("'", "'\\''")
        return f"{conda} run --no-capture-output -n {shlex.quote(value)} bash -c '{safe}'"

    if mode == "path" and value:
        # Replace the first occurrence of tool_name in cmd with the custom path.
        # Handles leading env-var assignments like "CUDA_VISIBLE_DEVICES=0 deepsignal_plant …"
        quoted = shlex.quote(value)
        return cmd.replace(tool_name, quoted, 1)

    return cmd  # "system" — no change


# ─── Tool validation ─────────────────────────────────────────────────────────

def validate_tool(tool_name: str, mode: str, value: str) -> Tuple[bool, str]:
    """
    Test whether *tool_name* is accessible under the given env config.

    Returns (success, output_snippet).  ``success`` is True when the process
    produces any output (stdout or stderr), regardless of exit code, because
    many bioinformatics tools return non-zero from ``--version``.
    """
    test_args = _VERIFY_ARGS.get(tool_name, ["--version"])
    last_output = ""
    last_ok = False

    for arg in test_args:
        base_cmd = f"{tool_name} {arg}"
        cmd = wrap_cmd(base_cmd, tool_name, mode, value)
        try:
            r = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            out = (r.stdout + r.stderr).strip()
            if out:
                return True, _trim(f"$ {cmd}\n{out}")
            last_output = f"$ {cmd}\n(no output, exit {r.returncode})"
            last_ok = False
        except subprocess.TimeoutExpired:
            last_output = f"$ {cmd}\nTimed out (30 s). Tool may be unavailable or conda activation is slow."
            last_ok = False
        except Exception as exc:
            last_output = f"$ {cmd}\nError: {exc}"
            last_ok = False

    return last_ok, _trim(last_output)


def _trim(s: str, limit: int = 800) -> str:
    return s[:limit] + ("…" if len(s) > limit else "")
