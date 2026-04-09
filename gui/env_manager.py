"""
env_manager.py — Conda environment discovery, tool-path resolution, and validation.

Supports three modes per tool:
  "system"  — use whatever is on the current $PATH (default)
  "conda"   — run the command inside a named conda env via ``conda run``
  "path"    — use an explicit executable path supplied by the user
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from typing import List, Tuple

# ─── Shell-operator detection ─────────────────────────────────────────────────
# Commands containing these tokens need a shell to interpret them.
_SHELL_OPS = ("|", ">", "<", "&&", "||", ";", "$(", "`")


def _has_shell_ops(cmd: str) -> bool:
    return any(op in cmd for op in _SHELL_OPS)


# ─── Tool → verification command ─────────────────────────────────────────────
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


# ─── Conda executable discovery ───────────────────────────────────────────────

def _conda_exe() -> str:
    """
    Return the **full filesystem path** to conda/mamba/micromamba.

    Using the full path matters because subprocess uses /bin/sh which may have
    a minimal PATH that does not include conda's bin directory, even when the
    parent Streamlit process can see conda perfectly well.
    """
    for name in ("conda", "mamba", "micromamba"):
        path = shutil.which(name)   # resolved in the parent process's PATH
        if path:
            return path
    return "conda"  # last-resort fallback


# ─── Conda environment discovery ─────────────────────────────────────────────

def list_conda_envs() -> List[str]:
    """
    Return the names of all conda environments found on this machine.
    Tries conda/mamba/micromamba in order.
    Returns an empty list if conda is not available.
    """
    conda = _conda_exe()
    try:
        r = subprocess.run(
            [conda, "env", "list", "--json"],
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
        pass
    return []


def conda_available() -> bool:
    return shutil.which("conda") is not None or \
           shutil.which("mamba") is not None or \
           shutil.which("micromamba") is not None


# ─── Command wrapping ─────────────────────────────────────────────────────────

def wrap_cmd(cmd: str, tool_name: str, mode: str, value: str) -> str:
    """
    Return *cmd* optionally wrapped to run inside a specific environment.

    Parameters
    ----------
    cmd       : Full shell command string (may include env-var prefixes such as
                ``CUDA_VISIBLE_DEVICES=0 deepsignal_plant ...`` or shell
                operators like pipes and redirects).
    tool_name : Bare executable name as it appears in *cmd* (used only by
                "path" mode).
    mode      : ``"system"`` | ``"conda"`` | ``"path"``
    value     : Conda env name or full exe path (ignored for ``"system"``).
    """
    value = (value or "").strip()

    if mode == "conda" and value:
        conda = _conda_exe()
        if _has_shell_ops(cmd):
            # Commands with pipes, redirects, env-var assignments etc. must go
            # through bash -c so the shell interprets those operators.
            # Escape single-quotes inside cmd for safe embedding.
            safe = cmd.replace("'", "'\\''")
            return (
                f"{shlex.quote(conda)} run -n {shlex.quote(value)}"
                f" --no-capture-output bash -c '{safe}'"
            )
        else:
            # Simple commands (no shell operators): pass tokens directly to
            # conda run — avoids bash -c quoting issues entirely.
            return (
                f"{shlex.quote(conda)} run -n {shlex.quote(value)}"
                f" --no-capture-output {cmd}"
            )

    if mode == "path" and value:
        return cmd.replace(tool_name, shlex.quote(value), 1)

    return cmd  # "system" — no change


# ─── Tool validation ─────────────────────────────────────────────────────────

def validate_tool(tool_name: str, mode: str, value: str) -> Tuple[bool, str]:
    """
    Step-by-step validation so the GUI can show exactly where a failure occurs.

    Steps for conda mode:
      1. Locate the conda binary (full path)
      2. Confirm conda itself runs in a subprocess
      3. Confirm the named environment exists
      4. Smoke-test ``conda run -n <env> echo CONDA_OK``
      5. Run the actual tool verification command

    Returns (success, diagnostic_text).
    """
    lines: list[str] = []

    # ══════════════════════════════════════════════════════════════════════════
    # CONDA MODE — incremental diagnostics
    # ══════════════════════════════════════════════════════════════════════════
    if mode == "conda":
        env_name = (value or "").strip()
        if not env_name:
            return False, "No conda environment name specified."

        # ── Step 1: find conda binary ─────────────────────────────────────────
        conda = _conda_exe()
        lines.append(f"[1] conda binary : {conda}")

        # ── Step 2: can conda itself run? ─────────────────────────────────────
        try:
            r2 = subprocess.run(
                [conda, "--version"],
                capture_output=True, text=True, timeout=15,
            )
            ver_out = (r2.stdout + r2.stderr).strip()
            if r2.returncode == 0 or ver_out:
                lines.append(f"[2] conda runs   : {ver_out}")
            else:
                lines.append(f"[2] conda FAILED (exit {r2.returncode}) — is conda on PATH?")
                return False, _trim("\n".join(lines))
        except FileNotFoundError:
            lines.append(f"[2] conda FAILED : '{conda}' not found — conda is not on PATH")
            return False, _trim("\n".join(lines))
        except Exception as exc:
            lines.append(f"[2] conda FAILED : {exc}")
            return False, _trim("\n".join(lines))

        # ── Step 3: does the env exist? ───────────────────────────────────────
        available = list_conda_envs()
        if env_name == "base" or env_name in available:
            lines.append(f"[3] env exists   : '{env_name}' ✓")
        else:
            lines.append(
                f"[3] env NOT FOUND: '{env_name}'\n"
                f"    Available    : {', '.join(available) or '(none detected)'}"
            )
            return False, _trim("\n".join(lines))

        # ── Step 4: smoke-test conda run ──────────────────────────────────────
        smoke_cmd = (
            f"{shlex.quote(conda)} run -n {shlex.quote(env_name)}"
            f" --no-capture-output echo CONDA_OK"
        )
        try:
            r4 = subprocess.run(
                smoke_cmd, shell=True,
                capture_output=True, text=True, timeout=30,
            )
            smoke_out = (r4.stdout + r4.stderr).strip()
            if "CONDA_OK" in smoke_out:
                lines.append(f"[4] conda run    : smoke test PASSED")
            else:
                lines.append(
                    f"[4] conda run    : smoke test FAILED\n"
                    f"    cmd          : {smoke_cmd}\n"
                    f"    output       : {smoke_out or '(empty)'}\n"
                    f"    exit code    : {r4.returncode}"
                )
                return False, _trim("\n".join(lines))
        except subprocess.TimeoutExpired:
            lines.append(f"[4] conda run    : timed out (30 s) — try again")
            return False, _trim("\n".join(lines))
        except Exception as exc:
            lines.append(f"[4] conda run    : error — {exc}")
            return False, _trim("\n".join(lines))

        lines.append("")  # blank line before tool output

    # ══════════════════════════════════════════════════════════════════════════
    # SYSTEM / PATH MODE — no preamble needed
    # ══════════════════════════════════════════════════════════════════════════

    # ── Step 5: run the actual tool verification ──────────────────────────────
    test_args = _VERIFY_ARGS.get(tool_name, ["--version"])
    last_output = ""

    for arg in test_args:
        base_cmd = f"{tool_name} {arg}"
        cmd = wrap_cmd(base_cmd, tool_name, mode, value)
        try:
            r = subprocess.run(
                cmd, shell=True,
                capture_output=True, text=True,
                timeout=60,
            )
            out = (r.stdout + r.stderr).strip()

            conda_err_tokens = (
                "EnvironmentLocationNotFound",
                "EnvironmentNameNotFound",
                "PackagesNotFoundError",
                "CommandNotFoundError",
            )
            if out and not any(t in out for t in conda_err_tokens):
                lines.append(f"$ {cmd}\n{out}")
                return True, _trim("\n".join(lines))

            last_output = f"$ {cmd}\n(no output, exit {r.returncode})\n{out}".strip()

        except subprocess.TimeoutExpired:
            last_output = f"$ {cmd}\nTimed out (60 s)."
        except Exception as exc:
            last_output = f"$ {cmd}\nError: {exc}"

    lines.append(last_output)
    return False, _trim("\n".join(lines))


def _trim(s: str, limit: int = 1500) -> str:
    return s[:limit] + ("…" if len(s) > limit else "")
