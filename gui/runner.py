"""
runner.py — Backend shell-call logic for DeepSignal-plant GUI.
Handles pre-flight checks, command building, and subprocess execution.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Tuple


# ─── Pre-flight checks ────────────────────────────────────────────────────────

def tool_in_path(name: str) -> bool:
    return shutil.which(name) is not None


def path_exists(p: str) -> bool:
    return bool(p) and Path(p).exists()


def is_dir(p: str) -> bool:
    return bool(p) and Path(p).is_dir()


def disk_free_gb(p: str) -> float:
    """Return free disk space in GB for the filesystem containing *p*."""
    try:
        target = p if Path(p).exists() else str(Path(p).parent)
        return shutil.disk_usage(target).free / 1024 ** 3
    except Exception:
        return -1.0


def preflight(checks: list[Tuple[bool, str]]) -> list[str]:
    """Return list of error messages for every failed (condition, message) pair."""
    return [msg for ok, msg in checks if not ok]


# ─── Command builders ─────────────────────────────────────────────────────────

def _q(s: str) -> str:
    """Quote a shell argument (pass-through on empty string)."""
    return shlex.quote(s) if s else '""'


def build_multi_to_single(input_dir: str, output_dir: str, threads: int = 30) -> str:
    return (
        f"multi_to_single_fast5 -i {_q(input_dir)} -s {_q(output_dir)}"
        f" -t {threads} --recursive"
    )


def build_guppy(
    input_dir: str,
    output_dir: str,
    config: str,
    device: str = "cuda:all:100%",
    min_qscore: int = 7,
    extra_args: str = "",
) -> str:
    cmd = (
        f"guppy_basecaller -i {_q(input_dir)} -r -s {_q(output_dir)}"
        f" --config {_q(config)} --device {_q(device)}"
        f" --min_qscore {min_qscore}"
    )
    if extra_args.strip():
        cmd += f" {extra_args.strip()}"
    return cmd


def build_cat_fastq(input_dir: str, output_file: str) -> str:
    return f"cat {_q(input_dir)}/*.fastq > {_q(output_file)}"


def build_tombo_annotate(
    fast5_basedir: str,
    fastq_file: str,
    summary_file: str,
    basecall_group: str,
    basecall_subgroup: str,
    processes: int,
    overwrite: bool = True,
) -> str:
    cmd = (
        f"tombo preprocess annotate_raw_with_fastqs"
        f" --fast5-basedir {_q(fast5_basedir)}"
        f" --fastq-filenames {_q(fastq_file)}"
        f" --sequencing-summary-filenames {_q(summary_file)}"
        f" --basecall-group {_q(basecall_group)}"
        f" --basecall-subgroup {_q(basecall_subgroup)}"
        f" --processes {processes}"
    )
    if overwrite:
        cmd += " --overwrite"
    return cmd


def build_tombo_resquiggle(
    fast5_dir: str,
    reference: str,
    corrected_group: str,
    basecall_group: str,
    processes: int,
    overwrite: bool = True,
    failed_reads_dir: str = "",
) -> str:
    cmd = (
        f"tombo resquiggle {_q(fast5_dir)} {_q(reference)}"
        f" --processes {processes}"
        f" --corrected-group {_q(corrected_group)}"
        f" --basecall-group {_q(basecall_group)}"
    )
    if overwrite:
        cmd += " --overwrite"
    if failed_reads_dir.strip():
        cmd += f" --failed-reads-filename {_q(failed_reads_dir.strip())}"
    return cmd


def build_dorado(
    dorado_model: str,
    pod5_dir: str,
    reference: str,
    output_bam: str,
    device: str = "cuda:all",
    extra_args: str = "",
) -> str:
    cmd = (
        f"dorado basecaller {_q(dorado_model)}"
        f" --emit-moves --device {_q(device)} {_q(pod5_dir)}"
        f" --reference {_q(reference)}"
    )
    if extra_args.strip():
        cmd += f" {extra_args.strip()}"
    cmd += f" > {_q(output_bam)}"
    return cmd


def build_call_mods(
    input_path: str,
    model_path: str,
    result_file: str,
    nproc: int,
    nproc_gpu: int,
    motifs: str,
    cuda_devices: str = "0",
    batch_size: int = 512,
    bam: str = "",
    corrected_group: str = "",
    basecall_subgroup: str = "BaseCalled_template",
    normalize_method: str = "mad",
    mod_loc: int = 0,
    is_dna: bool = True,
    reference_path: str = "",
    region: str = "",
    positions: str = "",
    gzip_out: bool = False,
    seq_len: int = 13,
    signal_len: int = 16,
    model_type: str = "both_bilstm",
) -> str:
    env = f"CUDA_VISIBLE_DEVICES={cuda_devices} "
    cmd = (
        f"{env}deepsignal_plant call_mods"
        f" --input_path {_q(input_path)}"
        f" --model_path {_q(model_path)}"
        f" --result_file {_q(result_file)}"
        f" --motifs {_q(motifs)}"
        f" --nproc {nproc} --nproc_gpu {nproc_gpu}"
        f" --batch_size {batch_size}"
        f" --normalize_method {normalize_method}"
        f" --mod_loc {mod_loc}"
        f" --seq_len {seq_len} --signal_len {signal_len}"
        f" --model_type {model_type}"
    )
    if bam.strip():
        cmd += f" --bam {_q(bam.strip())}"
    if corrected_group.strip():
        cmd += f" --corrected_group {_q(corrected_group.strip())}"
        cmd += f" --basecall_subgroup {_q(basecall_subgroup)}"
    if not is_dna:
        cmd += " --is_dna no"
    if reference_path.strip():
        cmd += f" --reference_path {_q(reference_path.strip())}"
    if region.strip():
        cmd += f" --region {_q(region.strip())}"
    if positions.strip():
        cmd += f" --positions {_q(positions.strip())}"
    if gzip_out:
        cmd += " --gzip"
    return cmd


def build_call_freq(
    input_path: str,
    result_file: str,
    prob_cf: float = 0.5,
    bed: bool = False,
    sort_out: bool = False,
    nproc: int = 1,
    contigs: str = "",
    gzip_out: bool = False,
) -> str:
    cmd = (
        f"deepsignal_plant call_freq"
        f" --input_path {_q(input_path)}"
        f" --result_file {_q(result_file)}"
        f" --prob_cf {prob_cf}"
        f" --nproc {nproc}"
    )
    if bed:
        cmd += " --bed"
    if sort_out:
        cmd += " --sort"
    if gzip_out:
        cmd += " --gzip"
    if contigs.strip():
        cmd += f" --contigs {_q(contigs.strip())}"
    return cmd


def build_split_freq(script_path: str, freqfile: str) -> str:
    return f"python {_q(script_path)} --freqfile {_q(freqfile)}"


def build_extract(
    input_dir: str,
    write_path: str,
    corrected_group: str,
    nproc: int,
    motifs: str,
    bam: str = "",
    basecall_subgroup: str = "BaseCalled_template",
    normalize_method: str = "mad",
    mod_loc: int = 0,
    seq_len: int = 13,
    signal_len: int = 16,
    methy_label: int = 1,
    reference_path: str = "",
    gzip_out: bool = False,
    w_is_dir: bool = False,
    w_batch_num: int = 200,
) -> str:
    cmd = (
        f"deepsignal_plant extract"
        f" -i {_q(input_dir)}"
        f" -o {_q(write_path)}"
        f" --corrected_group {_q(corrected_group)}"
        f" --nproc {nproc}"
        f" --motifs {_q(motifs)}"
        f" --basecall_subgroup {_q(basecall_subgroup)}"
        f" --normalize_method {normalize_method}"
        f" --mod_loc {mod_loc}"
        f" --seq_len {seq_len} --signal_len {signal_len}"
        f" --methy_label {methy_label}"
    )
    if bam.strip():
        cmd += f" --bam {_q(bam.strip())}"
    if reference_path.strip():
        cmd += f" --reference_path {_q(reference_path.strip())}"
    if gzip_out:
        cmd += " --gzip"
    if w_is_dir:
        cmd += f" --w_is_dir yes --w_batch_num {w_batch_num}"
    return cmd


def build_train(
    train_file: str,
    valid_file: str,
    model_dir: str,
    model_type: str = "both_bilstm",
    batch_size: int = 512,
    lr: float = 0.001,
    max_epoch_num: int = 10,
    min_epoch_num: int = 5,
    optim_type: str = "Adam",
    dropout_rate: float = 0.5,
    pos_weight: float = 1.0,
    init_model: str = "",
    seq_len: int = 13,
    signal_len: int = 16,
    hid_rnn: int = 256,
    layernum1: int = 3,
    layernum2: int = 1,
) -> str:
    cmd = (
        f"deepsignal_plant train"
        f" --train_file {_q(train_file)}"
        f" --valid_file {_q(valid_file)}"
        f" --model_dir {_q(model_dir)}"
        f" --model_type {model_type}"
        f" --batch_size {batch_size}"
        f" --lr {lr}"
        f" --max_epoch_num {max_epoch_num}"
        f" --min_epoch_num {min_epoch_num}"
        f" --optim_type {optim_type}"
        f" --dropout_rate {dropout_rate}"
        f" --pos_weight {pos_weight}"
        f" --seq_len {seq_len} --signal_len {signal_len}"
        f" --hid_rnn {hid_rnn}"
        f" --layernum1 {layernum1} --layernum2 {layernum2}"
    )
    if init_model.strip():
        cmd += f" --init_model {_q(init_model.strip())}"
    return cmd


def build_denoise(
    train_file: str,
    model_type: str = "signal_bilstm",
    batch_size: int = 512,
    lr: float = 0.001,
    epoch_num: int = 3,
    iterations: int = 10,
    rounds: int = 3,
    score_cf: float = 0.5,
    kept_ratio: float = 0.99,
    is_filter_fn: bool = False,
) -> str:
    cmd = (
        f"deepsignal_plant denoise"
        f" --train_file {_q(train_file)}"
        f" --model_type {model_type}"
        f" --batch_size {batch_size}"
        f" --lr {lr}"
        f" --epoch_num {epoch_num}"
        f" --iterations {iterations}"
        f" --rounds {rounds}"
        f" --score_cf {score_cf}"
        f" --kept_ratio {kept_ratio}"
    )
    if is_filter_fn:
        cmd += " --is_filter_fn yes"
    return cmd


def build_minimap2(
    reference: str,
    fastq: str,
    output_bam: str,
    threads: int = 12,
) -> str:
    return (
        f"minimap2 -t {threads} -ax map-ont {_q(reference)} {_q(fastq)}"
        f" | samtools sort -o {_q(output_bam)}"
        f" && samtools index {_q(output_bam)}"
    )


# ─── Subprocess execution ─────────────────────────────────────────────────────

def run_command(
    cmd: str,
    cwd: Optional[str] = None,
    line_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[int, str]:
    """
    Run *cmd* in a shell, streaming each output line to *line_callback*.
    Returns (returncode, full_output).
    """
    full: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            full.append(line)
            if line_callback:
                line_callback(line)
        proc.wait()
        return proc.returncode, "".join(full)
    except Exception as exc:
        msg = f"[runner] Error launching process: {exc}\n"
        if line_callback:
            line_callback(msg)
        return -1, msg


# ─── File discovery ───────────────────────────────────────────────────────────

def find_files(directory: str, extensions: List[str], recursive: bool = True) -> List[str]:
    """Return sorted list of file paths matching any of *extensions* under *directory*."""
    p = Path(directory)
    result: list[Path] = []
    glb = p.rglob if recursive else p.glob
    for ext in extensions:
        result.extend(glb(f"*{ext}"))
    return sorted(str(f) for f in result)
