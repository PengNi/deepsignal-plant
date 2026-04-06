"""
app.py — Streamlit Web GUI for DeepSignal-plant.

Launch with:
    streamlit run gui/app.py

Supports R9.4.1 (FAST5 + Tombo) and R10.4.1 (POD5/BAM + Dorado) workflows.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from runner import (
    path_exists, is_dir, disk_free_gb, preflight, run_command,
    build_multi_to_single, build_guppy, build_cat_fastq,
    build_tombo_annotate, build_tombo_resquiggle,
    build_dorado, build_call_mods, build_call_freq, build_split_freq,
    build_extract, build_train, build_denoise, build_minimap2,
)
from file_browser import path_input
from env_manager import (
    wrap_cmd, validate_tool, list_conda_envs, conda_available,
)


# ─── Paths ───────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_REPO = _HERE.parent
_SPLIT_SCRIPT = str(_REPO / "scripts" / "split_freq_file_by_5mC_motif.py")


# ─── Tool list ────────────────────────────────────────────────────────────────
# (tool_name, display_label, typical_env_hint)
_TOOLS: list[tuple[str, str, str]] = [
    ("deepsignal_plant",      "deepsignal_plant",      "deepsignalpenv"),
    ("guppy_basecaller",      "guppy_basecaller",      "deepsignalpenv"),
    ("dorado",                "dorado",                "deepsignalpenv"),
    ("tombo",                 "tombo",                 "deepsignalpenv"),
    ("samtools",              "samtools",              "deepsignalpenv"),
    ("minimap2",              "minimap2",              "deepsignalpenv"),
    ("multi_to_single_fast5", "multi_to_single_fast5", "deepsignalpenv"),
]
_TOOL_NAMES = [t[0] for t in _TOOLS]


# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepSignal-plant GUI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Session-state defaults ──────────────────────────────────────────────────
_TOOL_CFG_DEFAULT = {
    name: {"mode": "system", "value": "", "result": None}
    for name in _TOOL_NAMES
}

_DEFAULTS: dict = {
    # Protocol
    "protocol": "R9.4.1",
    # Common
    "work_dir": os.path.expanduser("~"),
    "reference": "",
    "model_path": "",
    # Tool environments
    "tool_config": _TOOL_CFG_DEFAULT,
    # R9 — paths
    "fast5_multi_dir": "",
    "fast5_single_dir": "",
    "guppy_output_dir": "",
    "guppy_config": "dna_r9.4.1_450bps_hac_prom.cfg",
    "guppy_device": "cuda:all:100%",
    "guppy_min_qscore": 7,
    "guppy_extra": "",
    "fastq_concat": "",
    "basecall_group": "Basecall_1D_000",
    "basecall_subgroup": "BaseCalled_template",
    "corrected_group": "RawGenomeCorrected_000",
    "tombo_procs": 10,
    "tombo_overwrite": True,
    "tombo_failed_reads": "",
    # R10 — paths
    "pod5_dir": "",
    "dorado_model_dir": "",
    "dorado_device": "cuda:all",
    "dorado_bam": "",
    "dorado_extra": "",
    # call_mods
    "cm_input": "",
    "cm_bam": "",
    "cm_result": "",
    "cm_motifs": "C",
    "cm_nproc": 32,
    "cm_nproc_gpu": 2,
    "cm_cuda": "0",
    "cm_batch_size": 512,
    "cm_normalize": "mad",
    "cm_mod_loc": 0,
    "cm_is_dna": True,
    "cm_gzip": False,
    "cm_ref": "",
    "cm_region": "",
    "cm_positions": "",
    "cm_seq_len": 13,
    "cm_signal_len": 16,
    "cm_model_type": "both_bilstm",
    # call_freq
    "cf_result": "",
    "cf_prob_cf": 0.5,
    "cf_nproc": 4,
    "cf_bed": True,
    "cf_sort": True,
    "cf_gzip": False,
    "cf_contigs": "",
    # extract
    "ex_output": "",
    "ex_corrected_group": "RawGenomeCorrected_000",
    "ex_basecall_subgroup": "BaseCalled_template",
    "ex_nproc": 30,
    "ex_motifs": "C",
    "ex_bam": "",
    "ex_normalize": "mad",
    "ex_mod_loc": 0,
    "ex_seq_len": 13,
    "ex_signal_len": 16,
    "ex_methy_label": 1,
    "ex_ref": "",
    "ex_gzip": False,
    "ex_w_is_dir": False,
    "ex_w_batch_num": 200,
    # train
    "tr_train_file": "",
    "tr_valid_file": "",
    "tr_model_dir": "",
    "tr_model_type": "both_bilstm",
    "tr_batch_size": 512,
    "tr_lr": 0.001,
    "tr_max_epoch": 10,
    "tr_min_epoch": 5,
    "tr_optim": "Adam",
    "tr_dropout": 0.5,
    "tr_pos_weight": 1.0,
    "tr_init_model": "",
    # denoise
    "dn_train_file": "",
    "dn_model_type": "signal_bilstm",
    "dn_batch_size": 512,
    "dn_lr": 0.001,
    "dn_epoch_num": 3,
    "dn_iterations": 10,
    "dn_rounds": 3,
    "dn_score_cf": 0.5,
    "dn_kept_ratio": 0.99,
    "dn_filter_fn": False,
    # viz
    "viz_fastq": "",
    "viz_bam_out": "",
    "viz_threads": 12,
    # internal
    "_log": {},
    "_status": {},
    "_fb_active": None,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Ensure every tool key exists (handles app upgrades)
for _name in _TOOL_NAMES:
    st.session_state["tool_config"].setdefault(
        _name, {"mode": "system", "value": "", "result": None}
    )


# ─── Status / log helpers ─────────────────────────────────────────────────────
_ICONS = {"idle": "⬜", "running": "⏳", "done": "✅", "error": "❌"}
_BADGE = {"idle": "idle", "running": "running…", "done": "done", "error": "failed"}


def _status(step: str) -> str:
    return st.session_state["_status"].get(step, "idle")


def _set_status(step: str, s: str) -> None:
    st.session_state["_status"][step] = s


def _log(step: str) -> str:
    return st.session_state["_log"].get(step, "")


def _append_log(step: str, text: str) -> None:
    st.session_state["_log"][step] = st.session_state["_log"].get(step, "") + text


def _clear_log(step: str) -> None:
    st.session_state["_log"][step] = ""


def _step_title(step: str, title: str) -> str:
    st_val = _status(step)
    return f"{_ICONS[st_val]} {title}  [{_BADGE[st_val]}]"


def _run_step(step: str, cmd: str, log_ph) -> bool:
    _clear_log(step)
    _set_status(step, "running")
    _append_log(step, f"$ {cmd}\n{'─'*60}\n")
    log_ph.code(_log(step), language="bash")

    def on_line(line: str) -> None:
        _append_log(step, line)
        log_ph.code(_log(step)[-5000:], language="bash")

    rc, _ = run_command(cmd, line_callback=on_line)
    tail = "\n✅ Completed.\n" if rc == 0 else f"\n❌ Exit code {rc}.\n"
    _append_log(step, tail)
    log_ph.code(_log(step)[-5000:], language="bash")
    _set_status(step, "done" if rc == 0 else "error")
    return rc == 0


def _show_log(step: str) -> None:
    log = _log(step)
    if log:
        st.code(log[-5000:], language="bash")


def _validate_errors(errors: list[str]) -> bool:
    for e in errors:
        st.error(e)
    return bool(errors)


# ─── Tool-env helper ──────────────────────────────────────────────────────────

def _wrap(cmd: str, tool_name: str) -> str:
    """Apply the user-configured environment/path for *tool_name* to *cmd*."""
    cfg = st.session_state["tool_config"].get(tool_name, {})
    return wrap_cmd(cmd, tool_name, cfg.get("mode", "system"), cfg.get("value", ""))


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌱 DeepSignal-plant")
    st.caption("Full-pipeline methylation calling GUI")

    # Protocol
    protocol = st.radio(
        "Sequencing protocol",
        ["R9.4.1 (FAST5 + Tombo)", "R10.4.1 (POD5/BAM + Dorado)"],
        index=0 if st.session_state["protocol"] == "R9.4.1" else 1,
        key="_proto_radio",
    )
    st.session_state["protocol"] = "R9.4.1" if "R9" in protocol else "R10.4.1"

    st.divider()

    # ── Common paths ──────────────────────────────────────────────────────────
    st.subheader("Common settings")
    path_input("Working directory", "work_dir",
               widget_key="si_work_dir",
               help_text="Root directory for output files",
               placeholder="/data/my_experiment",
               select_dirs_only=True)
    path_input("Reference genome (.fa/.fna)", "reference",
               widget_key="si_reference",
               placeholder="/data/ref/genome.fa",
               allowed_exts=[".fa", ".fna", ".fasta"])
    path_input("DeepSignal model (.ckpt)", "model_path",
               widget_key="si_model_path",
               placeholder="/data/models/model.ckpt",
               allowed_exts=[".ckpt"])

    st.divider()

    # ── Tool environments ─────────────────────────────────────────────────────
    with st.expander("🛠️ Tool environments", expanded=False):
        st.caption(
            "**System PATH** uses the current shell · "
            "**Conda env** activates a named env · "
            "**Custom path** uses a full exe path"
        )

        # Check conda availability once
        _has_conda = conda_available()
        if not _has_conda:
            st.warning("`conda` not found — Conda env mode disabled.")

        _conda_envs = list_conda_envs() if _has_conda else []

        for tool_name, tool_label, _hint in _TOOLS:
            cfg = st.session_state["tool_config"][tool_name]

            # ── Tool name ──────────────────────────────────────────────────
            st.markdown(f"**`{tool_label}`**")

            # ── Mode: horizontal radio (full width, short labels) ──────────
            _modes     = ["System", "Conda", "Path"]
            _mode_vals = ["system", "conda", "path"]
            cur_mode_idx = _mode_vals.index(cfg.get("mode", "system"))

            new_mode_label = st.radio(
                "mode",
                _modes,
                index=cur_mode_idx,
                key=f"_tenv_mode_{tool_name}",
                horizontal=True,
                label_visibility="collapsed",
            )
            new_mode = _mode_vals[_modes.index(new_mode_label)]
            cfg["mode"] = new_mode

            # ── Value input (full width, shown only when needed) ────────────
            if new_mode == "conda":
                if _conda_envs:
                    _options = ["(type below…)"] + _conda_envs
                    _sel = st.selectbox(
                        "Conda env",
                        _options,
                        index=_options.index(cfg.get("value", ""))
                              if cfg.get("value", "") in _options else 0,
                        key=f"_tenv_sel_{tool_name}",
                        label_visibility="collapsed",
                    )
                    if _sel != "(type below…)":
                        cfg["value"] = _sel
                new_val = st.text_input(
                    "Conda env name",
                    value=cfg.get("value", ""),
                    key=f"_tenv_val_{tool_name}",
                    placeholder=_hint,
                    label_visibility="collapsed",
                )
                cfg["value"] = new_val
            elif new_mode == "path":
                new_val = st.text_input(
                    "Executable path",
                    value=cfg.get("value", ""),
                    key=f"_tenv_val_{tool_name}",
                    placeholder=f"/path/to/{tool_name}",
                    label_visibility="collapsed",
                )
                cfg["value"] = new_val
            else:
                cfg["value"] = ""

            # ── Test button (full width) + inline status ────────────────────
            tb1, tb2 = st.columns([2, 3])
            with tb1:
                do_test = st.button("🧪 Test", key=f"_tenv_test_{tool_name}",
                                    use_container_width=True)
            with tb2:
                res = cfg.get("result")
                if res is not None:
                    ok, _ = res
                    st.markdown("✅ OK" if ok else "❌ Fail")

            # ── Test result output ──────────────────────────────────────────
            if do_test:
                with st.spinner(f"Testing {tool_name}…"):
                    ok, out = validate_tool(
                        tool_name, cfg["mode"], cfg.get("value", "")
                    )
                cfg["result"] = (ok, out)
                st.caption("✅ OK" if ok else "❌ Failed")
                st.code(out, language="bash")
            elif cfg.get("result") is not None:
                ok, out = cfg["result"]
                with st.expander("Last test output", expanded=False):
                    st.code(out, language="bash")

    # Disk space
    wd = st.session_state.get("work_dir", "")
    if wd and is_dir(wd):
        free = disk_free_gb(wd)
        color = "green" if free > 10 else ("orange" if free > 2 else "red")
        st.caption(f"💾 Free space: :{color}[{free:.1f} GB]")


# ─── Main area ────────────────────────────────────────────────────────────────
st.title("DeepSignal-plant  |  Full-pipeline GUI")

is_r9 = st.session_state["protocol"] == "R9.4.1"
proto_label = "R9.4.1 (FAST5 + Tombo)" if is_r9 else "R10.4.1 (POD5/BAM + Dorado)"
st.info(f"Active protocol: **{proto_label}**")

run_all = st.button("▶️  Run full pipeline", type="primary", use_container_width=True)

tab_pipeline, tab_training, tab_viz, tab_help = st.tabs(
    ["Pipeline", "Training (Stage 5)", "Visualisation", "Help & Troubleshooting"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_pipeline:

    # ── STEP 0 (R9): multi_to_single_fast5 ───────────────────────────────────
    if is_r9:
        with st.expander(
            _step_title("s0_m2s", "Step 1 (Optional) — Convert multi-read FAST5 → single-read FAST5"),
            expanded=False,
        ):
            st.markdown("Skip if your data are already in single-read FAST5 format.")
            col1, col2 = st.columns(2)
            with col1:
                path_input("Input multi-read FAST5 dir", "fast5_multi_dir",
                           widget_key="s0_fast5_multi", placeholder="/data/multi_fast5/",
                           select_dirs_only=True)
            with col2:
                path_input("Output single-read FAST5 dir", "fast5_single_dir",
                           widget_key="s0_fast5_single", placeholder="/data/fast5s/",
                           select_dirs_only=True)
            threads_m2s = st.number_input("Threads", 1, 256, 30, key="_m2s_threads")

            cmd_m2s = _wrap(
                build_multi_to_single(
                    st.session_state["fast5_multi_dir"],
                    st.session_state["fast5_single_dir"],
                    threads_m2s,
                ),
                "multi_to_single_fast5",
            )
            c1, c2 = st.columns([1, 3])
            with c1:
                run_m2s = st.button("▶ Run", key="_run_m2s")
            with c2:
                st.code(cmd_m2s, language="bash")
            log_ph_m2s = st.empty()
            _show_log("s0_m2s")

            if run_m2s or (run_all and _status("s0_m2s") == "idle"):
                errs = preflight([
                    (is_dir(st.session_state["fast5_multi_dir"]),
                     "Input multi-read FAST5 directory does not exist."),
                    (bool(st.session_state["fast5_single_dir"]),
                     "Output single-read FAST5 directory is required."),
                ])
                if not _validate_errors(errs):
                    os.makedirs(st.session_state["fast5_single_dir"], exist_ok=True)
                    _run_step("s0_m2s", cmd_m2s, log_ph_m2s)

    # ── STEP 1: Basecalling ───────────────────────────────────────────────────
    step_label_bc = "Step 2 — Basecalling (Guppy)" if is_r9 else "Step 1 — Basecalling (Dorado)"
    with st.expander(_step_title("s1_bc", step_label_bc), expanded=True):
        if is_r9:
            col1, col2 = st.columns(2)
            with col1:
                path_input("FAST5 input directory", "fast5_single_dir",
                           widget_key="s1_fast5_single", placeholder="/data/fast5s/",
                           select_dirs_only=True)
                path_input("Guppy output directory", "guppy_output_dir",
                           widget_key="s1_guppy_out", placeholder="/data/fast5s_guppy/",
                           select_dirs_only=True)
            with col2:
                _guppy_configs = [
                    "dna_r9.4.1_450bps_hac_prom.cfg",
                    "dna_r9.4.1_450bps_hac.cfg",
                    "dna_r9.4.1_450bps_fast.cfg",
                    "dna_r9.4.1_450bps_sup.cfg",
                    "dna_r9.4.1_e8.1_hac.cfg",
                ]
                guppy_cfg_sel = st.selectbox(
                    "Guppy config", _guppy_configs + ["custom…"],
                    index=0, key="_guppy_cfg_sel",
                )
                if guppy_cfg_sel == "custom…":
                    path_input("Custom Guppy config", "guppy_config",
                               widget_key="s1_guppy_cfg",
                               placeholder="dna_r9.4.1_450bps_hac_prom.cfg",
                               allowed_exts=[".cfg"])
                else:
                    st.session_state["guppy_config"] = guppy_cfg_sel
                st.session_state["guppy_device"] = st.text_input(
                    "Device", value=st.session_state["guppy_device"],
                    key="_guppy_dev", help="e.g. cuda:all:100%  or  cpu")
                st.session_state["guppy_min_qscore"] = st.number_input(
                    "Min Q-score", 0, 30, int(st.session_state["guppy_min_qscore"]), key="_guppy_qs")
                st.session_state["guppy_extra"] = st.text_input(
                    "Extra Guppy args", value=st.session_state.get("guppy_extra", ""), key="_guppy_extra")

            cmd_bc = _wrap(
                build_guppy(
                    st.session_state["fast5_single_dir"],
                    st.session_state["guppy_output_dir"],
                    st.session_state["guppy_config"],
                    st.session_state["guppy_device"],
                    int(st.session_state["guppy_min_qscore"]),
                    st.session_state["guppy_extra"],
                ),
                "guppy_basecaller",
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                path_input("POD5 input directory", "pod5_dir",
                           widget_key="s1_pod5", placeholder="/data/pod5/",
                           select_dirs_only=True)
                path_input("Dorado model directory", "dorado_model_dir",
                           widget_key="s1_dorado_model",
                           placeholder="/data/dorado_models/dna_r10.4.1_e8.2_400bps_hac@v4.2.0",
                           select_dirs_only=True)
            with col2:
                path_input("Reference genome (for Dorado)", "reference",
                           widget_key="s1_reference", placeholder="/data/ref/genome.fa",
                           allowed_exts=[".fa", ".fna", ".fasta"])
                path_input("Output BAM", "dorado_bam",
                           widget_key="s1_dorado_bam", placeholder="/data/demo.bam",
                           allowed_exts=[".bam"])
                st.session_state["dorado_device"] = st.selectbox(
                    "Device", ["cuda:all", "cuda:0", "cuda:0,1", "cpu"], key="_dor_dev")
                st.session_state["dorado_extra"] = st.text_input(
                    "Extra Dorado args", value=st.session_state.get("dorado_extra", ""), key="_dor_extra")

            cmd_bc = _wrap(
                build_dorado(
                    st.session_state["dorado_model_dir"],
                    st.session_state["pod5_dir"],
                    st.session_state["reference"],
                    st.session_state["dorado_bam"],
                    st.session_state["dorado_device"],
                    st.session_state["dorado_extra"],
                ),
                "dorado",
            )

        c1, c2 = st.columns([1, 3])
        with c1:
            run_bc = st.button("▶ Run", key="_run_bc")
        with c2:
            st.code(cmd_bc, language="bash")
        log_ph_bc = st.empty()
        _show_log("s1_bc")

        if run_bc or (run_all and _status("s1_bc") == "idle"):
            if is_r9:
                errs = preflight([
                    (is_dir(st.session_state["fast5_single_dir"]),
                     "FAST5 input directory does not exist."),
                    (bool(st.session_state["guppy_output_dir"]),
                     "Guppy output directory is required."),
                ])
            else:
                errs = preflight([
                    (is_dir(st.session_state["pod5_dir"]),
                     "POD5 input directory does not exist."),
                    (path_exists(st.session_state["reference"]),
                     "Reference genome file does not exist."),
                    (bool(st.session_state["dorado_bam"]),
                     "Output BAM path is required."),
                ])
            if not _validate_errors(errs):
                if is_r9:
                    os.makedirs(st.session_state["guppy_output_dir"], exist_ok=True)
                _run_step("s1_bc", cmd_bc, log_ph_bc)

    # ── STEP 2 (R9): Concatenate FASTQs ──────────────────────────────────────
    if is_r9:
        with st.expander(_step_title("s2_cat", "Step 3 — Concatenate FASTQ files"), expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                path_input("Guppy output directory", "guppy_output_dir",
                           widget_key="s2_guppy_out", placeholder="/data/fast5s_guppy/",
                           select_dirs_only=True)
            with col2:
                path_input("Concatenated FASTQ output", "fastq_concat",
                           widget_key="s2_fastq_concat", placeholder="/data/fast5s_guppy.fastq",
                           allowed_exts=[".fastq", ".fq", ".fastq.gz", ".fq.gz"])

            # cat is a shell builtin — no tool-env wrapping
            cmd_cat = build_cat_fastq(
                st.session_state["guppy_output_dir"],
                st.session_state["fastq_concat"],
            )
            c1, c2 = st.columns([1, 3])
            with c1:
                run_cat = st.button("▶ Run", key="_run_cat")
            with c2:
                st.code(cmd_cat, language="bash")
            log_ph_cat = st.empty()
            _show_log("s2_cat")

            if run_cat or (run_all and _status("s2_cat") == "idle"):
                errs = preflight([
                    (is_dir(st.session_state["guppy_output_dir"]),
                     "Guppy output directory does not exist."),
                    (bool(st.session_state["fastq_concat"]),
                     "Output FASTQ path is required."),
                ])
                if not _validate_errors(errs):
                    _run_step("s2_cat", cmd_cat, log_ph_cat)

    # ── STEP 3 (R9): Tombo annotate ──────────────────────────────────────────
    if is_r9:
        with st.expander(_step_title("s3_anno", "Step 4 — Tombo: annotate raw reads with FASTQs"), expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                path_input("FAST5 base directory", "fast5_single_dir",
                           widget_key="s3_fast5_single", placeholder="/data/fast5s/",
                           select_dirs_only=True)
                path_input("Concatenated FASTQ", "fastq_concat",
                           widget_key="s3_fastq_concat", placeholder="/data/fast5s_guppy.fastq",
                           allowed_exts=[".fastq", ".fq", ".fastq.gz", ".fq.gz"])
            with col2:
                st.session_state["basecall_group"] = st.text_input(
                    "Basecall group", st.session_state["basecall_group"],
                    key="_tombo_bcg",
                    help="Must match Guppy's output group (default: Basecall_1D_000)")
                st.session_state["basecall_subgroup"] = st.text_input(
                    "Basecall subgroup", st.session_state["basecall_subgroup"],
                    key="_tombo_bcsg")
                st.session_state["tombo_procs"] = st.number_input(
                    "Processes", 1, 256, int(st.session_state["tombo_procs"]), key="_tombo_p")

            summary_path = st.text_input(
                "Sequencing summary file",
                value=os.path.join(st.session_state.get("guppy_output_dir", ""),
                                   "sequencing_summary.txt"),
                key="_tombo_summ",
            )
            cmd_anno = _wrap(
                build_tombo_annotate(
                    st.session_state["fast5_single_dir"],
                    st.session_state["fastq_concat"],
                    summary_path,
                    st.session_state["basecall_group"],
                    st.session_state["basecall_subgroup"],
                    int(st.session_state["tombo_procs"]),
                    True,
                ),
                "tombo",
            )
            c1, c2 = st.columns([1, 3])
            with c1:
                run_anno = st.button("▶ Run", key="_run_anno")
            with c2:
                st.code(cmd_anno, language="bash")
            log_ph_anno = st.empty()
            _show_log("s3_anno")

            if run_anno or (run_all and _status("s3_anno") == "idle"):
                errs = preflight([
                    (is_dir(st.session_state["fast5_single_dir"]),
                     "FAST5 base directory does not exist."),
                    (path_exists(st.session_state["fastq_concat"]),
                     "Concatenated FASTQ file does not exist."),
                    (path_exists(summary_path),
                     "Sequencing summary file does not exist."),
                ])
                if not _validate_errors(errs):
                    _run_step("s3_anno", cmd_anno, log_ph_anno)

    # ── STEP 4 (R9): Tombo resquiggle ────────────────────────────────────────
    if is_r9:
        with st.expander(_step_title("s4_rsq", "Step 5 — Tombo: re-squiggle"), expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                path_input("FAST5 directory", "fast5_single_dir",
                           widget_key="s4_fast5_single", placeholder="/data/fast5s/",
                           select_dirs_only=True)
                path_input("Reference genome", "reference",
                           widget_key="s4_reference", placeholder="/data/ref/genome.fa",
                           allowed_exts=[".fa", ".fna", ".fasta"])
            with col2:
                st.session_state["corrected_group"] = st.text_input(
                    "Corrected group", st.session_state["corrected_group"],
                    key="_rsq_cg",
                    help="Must be identical to --corrected_group in call_mods.")
                st.session_state["basecall_group"] = st.text_input(
                    "Basecall group", st.session_state["basecall_group"], key="_rsq_bcg")
                st.session_state["tombo_procs"] = st.number_input(
                    "Processes", 1, 256, int(st.session_state["tombo_procs"]), key="_rsq_p")
                st.session_state["tombo_overwrite"] = st.checkbox(
                    "Overwrite existing results",
                    bool(st.session_state["tombo_overwrite"]), key="_rsq_ow")
                path_input("Failed reads output (optional)", "tombo_failed_reads",
                           widget_key="s4_failed_reads", placeholder="/data/failed_reads.txt")

            cmd_rsq = _wrap(
                build_tombo_resquiggle(
                    st.session_state["fast5_single_dir"],
                    st.session_state["reference"],
                    st.session_state["corrected_group"],
                    st.session_state["basecall_group"],
                    int(st.session_state["tombo_procs"]),
                    bool(st.session_state["tombo_overwrite"]),
                    st.session_state["tombo_failed_reads"],
                ),
                "tombo",
            )
            c1, c2 = st.columns([1, 3])
            with c1:
                run_rsq = st.button("▶ Run", key="_run_rsq")
            with c2:
                st.code(cmd_rsq, language="bash")
            log_ph_rsq = st.empty()
            _show_log("s4_rsq")

            if run_rsq or (run_all and _status("s4_rsq") == "idle"):
                errs = preflight([
                    (is_dir(st.session_state["fast5_single_dir"]),
                     "FAST5 directory does not exist."),
                    (path_exists(st.session_state["reference"]),
                     "Reference genome file does not exist."),
                ])
                if not _validate_errors(errs):
                    _run_step("s4_rsq", cmd_rsq, log_ph_rsq)

    # ── STEP 5: call_mods ─────────────────────────────────────────────────────
    step_idx_cm = 6 if is_r9 else 2
    with st.expander(_step_title("s5_cm", f"Step {step_idx_cm} — DeepSignal-plant: call_mods"), expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Input / Output**")
            if is_r9:
                path_input("FAST5 input directory (or feature file)", "cm_input",
                           widget_key="s5_cm_input_r9", placeholder="/data/fast5s/",
                           allowed_exts=[".tsv", ".gz"])   # dir OR feature file
            else:
                path_input("POD5 input directory", "cm_input",
                           widget_key="s5_cm_input_r10", placeholder="/data/pod5/",
                           select_dirs_only=True)
                path_input("BAM file (from Dorado)", "cm_bam",
                           widget_key="s5_cm_bam", placeholder="/data/demo.bam",
                           allowed_exts=[".bam"])
            path_input("Model path (.ckpt)", "model_path",
                       widget_key="s5_model_path", placeholder="/data/models/model.ckpt",
                       allowed_exts=[".ckpt"])
            path_input("Result file", "cm_result",
                       widget_key="s5_cm_result", placeholder="/data/fast5s.C.call_mods.tsv")

        with col2:
            st.markdown("**Methylation**")
            _motif_opts = ["C", "CG", "CHG", "CHH", "CG,CHG,CHH"]
            motif_sel = st.selectbox(
                "Motifs", _motif_opts + ["custom…"],
                index=_motif_opts.index(st.session_state["cm_motifs"])
                if st.session_state["cm_motifs"] in _motif_opts else len(_motif_opts),
                key="_cm_motif_sel",
            )
            if motif_sel == "custom…":
                st.session_state["cm_motifs"] = st.text_input(
                    "Custom motif(s)", st.session_state["cm_motifs"], key="_cm_motif_in")
            else:
                st.session_state["cm_motifs"] = motif_sel

            st.session_state["cm_mod_loc"] = st.number_input(
                "mod_loc (0-based)", 0, 12, int(st.session_state["cm_mod_loc"]), key="_cm_ml")
            if is_r9:
                st.session_state["corrected_group"] = st.text_input(
                    "corrected_group", st.session_state["corrected_group"],
                    key="_cm_cg", help="Must match --corrected-group used in Tombo re-squiggle")
                st.session_state["basecall_subgroup"] = st.text_input(
                    "basecall_subgroup", st.session_state["basecall_subgroup"], key="_cm_bsg")
            st.session_state["cm_normalize"] = st.selectbox(
                "Normalize method", ["mad", "zscore"],
                index=0 if st.session_state["cm_normalize"] == "mad" else 1, key="_cm_norm")

        with col3:
            st.markdown("**Performance**")
            st.session_state["cm_cuda"] = st.text_input(
                "CUDA_VISIBLE_DEVICES", st.session_state["cm_cuda"], key="_cm_cuda",
                help="e.g. 0  or  0,1  or  -1 for CPU-only")
            st.session_state["cm_nproc"] = st.number_input(
                "nproc (CPU processes)", 1, 256, int(st.session_state["cm_nproc"]), key="_cm_np")
            st.session_state["cm_nproc_gpu"] = st.number_input(
                "nproc_gpu", 0, 64, int(st.session_state["cm_nproc_gpu"]), key="_cm_npg")
            st.session_state["cm_batch_size"] = st.number_input(
                "Batch size", 1, 8192, int(st.session_state["cm_batch_size"]), key="_cm_bs")
            st.session_state["cm_model_type"] = st.selectbox(
                "Model type",
                ["both_bilstm", "seq_bilstm", "signal_bilstm"],
                index=["both_bilstm", "seq_bilstm", "signal_bilstm"].index(
                    st.session_state["cm_model_type"]),
                key="_cm_mt")
            st.session_state["cm_seq_len"] = st.number_input(
                "seq_len (kmer)", 1, 32, int(st.session_state["cm_seq_len"]), key="_cm_sl")
            st.session_state["cm_signal_len"] = st.number_input(
                "signal_len", 1, 64, int(st.session_state["cm_signal_len"]), key="_cm_sgl")

        with st.expander("Advanced options", expanded=False):
            ac1, ac2 = st.columns(2)
            with ac1:
                path_input("Reference path (optional)", "cm_ref", widget_key="s5_cm_ref",
                           allowed_exts=[".fa", ".fna", ".fasta"])
                st.session_state["cm_region"] = st.text_input(
                    "Region (optional)", st.session_state.get("cm_region", ""),
                    key="_cm_reg", placeholder="chr1:0-1000000")
            with ac2:
                path_input("Positions file (optional)", "cm_positions", widget_key="s5_cm_positions",
                           allowed_exts=[".tsv", ".txt", ".bed"])
                st.session_state["cm_is_dna"] = st.checkbox(
                    "DNA sample (uncheck for RNA)", bool(st.session_state["cm_is_dna"]), key="_cm_dna")
                st.session_state["cm_gzip"] = st.checkbox(
                    "Gzip output", bool(st.session_state["cm_gzip"]), key="_cm_gz")

        cmd_cm = _wrap(
            build_call_mods(
                input_path=st.session_state["cm_input"],
                model_path=st.session_state["model_path"],
                result_file=st.session_state["cm_result"],
                nproc=int(st.session_state["cm_nproc"]),
                nproc_gpu=int(st.session_state["cm_nproc_gpu"]),
                motifs=st.session_state["cm_motifs"],
                cuda_devices=st.session_state["cm_cuda"],
                batch_size=int(st.session_state["cm_batch_size"]),
                bam=st.session_state.get("cm_bam", "") if not is_r9 else "",
                corrected_group=st.session_state["corrected_group"] if is_r9 else "",
                basecall_subgroup=st.session_state["basecall_subgroup"],
                normalize_method=st.session_state["cm_normalize"],
                mod_loc=int(st.session_state["cm_mod_loc"]),
                is_dna=bool(st.session_state["cm_is_dna"]),
                reference_path=st.session_state.get("cm_ref", ""),
                region=st.session_state.get("cm_region", ""),
                positions=st.session_state.get("cm_positions", ""),
                gzip_out=bool(st.session_state["cm_gzip"]),
                seq_len=int(st.session_state["cm_seq_len"]),
                signal_len=int(st.session_state["cm_signal_len"]),
                model_type=st.session_state["cm_model_type"],
            ),
            "deepsignal_plant",
        )

        c1, c2 = st.columns([1, 3])
        with c1:
            run_cm = st.button("▶ Run", key="_run_cm")
        with c2:
            st.code(cmd_cm, language="bash")
        log_ph_cm = st.empty()
        _show_log("s5_cm")

        if run_cm or (run_all and _status("s5_cm") == "idle"):
            _input_ok = (is_dir(st.session_state["cm_input"])
                         or path_exists(st.session_state["cm_input"]))
            errs = preflight([
                (_input_ok, "Input path does not exist."),
                (path_exists(st.session_state["model_path"]), "Model file does not exist."),
                (bool(st.session_state["cm_result"]), "Result file path is required."),
            ])
            if not is_r9:
                errs += preflight([
                    (path_exists(st.session_state.get("cm_bam", "")), "BAM file does not exist."),
                ])
            if not _validate_errors(errs):
                _run_step("s5_cm", cmd_cm, log_ph_cm)

    # ── STEP 6: call_freq ─────────────────────────────────────────────────────
    step_idx_cf = 7 if is_r9 else 3
    with st.expander(_step_title("s6_cf", f"Step {step_idx_cf} — DeepSignal-plant: call_freq"), expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            path_input("call_mods output (TSV)", "cm_result",
                       widget_key="s6_cm_result", placeholder="/data/fast5s.C.call_mods.tsv",
                       allowed_exts=[".tsv", ".gz"])
            path_input("Frequency output file", "cf_result",
                       widget_key="s6_cf_result",
                       placeholder="/data/fast5s.C.call_mods.frequency.tsv")
        with col2:
            st.session_state["cf_prob_cf"] = st.slider(
                "prob_cf (confidence cutoff)", 0.0, 1.0,
                float(st.session_state["cf_prob_cf"]), 0.05, key="_cf_pcf",
                help="Remove calls where |prob1-prob0| < prob_cf. 0 = use all calls.")
            st.session_state["cf_bed"] = st.checkbox(
                "Output bedMethyl format", bool(st.session_state["cf_bed"]), key="_cf_bed")
            st.session_state["cf_sort"] = st.checkbox(
                "Sort output", bool(st.session_state["cf_sort"]), key="_cf_sort")
            st.session_state["cf_nproc"] = st.number_input(
                "nproc", 1, 256, int(st.session_state["cf_nproc"]), key="_cf_np")
            st.session_state["cf_gzip"] = st.checkbox(
                "Gzip output", bool(st.session_state["cf_gzip"]), key="_cf_gz")
            path_input("Contigs file/string (optional)", "cf_contigs",
                       widget_key="s6_cf_contigs",
                       help_text="Reference .fa or comma-separated contig names for parallel mode",
                       allowed_exts=[".fa", ".fna", ".fasta", ".txt"])

        cmd_cf = _wrap(
            build_call_freq(
                input_path=st.session_state["cm_result"],
                result_file=st.session_state["cf_result"],
                prob_cf=float(st.session_state["cf_prob_cf"]),
                bed=bool(st.session_state["cf_bed"]),
                sort_out=bool(st.session_state["cf_sort"]),
                nproc=int(st.session_state["cf_nproc"]),
                contigs=st.session_state.get("cf_contigs", ""),
                gzip_out=bool(st.session_state["cf_gzip"]),
            ),
            "deepsignal_plant",
        )
        c1, c2 = st.columns([1, 3])
        with c1:
            run_cf = st.button("▶ Run", key="_run_cf")
        with c2:
            st.code(cmd_cf, language="bash")
        log_ph_cf = st.empty()
        _show_log("s6_cf")

        if run_cf or (run_all and _status("s6_cf") == "idle"):
            errs = preflight([
                (path_exists(st.session_state["cm_result"]),
                 "call_mods output file does not exist."),
                (bool(st.session_state["cf_result"]),
                 "Frequency output file path is required."),
            ])
            if not _validate_errors(errs):
                _run_step("s6_cf", cmd_cf, log_ph_cf)

    # ── STEP 7: split by context ──────────────────────────────────────────────
    step_idx_sp = 8 if is_r9 else 4
    with st.expander(
        _step_title("s7_split", f"Step {step_idx_sp} (Optional) — Split frequency by methylation context"),
        expanded=False,
    ):
        st.markdown(
            "Splits the frequency file into: `*.CG.tsv`, `*.CHG.tsv`, `*.CHH.tsv`."
        )
        path_input("Frequency file to split", "cf_result",
                   widget_key="s7_cf_result",
                   placeholder="/data/fast5s.C.call_mods.frequency.tsv",
                   allowed_exts=[".tsv", ".gz"])

        if not path_exists(_SPLIT_SCRIPT):
            st.warning(f"Split script not found at `{_SPLIT_SCRIPT}`.")
            custom_script = st.text_input("Script path", _SPLIT_SCRIPT, key="_split_script")
        else:
            custom_script = _SPLIT_SCRIPT

        cmd_split = _wrap(
            build_split_freq(custom_script, st.session_state["cf_result"]),
            "python",
        )
        c1, c2 = st.columns([1, 3])
        with c1:
            run_split = st.button("▶ Run", key="_run_split")
        with c2:
            st.code(cmd_split, language="bash")
        log_ph_split = st.empty()
        _show_log("s7_split")

        if run_split or (run_all and _status("s7_split") == "idle"):
            errs = preflight([
                (path_exists(st.session_state["cf_result"]), "Frequency file does not exist."),
                (path_exists(custom_script), f"Split script not found: {custom_script}"),
            ])
            if not _validate_errors(errs):
                _run_step("s7_split", cmd_split, log_ph_split)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING (Stage 5)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_training:
    st.markdown(
        "Custom model training is optional. "
        "Requires paired Nanopore + bisulfite sequencing data."
    )

    # ── Extract features ──────────────────────────────────────────────────────
    with st.expander(_step_title("t1_ex", "Step T1 — Extract features"), expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            path_input("FAST5/POD5 input directory", "cm_input",
                       widget_key="t1_cm_input", placeholder="/data/fast5s/",
                       select_dirs_only=True)
            if not is_r9:
                path_input("BAM file (POD5 mode)", "ex_bam",
                           widget_key="t1_ex_bam", placeholder="/data/demo.bam",
                           allowed_exts=[".bam"])
            path_input("Output features file", "ex_output",
                       widget_key="t1_ex_output", placeholder="/data/features.C.tsv")
        with col2:
            st.session_state["ex_corrected_group"] = st.text_input(
                "corrected_group", st.session_state["ex_corrected_group"], key="_ex_cg")
            st.session_state["ex_basecall_subgroup"] = st.text_input(
                "basecall_subgroup", st.session_state["ex_basecall_subgroup"], key="_ex_bsg")
            st.session_state["ex_nproc"] = st.number_input(
                "nproc", 1, 256, int(st.session_state["ex_nproc"]), key="_ex_np")
            st.session_state["ex_motifs"] = st.text_input(
                "motifs", st.session_state["ex_motifs"], key="_ex_m")
        with col3:
            st.session_state["ex_methy_label"] = st.selectbox(
                "methy_label", [1, 0],
                index=0 if st.session_state["ex_methy_label"] == 1 else 1, key="_ex_ml")
            st.session_state["ex_normalize"] = st.selectbox(
                "normalize_method", ["mad", "zscore"], key="_ex_norm")
            st.session_state["ex_mod_loc"] = st.number_input(
                "mod_loc", 0, 12, int(st.session_state["ex_mod_loc"]), key="_ex_modl")
            st.session_state["ex_seq_len"] = st.number_input(
                "seq_len", 1, 32, int(st.session_state["ex_seq_len"]), key="_ex_sl")
            st.session_state["ex_signal_len"] = st.number_input(
                "signal_len", 1, 64, int(st.session_state["ex_signal_len"]), key="_ex_sgl")
            st.session_state["ex_gzip"] = st.checkbox(
                "Gzip output", bool(st.session_state["ex_gzip"]), key="_ex_gz")
            st.session_state["ex_w_is_dir"] = st.checkbox(
                "Write to directory", bool(st.session_state["ex_w_is_dir"]), key="_ex_wid")
            if st.session_state["ex_w_is_dir"]:
                st.session_state["ex_w_batch_num"] = st.number_input(
                    "Batch num per file", 1, 10000,
                    int(st.session_state["ex_w_batch_num"]), key="_ex_wbn")

        cmd_ex = _wrap(
            build_extract(
                input_dir=st.session_state["cm_input"],
                write_path=st.session_state["ex_output"],
                corrected_group=st.session_state["ex_corrected_group"],
                nproc=int(st.session_state["ex_nproc"]),
                motifs=st.session_state["ex_motifs"],
                bam=st.session_state.get("ex_bam", ""),
                basecall_subgroup=st.session_state["ex_basecall_subgroup"],
                normalize_method=st.session_state["ex_normalize"],
                mod_loc=int(st.session_state["ex_mod_loc"]),
                seq_len=int(st.session_state["ex_seq_len"]),
                signal_len=int(st.session_state["ex_signal_len"]),
                methy_label=int(st.session_state["ex_methy_label"]),
                gzip_out=bool(st.session_state["ex_gzip"]),
                w_is_dir=bool(st.session_state["ex_w_is_dir"]),
                w_batch_num=int(st.session_state.get("ex_w_batch_num", 200)),
            ),
            "deepsignal_plant",
        )
        c1, c2 = st.columns([1, 3])
        with c1:
            run_ex = st.button("▶ Run", key="_run_ex")
        with c2:
            st.code(cmd_ex, language="bash")
        log_ph_ex = st.empty()
        _show_log("t1_ex")

        if run_ex:
            errs = preflight([
                (is_dir(st.session_state["cm_input"]) or path_exists(st.session_state["cm_input"]),
                 "Input path does not exist."),
                (bool(st.session_state["ex_output"]), "Output path is required."),
            ])
            if not _validate_errors(errs):
                _run_step("t1_ex", cmd_ex, log_ph_ex)

    # ── Denoise ───────────────────────────────────────────────────────────────
    with st.expander(_step_title("t2_dn", "Step T2 — Denoise training samples"), expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            path_input("Training file (combined +/- samples)", "dn_train_file",
                       widget_key="t2_dn_train", placeholder="/data/train_combined.tsv",
                       allowed_exts=[".tsv", ".gz"])
            st.session_state["dn_model_type"] = st.selectbox(
                "Model type", ["signal_bilstm", "both_bilstm", "seq_bilstm"], key="_dn_mt")
            st.session_state["dn_batch_size"] = st.number_input(
                "Batch size", 1, 8192, int(st.session_state["dn_batch_size"]), key="_dn_bs")
            st.session_state["dn_epoch_num"] = st.number_input(
                "Epoch num", 1, 100, int(st.session_state["dn_epoch_num"]), key="_dn_en")
        with col2:
            st.session_state["dn_lr"] = st.number_input(
                "Learning rate", 1e-6, 1.0, float(st.session_state["dn_lr"]),
                format="%.4f", key="_dn_lr")
            st.session_state["dn_iterations"] = st.number_input(
                "Iterations", 1, 100, int(st.session_state["dn_iterations"]), key="_dn_it")
            st.session_state["dn_rounds"] = st.number_input(
                "Rounds", 1, 20, int(st.session_state["dn_rounds"]), key="_dn_r")
            st.session_state["dn_score_cf"] = st.slider(
                "Score cutoff", 0.0, 0.5, float(st.session_state["dn_score_cf"]),
                0.05, key="_dn_scf")
            st.session_state["dn_kept_ratio"] = st.slider(
                "Kept ratio", 0.5, 1.0, float(st.session_state["dn_kept_ratio"]),
                0.01, key="_dn_kr")
            st.session_state["dn_filter_fn"] = st.checkbox(
                "Filter false negatives", bool(st.session_state["dn_filter_fn"]), key="_dn_ffn")

        cmd_dn = _wrap(
            build_denoise(
                train_file=st.session_state["dn_train_file"],
                model_type=st.session_state["dn_model_type"],
                batch_size=int(st.session_state["dn_batch_size"]),
                lr=float(st.session_state["dn_lr"]),
                epoch_num=int(st.session_state["dn_epoch_num"]),
                iterations=int(st.session_state["dn_iterations"]),
                rounds=int(st.session_state["dn_rounds"]),
                score_cf=float(st.session_state["dn_score_cf"]),
                kept_ratio=float(st.session_state["dn_kept_ratio"]),
                is_filter_fn=bool(st.session_state["dn_filter_fn"]),
            ),
            "deepsignal_plant",
        )
        c1, c2 = st.columns([1, 3])
        with c1:
            run_dn = st.button("▶ Run", key="_run_dn")
        with c2:
            st.code(cmd_dn, language="bash")
        log_ph_dn = st.empty()
        _show_log("t2_dn")

        if run_dn:
            errs = preflight([
                (path_exists(st.session_state["dn_train_file"]), "Training file does not exist."),
            ])
            if not _validate_errors(errs):
                _run_step("t2_dn", cmd_dn, log_ph_dn)

    # ── Train ─────────────────────────────────────────────────────────────────
    with st.expander(_step_title("t3_tr", "Step T3 — Train model"), expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            path_input("Training file", "tr_train_file",
                       widget_key="t3_tr_train", placeholder="/data/train.tsv",
                       allowed_exts=[".tsv", ".gz"])
            path_input("Validation file", "tr_valid_file",
                       widget_key="t3_tr_valid", placeholder="/data/valid.tsv",
                       allowed_exts=[".tsv", ".gz"])
            path_input("Model output directory", "tr_model_dir",
                       widget_key="t3_tr_model_dir", placeholder="/data/my_model/",
                       select_dirs_only=True)
            path_input("Init model (.ckpt, optional)", "tr_init_model",
                       widget_key="t3_tr_init", allowed_exts=[".ckpt"])
        with col2:
            st.session_state["tr_model_type"] = st.selectbox(
                "Model type", ["both_bilstm", "seq_bilstm", "signal_bilstm"], key="_tr_mt")
            st.session_state["tr_optim"] = st.selectbox(
                "Optimizer", ["Adam", "RMSprop", "SGD", "Ranger"], key="_tr_opt")
            st.session_state["tr_batch_size"] = st.number_input(
                "Batch size", 1, 8192, int(st.session_state["tr_batch_size"]), key="_tr_bs")
            st.session_state["tr_lr"] = st.number_input(
                "Learning rate", 1e-6, 1.0, float(st.session_state["tr_lr"]),
                format="%.4f", key="_tr_lr")
        with col3:
            st.session_state["tr_max_epoch"] = st.number_input(
                "Max epochs", 1, 1000, int(st.session_state["tr_max_epoch"]), key="_tr_max")
            st.session_state["tr_min_epoch"] = st.number_input(
                "Min epochs", 1, 1000, int(st.session_state["tr_min_epoch"]), key="_tr_min")
            st.session_state["tr_dropout"] = st.slider(
                "Dropout rate", 0.0, 1.0, float(st.session_state["tr_dropout"]), key="_tr_dr")
            st.session_state["tr_pos_weight"] = st.number_input(
                "Positive weight", 0.1, 10.0, float(st.session_state["tr_pos_weight"]),
                step=0.1, key="_tr_pw")

        cmd_tr = _wrap(
            build_train(
                train_file=st.session_state["tr_train_file"],
                valid_file=st.session_state["tr_valid_file"],
                model_dir=st.session_state["tr_model_dir"],
                model_type=st.session_state["tr_model_type"],
                batch_size=int(st.session_state["tr_batch_size"]),
                lr=float(st.session_state["tr_lr"]),
                max_epoch_num=int(st.session_state["tr_max_epoch"]),
                min_epoch_num=int(st.session_state["tr_min_epoch"]),
                optim_type=st.session_state["tr_optim"],
                dropout_rate=float(st.session_state["tr_dropout"]),
                pos_weight=float(st.session_state["tr_pos_weight"]),
                init_model=st.session_state.get("tr_init_model", ""),
            ),
            "deepsignal_plant",
        )
        c1, c2 = st.columns([1, 3])
        with c1:
            run_tr = st.button("▶ Run", key="_run_tr")
        with c2:
            st.code(cmd_tr, language="bash")
        log_ph_tr = st.empty()
        _show_log("t3_tr")

        if run_tr:
            errs = preflight([
                (path_exists(st.session_state["tr_train_file"]), "Training file does not exist."),
                (path_exists(st.session_state["tr_valid_file"]), "Validation file does not exist."),
                (bool(st.session_state["tr_model_dir"]), "Model output directory is required."),
            ])
            if not _validate_errors(errs):
                os.makedirs(st.session_state["tr_model_dir"], exist_ok=True)
                _run_step("t3_tr", cmd_tr, log_ph_tr)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

with tab_viz:
    st.markdown("Align reads to the reference genome for genome-browser visualisation (IGV, UCSC).")

    with st.expander(_step_title("v1_mm2", "Step V1 (Optional) — Align reads with minimap2"), expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            path_input("Reference genome", "reference",
                       widget_key="v1_reference", placeholder="/data/ref/genome.fa",
                       allowed_exts=[".fa", ".fna", ".fasta"])
            path_input("Input FASTQ", "viz_fastq",
                       widget_key="v1_viz_fastq", placeholder="/data/reads.fastq",
                       allowed_exts=[".fastq", ".fq", ".fastq.gz", ".fq.gz"])
        with col2:
            path_input("Output BAM", "viz_bam_out",
                       widget_key="v1_viz_bam", placeholder="/data/reads.sorted.bam")
            st.session_state["viz_threads"] = st.number_input(
                "Threads", 1, 256, int(st.session_state["viz_threads"]), key="_viz_t")

        cmd_mm = _wrap(
            build_minimap2(
                reference=st.session_state["reference"],
                fastq=st.session_state["viz_fastq"],
                output_bam=st.session_state["viz_bam_out"],
                threads=int(st.session_state["viz_threads"]),
            ),
            "minimap2",
        )
        c1, c2 = st.columns([1, 3])
        with c1:
            run_mm = st.button("▶ Run", key="_run_mm")
        with c2:
            st.code(cmd_mm, language="bash")
        log_ph_mm = st.empty()
        _show_log("v1_mm2")

        if run_mm:
            errs = preflight([
                (path_exists(st.session_state["reference"]), "Reference genome does not exist."),
                (path_exists(st.session_state["viz_fastq"]), "FASTQ file does not exist."),
                (bool(st.session_state["viz_bam_out"]), "Output BAM path is required."),
            ])
            if not _validate_errors(errs):
                _run_step("v1_mm2", cmd_mm, log_ph_mm)

    st.info(
        "After alignment, load `*.sorted.bam` and `*.frequency.bed` into "
        "**IGV** or **UCSC Genome Browser**."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — HELP & TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════════════════════

with tab_help:
    st.markdown("""
## Tool environment modes

| Mode | When to use | Example |
|------|-------------|---------|
| **System PATH** | Tools are in the current shell's PATH | Default |
| **Conda env** | Tools are in a separate conda environment | `deepsignalpenv` |
| **Custom path** | Full path to the executable | `/opt/guppy/bin/guppy_basecaller` |

### Conda env mode — how it works

When *Conda env* is selected, every command is wrapped as:
```bash
conda run --no-capture-output -n <env_name> bash -c '<original command>'
```
This activates the environment before running the tool, without requiring you to manually `conda activate`.

### Custom path mode — how it works

The bare tool name in the command is replaced with the full path you provide:
```bash
# Original:  guppy_basecaller -i fast5s/ ...
# Wrapped:   /opt/ont/guppy/bin/guppy_basecaller -i fast5s/ ...
```

---

## Quick reference

| Step | Tool | Protocol |
|------|------|----------|
| Convert multi→single FAST5 | `multi_to_single_fast5` | R9 only |
| Basecalling | `guppy_basecaller` | R9 |
| Basecalling | `dorado` | R10 |
| Signal re-squiggling | `tombo resquiggle` | R9 only |
| Methylation calling | `deepsignal_plant call_mods` | Both |
| Frequency calculation | `deepsignal_plant call_freq` | Both |
| Context split | `split_freq_file_by_5mC_motif.py` | Both |

---

## Common issues

**`tombo resquiggle` fails for >20% of reads**
- Verify the reference genome matches the sequenced organism.
- Ensure `--basecall-group` matches the Guppy output group.

**`deepsignal_plant call_mods` runs on CPU only**
- Set `CUDA_VISIBLE_DEVICES=0` and ensure PyTorch CUDA is installed.
- Set `nproc_gpu = 0` for CPU-only mode.

**`--corrected_group` mismatch**
- Tombo uses hyphens (`--corrected-group`) while DeepSignal-plant uses
  underscores (`--corrected_group`). The *values* must be identical.

**NumPy version conflict**
- Do **not** install streamlit into the `deepsignalpenv` environment.
- Use the separate `deepsignal-gui` environment for the GUI and
  `deepsignalpenv` for the compute tools (see `environment_gui.yml`).
""")
