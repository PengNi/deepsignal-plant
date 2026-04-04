"""the feature extraction module.
output format:
chrom, pos, alignstrand, pos_in_strand, readname, read_loc, k_mer, signal_means,
signal_stds, signal_lens, base_probs, raw_signals, methy_label
"""

# TODO: use pyguppyclient (https://github.com/nanoporetech/pyguppyclient) instead of --fast5-out

from __future__ import absolute_import

import sys
import os
import argparse
import time

# import h5py
import numpy as np
import multiprocessing as mp

# TODO: when using below import, will raise AttributeError: 'Queue' object has no attribute '_size'
# TODO: in call_mods module, didn't figure out why
# from .utils.process_utils import Queue
from multiprocessing import Queue

from .utils.process_utils import str2bool
from .utils.process_utils import display_args
from .utils.process_utils import get_files
from .utils.process_utils import get_refloc_of_methysite_in_motif
from .utils.process_utils import get_motif_seqs
from .utils.process_utils import fill_files_queue
from .utils.process_utils import read_position_file

from .utils import bam_reader

from collections import namedtuple
from .utils.process_utils import get_logger
# pod5 is lazily imported inside functions that use it to avoid loading
# its native library in every subprocess (resource exhaustion)
from .extract_features import _get_signals_rect
from .extract_features import _write_featurestr

# from .utils.process_utils import split_list
from .extract_features import _normalize_signals

from .extract_features import key_sep

from typing import Dict, Any, Tuple, Optional, Set
import itertools
from .utils.process_utils import complement_seq

LOGGER = get_logger(__name__)

time_wait = 0.1

MAP_RES = namedtuple(
    "MAP_RES",
    (
        "read_id",
        "q_seq",
        "ref_seq",
        "ctg",
        "strand",
        "r_st",
        "r_en",
        "q_st",
        "q_en",
        "cigar",
        "mapq",
    ),
)

def _group_signals_by_movetable_v2(trimed_signals, movetable, stride):
    """

    :param trimed_signals:
    :param movetable: numpy array
    :param stride:
    :return:
    """
    assert movetable[0] == 1
    # TODO: signal_duration not exactly equal to call_events * stride
    # TODO: maybe this move * stride way is not right!
    assert len(trimed_signals) >= len(movetable) * stride
    signal_group = []
    move_pos = np.append(np.argwhere(movetable == 1).flatten(), len(movetable))
    for move_idx in range(len(move_pos) - 1):
        start, end = move_pos[move_idx], move_pos[move_idx + 1]
        signal_group.append(trimed_signals[(start * stride) : (end * stride)].tolist())
    assert len(signal_group) == np.sum(movetable)
    return signal_group

def get_q2tloc_from_cigar(r_cigar_tuple, strand, seq_len):
    """
    insertion: -1, deletion: -2, mismatch: -3
    :param r_cigar_tuple: pysam.alignmentSegment.cigartuples
    :param strand: 1/-1 for fwd/rev
    :param seq_len: read alignment length
    :return: query pos to ref pos
    """
    fill_invalid = -2
    # get each base calls genomic position
    q_to_r_poss = np.full(seq_len + 1, fill_invalid, dtype=np.int32)
    # process cigar ops in read direction
    curr_r_pos, curr_q_pos = 0, 0
    cigar_ops = r_cigar_tuple if strand == 1 else r_cigar_tuple[::-1]
    for op, op_len in cigar_ops:
        if op == 1:
            # inserted bases into ref
            for q_pos in range(curr_q_pos, curr_q_pos + op_len):
                q_to_r_poss[q_pos] = -1
            curr_q_pos += op_len
        elif op in (2, 3):
            # deleted ref bases
            curr_r_pos += op_len
        elif op in (0, 7, 8):
            # aligned bases
            for op_offset in range(op_len):
                q_to_r_poss[curr_q_pos + op_offset] = curr_r_pos + op_offset
            curr_q_pos += op_len
            curr_r_pos += op_len
        elif op == 6:
            # padding (shouldn't happen in mappy)
            pass
    q_to_r_poss[curr_q_pos] = curr_r_pos
    if q_to_r_poss[-1] == fill_invalid:
        raise ValueError(
            (
                "Invalid cigar string encountered. Reference length: {}  Cigar "
                + "implied reference length: {}"
            ).format(seq_len, curr_r_pos)
        )
    return q_to_r_poss


################utils###################
# def _read_position_file(position_file):
#     postions = set()
#     with open(position_file, "r") as rf:
#         for line in rf:
#             words = line.strip().split("\t")
#             postions.add(key_sep.join(words[:3]))
#     return postions


def _features_to_str(features):
    """
    :param features: a tuple
    :return:
    """
    (
        chrom,
        pos,
        alignstrand,
        loc_in_ref,
        readname,
        read_loc,
        k_mer,
        signal_means,
        signal_stds,
        signal_lens,
        k_signals_rect,
        methy_label,
    ) = features
    means_text = ",".join([str(x) for x in np.around(signal_means, decimals=6)])
    stds_text = ",".join([str(x) for x in np.around(signal_stds, decimals=6)])
    signal_len_text = ",".join([str(x) for x in signal_lens])
    # base_probs_text = ",".join([str(x) for x in np.around(k_baseprobs, decimals=6)])
    k_signals_text = ";".join([",".join([str(y) for y in x]) for x in k_signals_rect])
    # freq_text = ';'.join([",".join([str(y) for y in x]) for x in freq])

    return "\t".join(
        [
            chrom,
            str(pos),
            alignstrand,
            str(loc_in_ref),
            readname,
            str(read_loc),
            k_mer,
            means_text,
            stds_text,
            signal_len_text,
            k_signals_text,
            str(methy_label),
        ]
    )


def process_data(
    data,
    motif_seqs,
    positions,
    kmer_len,
    signals_len,
    mapq,
    coverage_ratio,
    methyloc=0,
    methy_label=1,
    norm_method="mad",
):
    if kmer_len % 2 == 0:
        raise ValueError("kmer_len must be odd")
    num_bases = (kmer_len - 1) // 2
    features_list = []
    
    signal, seq_read = data
    if seq_read.mapping_quality < mapq:
        return features_list
    read_dict = dict(seq_read.tags)
    mv_table = np.asarray(read_dict["mv"][1:])
    stride = int(read_dict["mv"][0])
    num_trimmed = read_dict["ts"]
    if seq_read.has_tag('sp'):
        num_trimmed += seq_read.get_tag('sp')
    signal_trimmed = signal[num_trimmed:] if num_trimmed >= 0 else signal[:num_trimmed]
    norm_signals = _normalize_signals(signal_trimmed, norm_method)
    seq = seq_read.get_forward_sequence()
    signal_group = _group_signals_by_movetable_v2(norm_signals, mv_table, stride)
    tsite_locs = get_refloc_of_methysite_in_motif(seq, set(motif_seqs), methyloc)
    strand = "."
    q_to_r_poss = None
    if not seq_read.is_unmapped:
        strand = "-" if seq_read.is_reverse else "+"
        strand_code = -1 if seq_read.is_reverse else 1
        ref_start = seq_read.reference_start
        ref_end = seq_read.reference_end
        cigar_tuples = seq_read.cigartuples
        qalign_start = seq_read.query_alignment_start
        qalign_end = seq_read.query_alignment_end
        if (qalign_end - qalign_start) / seq_read.query_length < coverage_ratio:
            return features_list
        if seq_read.is_reverse:
            seq_start = len(seq) - qalign_end
            seq_end = len(seq) - qalign_start
        else:
            seq_start = qalign_start
            seq_end = qalign_end
        q_to_r_poss = get_q2tloc_from_cigar(
            cigar_tuples, strand_code, (seq_end - seq_start)
        )
    if seq_read.reference_name is None:
        ref_name = "."
    else:
        ref_name = seq_read.reference_name

    # 为 tag 计算准备 tsite_locs 的索引
    for loc_idx, loc_in_read in enumerate(tsite_locs):
        if num_bases <= loc_in_read < len(seq) - num_bases :
            ref_pos = -1
            if not seq_read.is_unmapped:
                if seq_start <= loc_in_read < seq_end:
                    offset_idx = loc_in_read - seq_start
                    if q_to_r_poss[offset_idx] != -1:
                        if strand == "-":
                            ref_pos = ref_end - 1 - q_to_r_poss[offset_idx]
                        else:
                            ref_pos = ref_start + q_to_r_poss[offset_idx]
                else:
                    continue

            if (positions is not None) and (
                key_sep.join([ref_name, str(ref_pos), strand]) not in positions
            ):
                continue

            # 计算 tag 特征
            k_mer = seq[(loc_in_read - num_bases) : (loc_in_read + num_bases + 1 )]
            k_signals = signal_group[
                (loc_in_read - num_bases) : (loc_in_read + num_bases + 1 )
            ]

            signal_lens = [len(x) for x in k_signals]
            signal_means = [np.mean(x) for x in k_signals]
            signal_stds = [np.std(x) for x in k_signals]

            features_list.append(
                _features_to_str(
                    (
                        ref_name,
                        str(ref_pos),
                        strand,
                        ".",  # loc_in_strand placeholder
                        seq_read.query_name,
                        ".",  # read_loc placeholder
                        k_mer,
                        signal_means,
                        signal_stds,
                        signal_lens,
                        _get_signals_rect(k_signals, signals_len),
                        methy_label,
                    )
                )
            )
    return features_list


def process_sig_seq(
    seq_index,
    pod5s_q,
    feature_Q,
    motif_seqs,
    positions,
    kmer_len,
    signals_len,
    mapq,
    coverage_ratio,
    methyloc=0,
    methyl_label=1,
    norm_method="mad",
    nproc_extract=1,
    process_chr=None,
):
    import pod5 as _pod5
    LOGGER.info("extract_features process-{} starts".format(os.getpid()))
    while True:
        pod5_file = pod5s_q.get()
        if pod5_file == "kill":
            pod5s_q.put("kill")
            break
        with _pod5.Reader(pod5_file[0]) as reader:
            for read_record in reader.reads():
                while (
                    feature_Q.qsize() > (nproc_extract if nproc_extract > 1 else 2) * 3
                ):
                    time.sleep(time_wait)
                read_name = str(read_record.read_id)
                signal = read_record.signal
                if signal is None:
                    continue
                try:
                    for seq_read in seq_index.get_alignments(read_name):
                        reference_name= seq_read.reference_name
                        if process_chr is not None:
                            if isinstance(process_chr, list):
                                if reference_name not in process_chr:
                                    continue
                            else:
                                if process_chr[:2] != 'no':
                                    if reference_name != process_chr:
                                        continue
                                elif process_chr[:2] == 'no':
                                    if reference_name == process_chr[2:]:
                                        continue
                        seq = seq_read.get_forward_sequence()
                        if seq is None:
                            continue
                        data = (signal, seq_read)
                        feature_lists = process_data(
                            data,
                            motif_seqs,
                            positions,
                            kmer_len,
                            signals_len,
                            mapq,
                            coverage_ratio,
                            methyloc,
                            methyl_label,
                            norm_method,
                        )
                        if len(feature_lists) != 0:
                            feature_Q.put(feature_lists)
                except KeyError:
                    LOGGER.warn("Read:%s not found in BAM file" % read_name)
                    continue
    LOGGER.info("extract_features process-{} finished".format(os.getpid()))


chh_thresh = {
    "CAA": 0.95,
    "CAC": 0.90, "CAT": 0.90,
    "CTA": 0.90, "CTT": 0.90, "CTC": 0.90,
    "CCA": 0.85, "CCT": 0.85, "CCC": 0.85,
    "DEFAULT": 0.90   # 其他未列出的 CHH 使用 0.90
}

def load_fasta(fasta_file: str) -> Dict[str, str]:
    """
    读取 FASTA 文件，返回 {chrom: sequence} 字典，序列全转为大写。

    Parameters
    ----------
    fasta_file : str
        参考基因组 FASTA 文件路径（支持 .fa, .fasta, .fa.gz）

    Returns
    -------
    ref_seq : dict
        {chromosome_name: "ATCG..."}
    """
    import gzip

    ref_seq = {}
    current_chrom = None
    seq_lines = []

    open_func = gzip.open if fasta_file.endswith('.gz') else open

    with open_func(fasta_file, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                # 保存上一个染色体
                if current_chrom is not None:
                    ref_seq[current_chrom] = ''.join(seq_lines).upper()
                # 开始新的染色体
                current_chrom = line[1:].split()[0]  # 取 > 后的第一个词
                seq_lines = []
            else:
                seq_lines.append(line)

    # 保存最后一个
    if current_chrom is not None:
        ref_seq[current_chrom] = ''.join(seq_lines).upper()

    return ref_seq

def read_bed(
    bisulfite_bed: str,
    label: int = 1,                     # 0=负样本, 1=正样本
    context: str = "CHH",             # "CHH" | "CHG" | "CpG"
    ref_seq_dict: Optional[Dict[str, str]] = None,
    depth_threshold: int = 5,
    # --- CHH ---
    chh_pos_thresh_dict: Optional[Dict[str, float]] = None,
    chh_default_pos_thresh: float = 0.90,
    # --- CHG ---
    chg_pos_thresh: float = 0.98,
    # --- CpG ---
    cpg_pos_thresh: float = 0.99,
    cpg_neg_thresh: float = 0.02,
    # --- strict (CpG only) ---
    strict: bool = False,
    strict_pos_thresh: float = 95.0,
    strict_neg_thresh: float = 5.0
) ->Set[str]:
    """
        读取 BS-seq BED，返回高置信正/负样本位点集合（chrom||pos）。
        使用 rev_seq_dict（整条染色体已反向互补）来提取负链 motif：
        - 正链:  ref_seq[pos : pos+3]
        - 负链:  rev_seq[ len - 1 - pos : len - 1 - pos + 3 ]
    """
    if label not in (0, 1):
        raise ValueError("label must be 0 or 1")
    if context in ("CHH", "CHG") and ref_seq_dict is None:
        raise ValueError(f"ref_seq_dict required for {context}")

    # ---------- 1. 预构建反向互补参考 ----------
    rev_seq_dict: Dict[str, str] = {}
    if ref_seq_dict is not None:
        for chrom, seq in ref_seq_dict.items():
            rev_seq_dict[chrom] = complement_seq(seq)
    chrom_lengths = {chrom: len(seq) for chrom, seq in ref_seq_dict.items()}

    # ---------- 2. 读取 BED ----------
    key_sep = "||"
    value_info: Dict[str, float] = {}
    with open(bisulfite_bed, "r") as rf:
        for line in rf:
            words = line.strip().split()
            if len(words) < 6:
                continue
            chrom, pos_str, strand = words[0], words[1], words[5]
            pos = int(pos_str)

            if len(words) == 6:
                freq = 100.0 if label == 1 else 0.0
            else:
                depth = int(words[9])
                if depth < depth_threshold:
                    continue
                freq = float(words[10])

            m_key = key_sep.join([chrom, pos_str, strand])
            value_info[m_key] = freq

    positions: Set[str] = set()

    # ---------- 3. 按 context 处理 ----------
    if context == "CHH":
        pos_thresh = chh_pos_thresh_dict or {}
        # convert default to percentage scale (一致与 freq 单位 0~100)
        default_thresh = chh_default_pos_thresh 

        for m_key, freq in value_info.items():
            chrom, pos_str, strand = m_key.split(key_sep)
            pos = int(pos_str)
            seq_len = chrom_lengths[chrom]

            # 选择参考序列（注意：rev_seq_dict 已经是反向互补序列）
            ref_seq = rev_seq_dict[chrom] if strand == '-' else ref_seq_dict[chrom]

            # 提取 motif（根据你验证过的坐标方式）
            if strand == '+':
                if pos < 0 or pos + 3 > seq_len:
                    continue
                motif = ref_seq[pos:pos + 3].upper()
            else:
                rev_start = seq_len - 1 - pos
                if rev_start < 0 or rev_start + 3 > seq_len:
                    continue
                motif = ref_seq[rev_start:rev_start + 3].upper()

            # CHH 检查
            if not (motif[0] == 'C' and motif[1] in 'ATC' and motif[2] in 'ATC'):
                continue

            base_key = key_sep.join([chrom, pos_str, strand])
            if label == 1:
                thresh = pos_thresh.get(motif, default_thresh)* 100.0
                if freq >= thresh:
                    positions.add(base_key)
            else:
                if freq <= 0.0:
                    positions.add(base_key)

    elif context == "CHG":
        # CHG 使用单一阈值 chg_pos_thresh（比例），按 freq 的单位 0~100 比较
        chg_thresh_pct = chg_pos_thresh * 100.0

        for m_key, freq in value_info.items():
            chrom, pos_str, strand = m_key.split(key_sep)
            pos = int(pos_str)
            seq_len = chrom_lengths[chrom]

            ref_seq = rev_seq_dict[chrom] if strand == '-' else ref_seq_dict[chrom]

            if strand == '+':
                if pos < 0 or pos + 3 > seq_len:
                    continue
                motif = ref_seq[pos:pos + 3].upper()
            else:
                rev_start = seq_len - 1 - pos
                if rev_start < 0 or rev_start + 3 > seq_len:
                    continue
                motif = ref_seq[rev_start:rev_start + 3].upper()

            # CHG 检查
            if not (motif[0] == 'C' and motif[1] in 'ATC' and motif[2] == 'G'):
                continue

            base_key = key_sep.join([chrom, pos_str, strand])
            if label == 1:
                if freq >= chg_thresh_pct:
                    positions.add(base_key)
            else:
                if freq <= 0.0:
                    positions.add(base_key)

    elif context == "CG":
        if strict:
            for m_key, freq in value_info.items():
                chrom, pos_str, strand = m_key.split(key_sep)
                pos = int(pos_str)
                pair_pos = pos + 1 if strand == '+' else pos - 1
                pair_strand = '-' if strand == '+' else '+'
                pair_key = key_sep.join([chrom, str(pair_pos), pair_strand])
                if pair_key not in value_info:
                    continue
                pair_freq = value_info[pair_key]
                base_key = key_sep.join([chrom, pos_str, strand])
                if label == 1:
                    if freq >= strict_pos_thresh and pair_freq >= strict_pos_thresh:
                        positions.add(base_key)
                else:
                    if freq <= strict_neg_thresh and pair_freq <= strict_neg_thresh:
                        positions.add(base_key)
        else:
            # 非 strict 情况：只按单个位点 freq 与 CpG 阈值过滤（假定 pos 对应 C）
            cpg_thresh_pct = cpg_pos_thresh * 100.0
            cpg_neg_pct = cpg_neg_thresh * 100.0
            for m_key, freq in value_info.items():
                chrom, pos_str, strand = m_key.split(key_sep)
                pos = int(pos_str)
                seq_len = chrom_lengths[chrom]
                ref_seq = rev_seq_dict[chrom] if strand == '-' else ref_seq_dict[chrom]

                if strand == '+':
                    if pos < 0 or pos + 3 > seq_len:
                        continue
                    motif = ref_seq[pos:pos + 3].upper()
                else:
                    rev_start = seq_len - 1 - pos
                    if rev_start < 0 or rev_start + 3 > seq_len:
                        continue
                    motif = ref_seq[rev_start:rev_start + 3].upper()

                # 确保是 CpG（motif 前两位为 CG）
                if not motif.startswith("CG"):
                    continue

                base_key = key_sep.join([chrom, pos_str, strand])
                if label == 1:
                    if freq >= cpg_thresh_pct:
                        positions.add(base_key)
                else:
                    if freq <= cpg_neg_pct:
                        positions.add(base_key)
    else:
        raise ValueError("context must be 'CHH', 'CHG', or 'CpG'")

    LOGGER.info("extracted %d positions", len(positions))
    return positions

def extract_features(args):
    start = time.time()
    if args.reference_path is not None:
        ref_path = os.path.abspath(args.reference_path)
        if not os.path.exists(ref_path):
            raise ValueError("--reference_path not set right!")
    else:
        ref_path = None

    LOGGER.info("[extract] starts")

    input_dir = os.path.abspath(args.input_path)
    if not os.path.exists(input_dir):
        raise ValueError("--input_dir not set right!")
    if not os.path.isdir(input_dir):
        raise NotADirectoryError("--input_dir not a directory")
    LOGGER.info("read position file if it is not None")
    positions = None
    if args.positions is not None:
        positions = read_position_file(args.positions)
    elif args.bed is not None:
        if args.reference_path is None:
            ref=None
        else:
            ref=load_fasta(args.reference_path)
        positions = read_bed(args.bed,args.methy_label,args.motifs,ref,args.depth,chh_thresh)
    is_recursive = str2bool(args.recursively)
    is_dna = False if args.rna else True
    motif_seqs = get_motif_seqs(args.motifs, is_dna)
    bam_index = bam_reader.ReadIndexedBam(args.bam)
    pod5_dr = get_files(input_dir, is_recursive, ".pod5")

    pod5s_q = Queue()
    fill_files_queue(pod5s_q, pod5_dr)
    pod5s_q.put("kill")
    features_batch_q = Queue()
    # error_q = Queue()
    p_rfs = []

    nproc = args.nproc - 1
    for proc_idx in range(nproc):
        p_rf = mp.Process(
            target=process_sig_seq,
            args=(
                bam_index,
                pod5s_q,
                features_batch_q,
                motif_seqs,
                positions,
                args.seq_len,
                args.signal_len,
                args.mapq,
                args.coverage_ratio,
                args.mod_loc,
                args.methy_label,
                args.normalize_method,
                nproc,
                args.chr,
            ),
            name="extracter_{:03d}".format(proc_idx),
        )
        p_rf.daemon = True
        p_rf.start()
        p_rfs.append(p_rf)

    p_w = mp.Process(
        target=_write_featurestr,
        args=(
            args.write_path,
            features_batch_q,
            args.w_batch_num,
            str2bool(args.w_is_dir),
        ),
        name="writer",
    )
    p_w.daemon = True
    p_w.start()
    # finish processes
    for pb in p_rfs:
        pb.join()
    features_batch_q.put("kill")

    p_w.join()

    LOGGER.info("[extract] finished, cost {:.1f}s".format(time.time() - start))


def main():
    extraction_parser = argparse.ArgumentParser(
        "extract features from pod5 for "
        "training or testing."
        "\nIt is suggested that running this module 1 flowcell a time, "
        "or a group of flowcells a time, "
        "if the whole data is extremely large."
    )

    ep_input = extraction_parser.add_argument_group("INPUT")
    ep_input.add_argument(
        "--input_path",
        "-i",
        action="store",
        type=str,
        required=True,
        help="the directory of pod5 files",
    )
    ep_input.add_argument("--bam", type=str, help="the bam filepath")
    ep_input.add_argument(
        "--recursively",
        "-r",
        action="store",
        type=str,
        required=False,
        default="yes",
        help="is to find pod5 files from input_dir recursively. "
        "default true, t, yes, 1",
    )
    ep_input.add_argument(
        "--reference_path",
        action="store",
        type=str,
        required=False,
        default=None,
        help="the reference file to be used, usually is a .fa file",
    )
    ep_input.add_argument(
        "--bed",
        action="store",
        type=str,
        required=False,
        default=None,
        help="the wgbs file to be used, usually is a .bed file",
    )
    ep_input.add_argument(
        "--rna",
        action="store_true",
        default=False,
        required=False,
        help="the fast5/pod5 files are from RNA samples. if is rna, the signals are reversed. "
        "NOTE: Currently no use, waiting for further extentsion",
    )
    ep_input.add_argument("--bam", type=str, help="the bam filepath")
    ep_extraction = extraction_parser.add_argument_group("EXTRACTION")
    ep_extraction.add_argument(
        "--normalize_method",
        action="store",
        type=str,
        choices=["mad", "zscore"],
        default="mad",
        required=False,
        help="the way for normalizing signals in read level. "
        "mad or zscore, default mad",
    )
    ep_extraction.add_argument(
        "--methy_label",
        action="store",
        type=int,
        choices=[1, 0],
        required=False,
        default=1,
        help="the label of the interested modified bases, this is for training."
        " 0 or 1, default 1",
    )
    ep_extraction.add_argument(
        "--seq_len",
        action="store",
        type=int,
        required=False,
        default=21,
        help="len of kmer. default 21",
    )
    ep_extraction.add_argument(
        "--signal_len",
        action="store",
        type=int,
        required=False,
        default=15,
        help="the number of signals of one base to be used in deepsignal, default 15",
    )
    ep_extraction.add_argument("--chr", type=str, required=False, help='only extract chr')
    ep_extraction.add_argument(
        "--motifs",
        action="store",
        type=str,
        required=False,
        default="CG",
        help="motif seq to be extracted, default: CG. "
        "can be multi motifs splited by comma "
        "(no space allowed in the input str), "
        "or use IUPAC alphabet, "
        "the mod_loc of all motifs must be "
        "the same",
    )
    ep_extraction.add_argument(
        "--mod_loc",
        action="store",
        type=int,
        required=False,
        default=0,
        help="0-based location of the targeted base in the motif, default 0",
    )
    # ep_extraction.add_argument("--region", action="store", type=str,
    #                            required=False, default=None,
    #                            help="region of interest, e.g.: chr1:0-10000, default None, "
    #                                 "for the whole region")
    ep_extraction.add_argument(
        "--positions",
        action="store",
        type=str,
        required=False,
        default=None,
        help="file with a list of positions interested (must be formatted as tab-separated file"
        " with chromosome, position (in fwd strand), and strand. motifs/mod_loc are still "
        "need to be set. --positions is used to narrow down the range of the trageted "
        "motif locs. default None",
    )
    ep_extraction.add_argument(
        "--pad_only_r",
        action="store_true",
        default=False,
        help="pad zeros to only the right of signals array of one base, "
        "when the number of signals is less than --signal_len. "
        "default False (pad in two sides).",
    )
    ep_extraction.add_argument(
        "--depth",
        action="store",
        type=int,
        default=5,
        required=False,
        help="depth of kmer, default 5",
    )
    ep_mapping = extraction_parser.add_argument_group("MAPPING")
    ep_mapping.add_argument(
        "--mapping",
        action="store_true",
        default=False,
        required=False,
        help="use MAPPING to get alignment, default false",
    )
    ep_mapping.add_argument(
        "--mapq",
        type=int,
        default=1,
        required=False,
        help="MAPping Quality cutoff for selecting alignment items, default 1",
    )
    ep_mapping.add_argument(
        "--identity",
        type=float,
        default=0.0,
        required=False,
        help="identity cutoff for selecting alignment items, default 0.0",
    )
    ep_mapping.add_argument(
        "--coverage_ratio",
        type=float,
        default=0.50,
        required=False,
        help="percent of coverage, read alignment len against read len, default 0.50",
    )
    ep_output = extraction_parser.add_argument_group("OUTPUT")
    ep_output.add_argument(
        "--write_path",
        "-o",
        action="store",
        type=str,
        required=True,
        help="file path to save the features",
    )
    ep_output.add_argument(
        "--w_is_dir",
        action="store",
        type=str,
        required=False,
        default="no",
        help="if using a dir to save features into multiple files",
    )
    ep_output.add_argument(
        "--w_batch_num",
        action="store",
        type=int,
        required=False,
        default=200,
        help="features batch num to save in a single writed file when --is_dir is true",
    )
    extraction_parser.add_argument(
        "--nproc",
        "-p",
        action="store",
        type=int,
        default=10,
        required=False,
        help="number of processes to be used, default 10",
    )

    extraction_args = extraction_parser.parse_args()
    display_args(extraction_args)
    extract_features(extraction_args)


if __name__ == "__main__":
    sys.exit(main())
