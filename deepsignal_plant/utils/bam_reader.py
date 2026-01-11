import pysam
from collections import defaultdict
from functools import cached_property
from dataclasses import dataclass

import numpy as np


def get_parent_id(bam_read):
    try:
        # if pi tag is present this is a child read
        return bam_read.get_tag("pi")
    except KeyError:
        # else this is the parent read so return query_name
        return bam_read.query_name


@dataclass
class ReadIndexedBam:
    bam_path: str

    @property
    def filename(self):
        """Alias to mimic AlignmentFile attribute"""
        return self.bam_path

    def __post_init__(self):
        self.num_reads = None
        self.bam_fh = None
        self._bam_idx = None
        self._iter = None
        self.compute_read_index()

    def has_index(self):
        """Alias to mimic AlignmentFile attribute"""
        if self.bam_fh is None:
            self.open()
        return self.bam_fh.has_index()

    def open(self):
        # hide warnings for no index when using unmapped or unsorted files
        self.pysam_save = pysam.set_verbosity(0)
        self.bam_fh = pysam.AlignmentFile(self.bam_path, mode="rb", check_sq=False)
        return self

    def close(self):
        self.bam_fh.close()
        self.bam_fh = None
        pysam.set_verbosity(self.pysam_save)

    def compute_read_index(self):
        bam_was_closed = self.bam_fh is None
        if bam_was_closed:
            self.open()
        self._bam_idx = defaultdict(list)
        self.num_records = 0
        while True:
            read_ptr = self.bam_fh.tell()
            try:
                read = next(self.bam_fh)
            except StopIteration:
                break
            index_read_id = get_parent_id(read)
            #index_read_id = read.query_name
            if read.is_supplementary or read.is_secondary:
                continue
            self.num_records += 1
            self._bam_idx[index_read_id].append(read_ptr)
        if bam_was_closed:
            self.close()
        self._bam_idx = dict(self._bam_idx)
        self.num_reads = len(self._bam_idx)

    def get_alignments(self, read_id):  # 多重序列比对，一条read可能map到多个位置
        if self._bam_idx is None:
            None
        if self.bam_fh is None:
            self.open()
        # try:
        read_ptrs = self._bam_idx[read_id]
        # except KeyError:
        #    None
        # throw keyerror
        for read_ptr in read_ptrs:
            self.bam_fh.seek(read_ptr)
            try:
                bam_read = next(self.bam_fh)
            except OSError as e:
                None
            yield bam_read

    def get_first_alignment(self, read_id):
        return next(self.get_alignments(read_id))

    def __contains__(self, read_id):
        return read_id in self._bam_idx

    def __getitem__(self, read_id):
        return self._bam_idx[read_id]

    def __del__(self):
        if self.bam_fh is not None:
            self.bam_fh.close()

    @cached_property
    def read_ids(self):
        return list(self._bam_idx.keys())

    def __iter__(self):
        self.bam_fh.reset()
        self._iter = iter(self.bam_fh)
        return self._iter

    def __next__(self):
        if self._iter is None:
            self._iter = iter(self.bam_fh)
        return next(self._iter)


def get_read_ids(bam_idx, pod5_dr, num_reads=None, return_num_bam_reads=False):
    """Get overlapping read ids from bam index and pod5 file

    Args:
        bam_idx (ReadIndexedBam): Read indexed BAM
        pod5_dr (pod5.DatasetReader): POD5 Dataset Reader
        num_reads (int): Maximum number of reads, or None for no max
        return_num_child_reads (bool): Return the number of bam records (child
            reads and multiple mappings) with a parent read ID. When set to
            False the number of parent read IDs is returned.
    """
    if isinstance(pod5_dr, str):
        both_read_ids = list(bam_idx.read_ids)
    else:
        pod5_read_ids = set(pod5_dr.read_ids)
        both_read_ids = list(pod5_read_ids.intersection(bam_idx.read_ids))
    num_both_read_ids = sum(
        len(bam_idx._bam_idx[parent_read_id]) for parent_read_id in both_read_ids
    )
    print(
        f"Found {bam_idx.num_records:,} valid BAM records. Found signal "
        f"in POD5 for {num_both_read_ids / bam_idx.num_records:.2%} of BAM "
        "records."
    )
    if not return_num_bam_reads:
        num_both_read_ids = len(both_read_ids)
    if num_reads is None:
        num_reads = num_both_read_ids
    else:
        num_reads = min(num_reads, num_both_read_ids)
    return both_read_ids, num_reads


class Read:
    def __init__(self, pod5_record, bam_record, read_id):  # pysam.AlignedSegment
        self._readid = read_id
        self._tag = dict(bam_record.tags)
        self._signal = np.asarray(pod5_record.signal)
        self._seq = bam_record.query_sequence
        self._num_trimmed = self._tag["ts"]
        self._norm_shift = self._tag["sm"]
        self._norm_scale = self._tag["sd"]

        self._ref_name = bam_record.reference_name
        self._strand = "-" if bam_record.is_reverse else "+"
        self._read_start = bam_record.query_alignment_start
        self._read_end = bam_record.query_alignment_end
        self._ref_start = bam_record.reference_start
        self._ref_end = bam_record.reference_end
        # self._ref_poses=bam_record.get_reference_positions()
        # self._read_poses=bam_record.positions

    def get_readid(self):
        return self._readid

    def get_raw_signal(self):
        return self._signal

    def get_seq(self):
        return self._seq.strip()

    def get_move(self):
        return np.asarray(self._tag["mv"][1:])

    def get_stride(self):
        return int(self._tag["mv"][0])

    def rescale_signals(self):
        num_trimmed = self._num_trimmed
        signal = self._signal
        if num_trimmed >= 0:
            self._signal = (signal[num_trimmed:] - self._norm_shift) / self._norm_scale
        else:
            self._signal = (signal[:num_trimmed] - self._norm_shift) / self._norm_scale
        return self._signal

    def check_signal(self):
        assert self._signal is not None

    def check_seq(self):
        assert self._seq is not None

    def check_map(self, bam_record):
        assert bam_record.is_unmapped is False

    def get_map_info(self, bam_record):
        cigars = bam_record.cigarstring
        chrom_strands = (self._ref_name, self._strand)
        frags = (
            self._read_start,
            self._read_end,
            self._ref_start,
            self._ref_end,
        )  # add tuple will occur error
        mapinfo = []
        mapinfo.append((cigars, chrom_strands, frags))
        return mapinfo
