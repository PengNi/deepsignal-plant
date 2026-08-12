from torch.utils.data import Dataset
import os
import numpy as np

from .utils.process_utils import base2code_dna


def clear_linecache():
    pass


def parse_a_line2(line):
    words = line.strip().split("\t")

    sampleinfo = "\t".join(words[0:6])

    kmer = np.array([base2code_dna[x] for x in words[6]])
    base_means = np.array([float(x) for x in words[7].split(",")])
    base_stds = np.array([float(x) for x in words[8].split(",")])
    base_signal_lens = np.array([int(x) for x in words[9].split(",")])
    k_signals = np.array([[float(y) for y in x.split(",")] for x in words[10].split(";")])
    label = int(words[11])

    return sampleinfo, kmer, base_means, base_stds, base_signal_lens, k_signals, label


class SignalFeaData2(Dataset):
    def __init__(self, filename, transform=None):
        self._filename = os.path.abspath(filename)
        self._transform = transform
        self._file = None
        # Build a byte-offset index so __getitem__ can seek directly to any line
        # without loading the entire file into memory (linecache would cache ~10-20 GB
        # for a 20M-line file, causing OOM / SIGTERM on cluster nodes).
        offsets = []
        with open(self._filename, "rb") as f:
            pos = 0
            for line in f:
                offsets.append(pos)
                pos += len(line)
        self._offsets = np.array(offsets, dtype=np.int64)
        self._total_data = len(self._offsets)

    def __getitem__(self, idx):
        if self._file is None:
            self._file = open(self._filename, "r", encoding="utf-8")
        self._file.seek(int(self._offsets[idx]))
        line = self._file.readline()
        if line == "":
            return None
        output = parse_a_line2(line)
        if self._transform is not None:
            output = self._transform(output)
        return output

    def __len__(self):
        return self._total_data

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
