"""
for parseing line-based files
"""

key_sep = "||"


class ModRecord:
    def __init__(self, fields):
        self._chromosome = fields[0]
        self._pos = int(fields[1])
        self._site_key = key_sep.join([self._chromosome, str(self._pos)])

        self._strand = fields[2]
        self._pos_in_strand = int(fields[3])
        self._readname = fields[4]
        self._read_strand = fields[5]
        self._prob_0 = float(fields[6])
        self._prob_1 = float(fields[7])
        self._called_label = int(fields[8])
        self._kmer = fields[9]

    def is_record_callable(self, prob_threshold):
        if abs(self._prob_0 - self._prob_1) < prob_threshold:
            return False
        return True


def split_key(key):
    words = key.split(key_sep)
    return words[0], int(words[1])


class SiteStats:
    def __init__(self, strand, pos_in_strand, kmer):

        self._strand = strand
        self._pos_in_strand = pos_in_strand
        self._kmer = kmer

        self._prob_0 = 0.0
        self._prob_1 = 0.0
        self._met = 0
        self._unmet = 0
        self._coverage = 0
        # self._rmet = -1.0
