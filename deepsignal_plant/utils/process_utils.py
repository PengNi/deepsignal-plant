from __future__ import absolute_import
import fnmatch
import os
import random

import multiprocessing as mp
# import torch.multiprocessing as mp

import multiprocessing.queues
# import torch.multiprocessing.queue

import numpy as np
import gc
import math

basepairs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
             'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y',
             'Y': 'R', 'B': 'V', 'V': 'B', 'D': 'H', 'H': "D",
             'Z': 'Z'}
basepairs_rna = {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A', 'N': 'N',
                 'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y',
                 'Y': 'R', 'B': 'V', 'V': 'B', 'D': 'H', 'H': "D",
                 'Z': 'Z'}

base2code_dna = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4,
                 'W': 5, 'S': 6, 'M': 7, 'K': 8, 'R': 9,
                 'Y': 10, 'B': 11, 'V': 12, 'D': 13, 'H': 14,
                 'Z': 15}
code2base_dna = dict((v, k) for k, v in base2code_dna.items())
base2code_rna = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'N': 4,
                 'W': 5, 'S': 6, 'M': 7, 'K': 8, 'R': 9,
                 'Y': 10, 'B': 11, 'V': 12, 'D': 13, 'H': 14,
                 'Z': 15}
code2base_rna = dict((v, k) for k, v in base2code_rna.items())

iupac_alphabets = {'A': ['A'], 'T': ['T'], 'C': ['C'], 'G': ['G'],
                   'R': ['A', 'G'], 'M': ['A', 'C'], 'S': ['C', 'G'],
                   'Y': ['C', 'T'], 'K': ['G', 'T'], 'W': ['A', 'T'],
                   'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
                   'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'],
                   'N': ['A', 'C', 'G', 'T']}
iupac_alphabets_rna = {'A': ['A'], 'C': ['C'], 'G': ['G'], 'U': ['U'],
                       'R': ['A', 'G'], 'M': ['A', 'C'], 'S': ['C', 'G'],
                       'Y': ['C', 'U'], 'K': ['G', 'U'], 'W': ['A', 'U'],
                       'B': ['C', 'G', 'U'], 'D': ['A', 'G', 'U'],
                       'H': ['A', 'C', 'U'], 'V': ['A', 'C', 'G'],
                       'N': ['A', 'C', 'G', 'U']}

# max_queue_size = 2000

nproc_to_call_mods_in_cpu_mode = 2


def str2bool(v):
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")


def _alphabet(letter, dbasepairs):
    if letter in dbasepairs.keys():
        return dbasepairs[letter]
    return 'N'


def complement_seq(base_seq, seq_type="DNA"):
    rbase_seq = base_seq[::-1]
    comseq = ''
    try:
        if seq_type == "DNA":
            comseq = ''.join([_alphabet(x, basepairs) for x in rbase_seq])
        elif seq_type == "RNA":
            comseq = ''.join([_alphabet(x, basepairs_rna) for x in rbase_seq])
        else:
            raise ValueError("the seq_type must be DNA or RNA")
    except Exception:
        print('something wrong in the dna/rna sequence.')
    return comseq


# def get_refloc_of_methysite_in_motif(seqstr, motif='CG', methyloc_in_motif=0):
#     """
#
#     :param seqstr:
#     :param motif:
#     :param methyloc_in_motif: 0-based
#     :return:
#     """
#     strlen = len(seqstr)
#     motiflen = len(motif)
#     sites = []
#     for i in range(0, strlen - motiflen + 1):
#         if seqstr[i:i + motiflen] == motif:
#             sites.append(i+methyloc_in_motif)
#     return sites


def get_refloc_of_methysite_in_motif(seqstr, motifset, methyloc_in_motif=0):
    """

    :param seqstr:
    :param motifset:
    :param methyloc_in_motif: 0-based
    :return:
    """
    motifset = set(motifset)
    strlen = len(seqstr)
    motiflen = len(list(motifset)[0])
    sites = []
    for i in range(0, strlen - motiflen + 1):
        if seqstr[i:i + motiflen] in motifset:
            sites.append(i+methyloc_in_motif)
    return sites


def _convert_motif_seq(ori_seq, is_dna=True):
    outbases = []
    for bbase in ori_seq:
        if is_dna:
            outbases.append(iupac_alphabets[bbase])
        else:
            outbases.append(iupac_alphabets_rna[bbase])

    def recursive_permute(bases_list):
        if len(bases_list) == 1:
            return bases_list[0]
        elif len(bases_list) == 2:
            pseqs = []
            for fbase in bases_list[0]:
                for sbase in bases_list[1]:
                    pseqs.append(fbase + sbase)
            return pseqs
        else:
            pseqs = recursive_permute(bases_list[1:])
            pseq_list = [bases_list[0], pseqs]
            return recursive_permute(pseq_list)
    return recursive_permute(outbases)


def get_motif_seqs(motifs, is_dna=True):
    ori_motif_seqs = motifs.strip().split(',')

    motif_seqs = []
    for ori_motif in ori_motif_seqs:
        motif_seqs += _convert_motif_seq(ori_motif.strip().upper(), is_dna)
    return motif_seqs


def get_fast5s(fast5_dir, is_recursive=True):
    fast5_dir = os.path.abspath(fast5_dir)
    fast5s = []
    if is_recursive:
        for root, dirnames, filenames in os.walk(fast5_dir):
            for filename in fnmatch.filter(filenames, '*.fast5'):
                fast5_path = os.path.join(root, filename)
                fast5s.append(fast5_path)
    else:
        for fast5_name in os.listdir(fast5_dir):
            if fast5_name.endswith('.fast5'):
                fast5_path = '/'.join([fast5_dir, fast5_name])
                fast5s.append(fast5_path)
    return fast5s


def parse_region_str(regionstr):
    """
    :param regionstr: chrom:start-end or chrom:start or chrom.
     0-based, half-open region: [start, end)
    :return: chrom, start, end
    """
    try:
        if regionstr is None:
            return None, None, None
        elif ":" in regionstr:
            regioncse = regionstr.strip().split(":")
            assert len(regioncse) == 2
            chrom, se = regioncse[0], regioncse[1]
            if "-" in se:
                sne = se.split("-")
                assert len(sne) == 2
                s, e = int(sne[0]), int(sne[1])
                return chrom, s, e
            else:
                return chrom, int(se), None
        else:
            return regionstr.strip(), None, None
    except Exception:
        raise ValueError("--region not set right!")


# https://thispointer.com/python-three-ways-to-check-if-a-file-is-empty/
def is_file_empty(file_name):
    """ Check if file is empty by confirming if its size is 0 bytes"""
    # Check if file exist and it is empty
    return os.path.isfile(file_name) and os.path.getsize(file_name) == 0


# functions for combining files and random sampling lines of txt files ================
def count_line_num(sl_filepath, fheader=False, printlog=True):
    count = 0
    with open(sl_filepath, 'r') as rf:
        if fheader:
            next(rf)
        for _ in rf:
            count += 1
    if printlog:
        print('done count the lines of file {}'.format(sl_filepath))
    return count


def random_select_file_rows(ori_file, w_file, w_other_file=None, maxrownum=100000000, header=False):
    """

    :param ori_file:
    :param w_file:
    :param w_other_file:
    :param maxrownum:
    :param header:
    :return:
    """
    # whole_rows = open(ori_file).readlines()
    # nrows = len(whole_rows) - 1

    nrows = 0
    with open(ori_file) as rf:
        for _ in rf:
            nrows += 1
    if header:
        nrows -= 1
    print('thera are {} lines (rm header if a header exists) in the file {}'.format(nrows, ori_file))

    actual_nline = maxrownum
    if nrows <= actual_nline:
        actual_nline = nrows
        print('gonna return all lines in ori_file {}'.format(ori_file))

    random_lines = random.sample(range(1, nrows+1), actual_nline)
    random_lines = [0] + sorted(random_lines)
    random_lines[-1] = nrows

    wf = open(w_file, 'w')
    if w_other_file is not None:
        wlf = open(w_other_file, 'w')
    with open(ori_file) as rf:
        if header:
            lineheader = next(rf)
            wf.write(lineheader)
            if w_other_file is not None:
                wlf.write(lineheader)
        for i in range(1, len(random_lines)):
            chosen_line = ''
            for j in range(0, random_lines[i]-random_lines[i-1] - 1):
                other_line = next(rf)
                if w_other_file is not None:
                    wlf.write(other_line)
            chosen_line = next(rf)
            wf.write(chosen_line)
    wf.close()
    if w_other_file is not None:
        wlf.close()
    print('random_select_file_rows finished..')


def random_select_file_rows_s(ori_file, w_file, w_other_file, maxrownum=100000000, header=False):
    """
    split line indexs to two arrays randomly, write the two group of lines into two files,
     and return the arrays
    :param ori_file:
    :param w_file:
    :param w_other_file:
    :param maxrownum:
    :param header:
    :return:
    """
    # whole_rows = open(ori_file).readlines()
    # nrows = len(whole_rows) - 1

    nrows = 0
    with open(ori_file) as rf:
        for _ in rf:
            nrows += 1
    if header:
        nrows -= 1
    print('thera are {} lines (rm header if a header exists) in the file {}'.format(nrows, ori_file))

    actual_nline = maxrownum
    if nrows <= actual_nline:
        actual_nline = nrows
        print('gonna return all lines in ori_file {}'.format(ori_file))

    random_lines = random.sample(range(1, nrows+1), actual_nline)
    random_lines = [0] + sorted(random_lines)
    random_lines[-1] = nrows

    wf = open(w_file, 'w')
    wlf = open(w_other_file, 'w')
    lidxs1, lidxs2 = [], []
    lidx_cnt = 0
    with open(ori_file) as rf:
        if header:
            lineheader = next(rf)
            wf.write(lineheader)
            wlf.write(lineheader)
        for i in range(1, len(random_lines)):
            chosen_line = ''
            for j in range(0, random_lines[i]-random_lines[i-1] - 1):
                wlf.write(next(rf))
                lidxs2.append(lidx_cnt)
                lidx_cnt += 1
            chosen_line = next(rf)
            wf.write(chosen_line)
            lidxs1.append(lidx_cnt)
            lidx_cnt += 1
    wf.close()
    wlf.close()
    print('random_select_file_rows_s finished, file1: {}, file2: {}..'.format(len(lidxs1),
                                                                              len(lidxs2)))
    return lidxs1, lidxs2


def read_one_shuffle_info(filepath, shuffle_lines_num, total_lines_num, checked_lines_num, isheader):
    with open(filepath, 'r') as rf:
        if isheader:
            next(rf)
        count = 0
        while count < checked_lines_num:
            next(rf)
            count += 1

        count = 0
        lines_info = []
        lines_num = min(shuffle_lines_num, (total_lines_num - checked_lines_num))
        for line in rf:
            if count < lines_num:
                lines_info.append(line.strip())
                count += 1
            else:
                break
        print('done reading file {}'.format(filepath))
        return lines_info


def shuffle_samples(samples_info):
    mark = list(range(len(samples_info)))
    np.random.shuffle(mark)
    shuffled_samples = []
    for i in mark:
        shuffled_samples.append(samples_info[i])
    return shuffled_samples


def write_to_one_file_append(features_info, wfilepath):
    with open(wfilepath, 'a') as wf:
        for i in range(0, len(features_info)):
            wf.write(features_info[i] + '\n')
    print('done writing features info to {}'.format(wfilepath))


def concat_two_files(file1, file2, concated_fp, shuffle_lines_num=2000000,
                     lines_num=1000000000000, isheader=False):
    open(concated_fp, 'w').close()

    if isheader:
        rf1 = open(file1, 'r')
        wf = open(concated_fp, 'a')
        wf.write(next(rf1))
        wf.close()
        rf1.close()

    f1line_count = count_line_num(file1, isheader)
    f2line_count = count_line_num(file2, False)

    line_ratio = float(f2line_count) / f1line_count
    shuffle_lines_num2 = round(line_ratio * shuffle_lines_num) + 1

    checked_lines_num1, checked_lines_num2 = 0, 0
    while checked_lines_num1 < lines_num or checked_lines_num2 < lines_num:
        file1_info = read_one_shuffle_info(file1, shuffle_lines_num, lines_num, checked_lines_num1, isheader)
        checked_lines_num1 += len(file1_info)
        file2_info = read_one_shuffle_info(file2, shuffle_lines_num2, lines_num, checked_lines_num2, False)
        checked_lines_num2 += len(file2_info)
        if len(file1_info) == 0 and len(file2_info) == 0:
            break
        samples_info = shuffle_samples(file1_info + file2_info)
        write_to_one_file_append(samples_info, concated_fp)

        del file1_info
        del file2_info
        del samples_info
        gc.collect()
    print('done concating files to: {}'.format(concated_fp))
# =====================================================================================


def display_args(args):
    arg_vars = vars(args)
    print("# ===============================================")
    print("## parameters: ")
    for arg_key in arg_vars.keys():
        if arg_key != 'func':
            print("{}:\n\t{}".format(arg_key, arg_vars[arg_key]))
    print("# ===============================================")


# for balancing kmer distri in training samples ===
def _count_kmers_of_feafile(feafile):
    kmer_count = {}
    kmers = set()
    with open(feafile, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            kmer = words[6]
            if kmer not in kmers:
                kmers.add(kmer)
                kmer_count[kmer] = 0
            kmer_count[kmer] += 1
    return kmer_count


# for balancing kmer distri in training samples ===
def _get_kmer2ratio_n_totalline(kmer_count):
    total_cnt = sum(list(kmer_count.values()))
    kmer_ratios = dict()
    for kmer in kmer_count.keys():
        kmer_ratios[kmer] = float(kmer_count[kmer])/total_cnt
    return kmer_ratios, total_cnt


# for balancing kmer distri in training samples ===
def _get_kmer2lines(feafile):
    kmer2lines = {}
    kmers = set()
    with open(feafile, "r") as rf:
        lcnt = 0
        for line in rf:
            words = line.strip().split("\t")
            kmer = words[6]
            if kmer not in kmers:
                kmers.add(kmer)
                kmer2lines[kmer] = []
            kmer2lines[kmer].append(lcnt)
            lcnt += 1
    return kmer2lines


# for balancing kmer distri in training samples ===
def _rand_select_by_kmer_ratio(kmer2lines, kmer2ratios, totalline):
    inter_kmers = set(kmer2lines.keys()).intersection(set(kmer2ratios.keys()))
    line_kmer_diff = set(kmer2lines.keys()).difference(set(kmer2ratios.keys()))
    ratio_kmer_diff = set(kmer2ratios.keys()).difference(set(kmer2lines.keys()))
    print("comm kmers: {}, line_kmers_diff: {}, ratio_kmers_diff: {}".format(len(inter_kmers),
                                                                             len(line_kmer_diff),
                                                                             len(ratio_kmer_diff)))
    selected_lines = []
    unselected_lines = []
    unratioed_kmers = line_kmer_diff
    cnts = 0
    for kmer in inter_kmers:
        linenum = int(math.ceil(totalline * kmer2ratios[kmer]))
        lines = kmer2lines[kmer]
        if len(lines) <= linenum:
            selected_lines += lines
            cnts += (linenum - len(lines))
        else:
            seledtmp = random.sample(lines, linenum)
            selected_lines += seledtmp
            unselected_lines += list(set(lines).difference(seledtmp))
    print("for {} common kmers, fill {} samples, "
          "{} samples that can't be filled".format(len(inter_kmers),
                                                   len(selected_lines),
                                                   cnts))
    print("for {} ratio_diff kmers, "
          "{} samples that cant't be filled".format(len(ratio_kmer_diff),
                                                    sum([round(totalline * kmer2ratios[kmer])
                                                         for kmer in ratio_kmer_diff])))
    unfilled_cnt = totalline - len(selected_lines)
    print("totalline: {}, need to fill: {}".format(totalline, unfilled_cnt))
    if unfilled_cnt > 0 and len(unratioed_kmers) > 0:
        minlinenum = int(math.ceil(float(unfilled_cnt)/len(unratioed_kmers)))
        cnts = 0
        for kmer in unratioed_kmers:
            lines = kmer2lines[kmer]
            if len(lines) <= minlinenum:
                selected_lines += lines
                cnts += len(lines)
            else:
                seledtmp = random.sample(lines, minlinenum)
                selected_lines += seledtmp
                cnts += minlinenum
                unselected_lines += list(set(lines).difference(seledtmp))
        print("extract {} samples from {} line_diff kmers".format(cnts, len(unratioed_kmers)))
    unfilled_cnt = totalline - len(selected_lines)
    if unfilled_cnt > 0:
        print("totalline: {}, still need to fill: {}".format(totalline, unfilled_cnt))
        random.shuffle(unselected_lines)
        triplefill_cnt = unfilled_cnt
        if len(unselected_lines) <= unfilled_cnt:
            selected_lines += unselected_lines
            triplefill_cnt = len(unselected_lines)
        else:
            selected_lines += unselected_lines[:unfilled_cnt]
        print("extract {} samples from {} samples not used above".format(triplefill_cnt, len(unselected_lines)))
    selected_lines = sorted(selected_lines)
    selected_lines = [-1] + selected_lines
    return selected_lines


# for balancing kmer distri in training samples ===
def _write_randsel_lines(feafile, wfile, seled_lines):
    wf = open(wfile, 'w')
    with open(feafile) as rf:
        for i in range(1, len(seled_lines)):
            chosen_line = ''
            for j in range(0, seled_lines[i] - seled_lines[i - 1]):
                # print(j)
                chosen_line = next(rf)
            wf.write(chosen_line)
    wf.close()
    print('_write_randsel_lines finished..')


# balance kmer distri in neg_training file as pos_training file
def select_negsamples_asposkmer(pos_file, totalneg_file, seled_neg_file):
    kmer_count = _count_kmers_of_feafile(pos_file)
    kmer2ratio, totalline = _get_kmer2ratio_n_totalline(kmer_count)

    print("{} kmers from kmer2ratio file: {}".format(len(kmer2ratio), pos_file))
    kmer2lines = _get_kmer2lines(totalneg_file)
    sel_lines = _rand_select_by_kmer_ratio(kmer2lines, kmer2ratio, totalline)
    _write_randsel_lines(totalneg_file, seled_neg_file, sel_lines)


# get model type main params
def get_model_type_str(model_type, is_base, is_signallen):
    if model_type != "signal_bilstm":
        basestr = "with_base" if is_base else "no_base"
        slenstr = "with_slen" if is_signallen else "no_slen"
        return "_".join([model_type, basestr, slenstr])
    else:
        return "_".join([model_type])


class SharedCounter(object):
    """ A synchronized shared counter.
    The locking done by multiprocessing.Value ensures that only a single
    process or thread may read or write the in-memory ctypes object. However,
    in order to do n += 1, Python performs a read followed by a write, so a
    second process may read the old value before the new one is written by the
    first process. The solution is to use a multiprocessing.Lock to guarantee
    the atomicity of the modifications to Value.
    This class comes almost entirely from Eli Bendersky's blog:
    http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/
    """

    def __init__(self, n=0):
        self.count = mp.Value('i', n)

    def increment(self, n=1):
        """ Increment the counter by n (default = 1) """
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        """ Return the value of the counter """
        return self.count.value


# https://github.com/vterron/lemon/commit/9ca6b4b1212228dbd4f69b88aaf88b12952d7d6f
class MyQueue(multiprocessing.queues.Queue):
    """ A portable implementation of multiprocessing.Queue.
    Because of multithreading / multiprocessing semantics, Queue.qsize() may
    raise the NotImplementedError exception on Unix platforms like Mac OS X
    where sem_getvalue() is not implemented. This subclass addresses this
    problem by using a synchronized shared counter (initialized to zero) and
    increasing / decreasing its value every time the put() and get() methods
    are called, respectively. This not only prevents NotImplementedError from
    being raised, but also allows us to implement a reliable version of both
    qsize() and empty().
    """

    def __init__(self, *args, **kwargs):
        super(MyQueue, self).__init__(*args, ctx=mp.get_context(), **kwargs)
        self._size = SharedCounter(0)

    def put(self, *args, **kwargs):
        super(MyQueue, self).put(*args, **kwargs)
        self._size.increment(1)

    def get(self, *args, **kwargs):
        self._size.increment(-1)
        return super(MyQueue, self).get(*args, **kwargs)

    def qsize(self) -> int:
        """ Reliable implementation of multiprocessing.Queue.qsize() """
        return self._size.value

    def empty(self) -> bool:
        """ Reliable implementation of multiprocessing.Queue.empty() """
        return self.qsize() == 0
