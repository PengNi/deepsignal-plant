#! /usr/bin/env python
"""
calculate modification frequency at genome level
"""

from __future__ import absolute_import

import argparse
import os
import sys
import time

from .utils.txt_formater import ModRecord
from .utils.txt_formater import SiteStats
from .utils.txt_formater import split_key

import multiprocessing as mp
from .utils.process_utils import MyQueue as Queue
from .utils.process_utils import is_file_empty
import uuid

time_wait = 3


def calculate_mods_frequency(mods_files, prob_cf, contig_name=None):
    """
    call mod_freq from call_mods files
    :param mods_files: a list of call_mods files
    :param prob_cf:
    :param contig_name:
    :return: key2value obj
    """
    sitekeys = set()
    sitekey2stats = dict()

    if type(mods_files) is str:
        mods_files = [mods_files, ]

    count, used = 0, 0
    for mods_file in mods_files:
        with open(mods_file, 'r') as rf:
            for line in rf:
                words = line.strip().split("\t")
                mod_record = ModRecord(words)
                if contig_name is not None and mod_record._chromosome != contig_name:
                    continue
                if mod_record.is_record_callable(prob_cf):
                    if mod_record._site_key not in sitekeys:
                        sitekeys.add(mod_record._site_key)
                        sitekey2stats[mod_record._site_key] = SiteStats(mod_record._strand,
                                                                        mod_record._pos_in_strand,
                                                                        mod_record._kmer)
                    sitekey2stats[mod_record._site_key]._prob_0 += mod_record._prob_0
                    sitekey2stats[mod_record._site_key]._prob_1 += mod_record._prob_1
                    sitekey2stats[mod_record._site_key]._coverage += 1
                    if mod_record._called_label == 1:
                        sitekey2stats[mod_record._site_key]._met += 1
                    else:
                        sitekey2stats[mod_record._site_key]._unmet += 1
                    used += 1
                count += 1
    if contig_name is None:
        print("{:.2f}% ({} of {}) calls used..".format(used/float(count) * 100, used, count))
    else:
        print("{:.2f}% ({} of {}) calls used for {}..".format(used / float(count) * 100, used, count, contig_name))
    return sitekey2stats


def write_sitekey2stats(sitekey2stats, result_file, is_sort, is_bed):
    """
    write methylfreq of sites into files
    :param sitekey2stats:
    :param result_file:
    :param is_sort: sorted by poses
    :param is_bed: in bed format or not
    :return:
    """
    if is_sort:
        keys = sorted(list(sitekey2stats.keys()), key=lambda x: split_key(x))
    else:
        keys = list(sitekey2stats.keys())

    with open(result_file, 'w') as wf:
        # wf.write('\t'.join(['chromosome', 'pos', 'strand', 'pos_in_strand', 'prob0', 'prob1',
        #                     'met', 'unmet', 'coverage', 'Rmet', 'kmer']) + '\n')
        for key in keys:
            chrom, pos = split_key(key)
            sitestats = sitekey2stats[key]
            assert(sitestats._coverage == (sitestats._met + sitestats._unmet))
            if sitestats._coverage > 0:
                rmet = float(sitestats._met) / sitestats._coverage
                if is_bed:
                    wf.write("\t".join([chrom, str(pos), str(pos + 1), ".", str(sitestats._coverage),
                                        sitestats._strand,
                                        str(pos), str(pos + 1), "0,0,0", str(sitestats._coverage),
                                        str(int(round(rmet * 100, 0)))]) + "\n")
                else:
                    wf.write("%s\t%d\t%s\t%d\t%.3f\t%.3f\t%d\t%d\t%d\t%.4f\t%s\n" % (chrom, pos, sitestats._strand,
                                                                                     sitestats._pos_in_strand,
                                                                                     sitestats._prob_0,
                                                                                     sitestats._prob_1,
                                                                                     sitestats._met, sitestats._unmet,
                                                                                     sitestats._coverage, rmet,
                                                                                     sitestats._kmer))
            else:
                print("{} {} has no coverage..".format(chrom, pos))


def _read_file_lines(cfile):
    with open(cfile, "r") as rf:
        return rf.read().splitlines()


def _get_contigfile_name(wprefix, contig):
    return wprefix + "." + contig + ".txt"


def _split_file_by_contignames(mods_files, wprefix, contigs):
    contigs = set(contigs)
    wfs = {}
    for contig in contigs:
        wfs[contig] = open(_get_contigfile_name(wprefix, contig), "w")
    for input_file in mods_files:
        with open(input_file, "r") as rf:
            for line in rf:
                chrom = line.strip().split("\t")[0]
                if chrom not in contigs:
                    continue
                wfs[chrom].write(line)
    for contig in contigs:
        wfs[contig].flush()
        wfs[contig].close()


def _call_and_write_modsfreq_process(wprefix, prob_cf, result_file, issort, isbed, contigs_q, resfiles_q):
    print("process-{} -- starts".format(os.getpid()))
    while True:
        if contigs_q.empty():
            time.sleep(time_wait)
        contig_name = contigs_q.get()
        if contig_name == "kill":
            contigs_q.put("kill")
            break
        print("process-{} for contig-{} -- reading the input files..".format(os.getpid(), contig_name))
        input_file = _get_contigfile_name(wprefix, contig_name)
        if not os.path.isfile(input_file):
            print("process-{} for contig-{} -- the input file does not exist..".format(os.getpid(), contig_name))
            continue
        if is_file_empty(input_file):
            print("process-{} for contig-{} -- the input file is empty..".format(os.getpid(), contig_name))
        else:
            sites_stats = calculate_mods_frequency(input_file, prob_cf, contig_name)
            print("process-{} for contig-{} -- writing the result..".format(os.getpid(), contig_name))
            fname, fext = os.path.splitext(result_file)
            c_result_file = fname + "." + contig_name + "." + str(uuid.uuid1()) + fext
            write_sitekey2stats(sites_stats, c_result_file, issort, isbed)
            resfiles_q.put(c_result_file)
        os.remove(input_file)
    print("process-{} -- ends".format(os.getpid()))


def _concat_contig_results(contig_files, result_file):
    wf = open(result_file, "w")
    for cfile in sorted(contig_files):
        with open(cfile, 'r') as rf:
            for line in rf:
                wf.write(line)
        os.remove(cfile)
    wf.flush()
    wf.close()


def call_mods_frequency_to_file(args):
    print("[main]call_freq starts..")
    start = time.time()

    input_paths = args.input_path
    result_file = args.result_file
    prob_cf = args.prob_cf
    file_uid = args.file_uid
    issort = args.sort
    isbed = args.bed

    mods_files = []
    for ipath in input_paths:
        input_path = os.path.abspath(ipath)
        if os.path.isdir(input_path):
            for ifile in os.listdir(input_path):
                if file_uid is None:
                    mods_files.append('/'.join([input_path, ifile]))
                elif ifile.find(file_uid) != -1:
                    mods_files.append('/'.join([input_path, ifile]))
        elif os.path.isfile(input_path):
            mods_files.append(input_path)
        else:
            raise ValueError("--input_path is not a file or a directory!")
    print("get {} input file(s)..".format(len(mods_files)))

    contigs = None
    if args.contigs is not None:
        if os.path.isfile(args.contigs):
            contigs = sorted(list(set(_read_file_lines(args.contigs))))
        else:
            contigs = sorted(list(set(args.contigs.strip().split(","))))

    if contigs is None:
        print("read the input files..")
        sites_stats = calculate_mods_frequency(mods_files, prob_cf)
        print("write the result..")
        write_sitekey2stats(sites_stats, result_file, issort, isbed)
    else:
        print("start processing {} contigs..".format(len(contigs)))
        wprefix = os.path.dirname(os.path.abspath(result_file)) + "/tmp." + str(uuid.uuid1())
        print("generate input files for each contig..")
        _split_file_by_contignames(mods_files, wprefix, contigs)
        print("read the input files of each contig..")
        contigs_q = Queue()
        for contig in contigs:
            contigs_q.put(contig)
        contigs_q.put("kill")
        resfiles_q = Queue()
        procs_contig = []
        for _ in range(args.nproc):
            p_contig = mp.Process(target=_call_and_write_modsfreq_process,
                                  args=(wprefix, prob_cf, result_file, issort, isbed,
                                        contigs_q, resfiles_q))
            p_contig.daemon = True
            p_contig.start()
            procs_contig.append(p_contig)
        resfiles_cs = []
        while True:
            running = any(p.is_alive() for p in procs_contig)
            while not resfiles_q.empty():
                resfiles_cs.append(resfiles_q.get())
            if not running:
                break
        for p in procs_contig:
            p.join()
        try:
            assert len(contigs) == len(resfiles_cs)
        except AssertionError:
            print("!!!Please check the result files -- seems not all inputed contigs have result!!!")
        print("combine results of {} contigs..".format(len(resfiles_cs)))
        _concat_contig_results(resfiles_cs, result_file)
    print("[main]call_freq costs %.1f seconds.." % (time.time() - start))


def main():
    parser = argparse.ArgumentParser(description='calculate frequency of interested sites at genome level')
    parser.add_argument('--input_path', '-i', action="append", type=str, required=True,
                        help='an output file from call_mods/call_modifications.py, or a directory contains '
                             'a bunch of output files. this arg is in "append" mode, can be used multiple times')
    parser.add_argument('--file_uid', type=str, action="store", required=False, default=None,
                        help='a unique str which all input files has, this is for finding all input files '
                             'and ignoring the not-input-files in a input directory. if input_path is a file, '
                             'ignore this arg.')

    parser.add_argument('--result_file', '-o', action="store", type=str, required=True,
                        help='the file path to save the result')

    parser.add_argument('--contigs', action="store", type=str, required=False, default=None,
                        help="path of a file contains chromosome/contig names, one name each line; "
                             "or a string contains multiple chromosome names splited by comma. "
                             "default None, which means all chromosomes will be processed at one time. "
                             "If not None, one chromosome will be processed by one subprocess.")
    parser.add_argument('--nproc', action="store", type=int, required=False, default=1,
                        help="number of subprocesses used when --contigs is set. i.e., number of contigs processed "
                             "in parallel. default 1")

    parser.add_argument('--bed', action='store_true', default=False, help="save the result in bedMethyl format")
    parser.add_argument('--sort', action='store_true', default=False, help="sort items in the result")
    parser.add_argument('--prob_cf', type=float, action="store", required=False, default=0.5,
                        help='this is to remove ambiguous calls. '
                             'if abs(prob1-prob0)>=prob_cf, then we use the call. e.g., proc_cf=0 '
                             'means use all calls. range [0, 1], default 0.5.')

    args = parser.parse_args()

    call_mods_frequency_to_file(args)


if __name__ == '__main__':
    sys.exit(main())
