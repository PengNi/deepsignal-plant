import os
import argparse
import gzip


basepairs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
             'W': 'W', 'S': 'S', 'M': 'K', 'K': 'M', 'R': 'Y',
             'Y': 'R', 'B': 'V', 'V': 'B', 'D': 'H', 'H': "D",
             'Z': 'Z'}

iupac_alphabets = {'A': ['A'], 'T': ['T'], 'C': ['C'], 'G': ['G'],
                   'R': ['A', 'G'], 'M': ['A', 'C'], 'S': ['C', 'G'],
                   'Y': ['C', 'T'], 'K': ['G', 'T'], 'W': ['A', 'T'],
                   'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'],
                   'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'],
                   'N': ['A', 'C', 'G', 'T']}


def complement_seq(dnaseq):
    rdnaseq = dnaseq[::-1]
    comseq = ''
    try:
        comseq = ''.join([basepairs[x] for x in rdnaseq])
    except Exception:
        print('something wrong in the dna sequence.')
    return comseq


def _convert_motif_seq(ori_seq, is_dna=True):
    outbases = []
    for bbase in ori_seq:
        if is_dna:
            outbases.append(iupac_alphabets[bbase])
        else:
            raise ValueError()

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


def get_c_motif2seq():
    motif2seq = dict()
    motif2seq["CGN"] = set(_convert_motif_seq("CGN"))
    motif2seq["CGN"].add("CGN")

    motif2seq["CHG"] = set(_convert_motif_seq("CHG"))
    motif2seq["CHG"].add("CHG")

    motif2seq["CHH"] = set(_convert_motif_seq("CHH"))
    motif2seq["CHH"].add("CHH")
    return motif2seq


def _split_callmods_file(callmods_file):
    fname, fext = os.path.splitext(callmods_file) if not callmods_file.endswith(".gz") else os.path.splitext(callmods_file[:-3])

    motif2seq = get_c_motif2seq()
    motifs = list(motif2seq.keys())
    seq2motif = dict()
    for motif in motifs:
        seqs = list(motif2seq[motif])
        for seq in seqs:
            seq2motif[seq] = motif

    motif2idx = {motifs[i]: i for i in range(len(motifs))}
    wfobjs = []
    for motif in motifs:
        if motif.startswith("CG"):
            motifstr = "CG"
        else:
            motifstr = motif
        wfobjs.append(open(fname + "." + motifstr + fext, "w"))

    count, count_fail = 0, 0
    # call_mods.tsv
    if callmods_file.endswith(".gz"):
        rf = gzip.open(callmods_file, "rt")
    else:
        rf = open(callmods_file, "r")
    for line in rf:
        count += 1
        words = line.strip().split("\t")
        kmer = words[-1]
        cenpos = len(kmer)//2
        seq = kmer[cenpos:(cenpos+3)]
        try:
            wfobjs[motif2idx[seq2motif[seq]]].write(line)
        except KeyError:
            count_fail += 1
            print("seq: {}, line: {}".format(seq, line.strip()))
    rf.close()
    for wf in wfobjs:
        wf.flush()
        wf.close()
    print("total lines: {}, failed lines: {}".format(count, count_fail))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--callmods_file", type=str, required=True,
                        help="call_mods file from call_mods module")

    args = parser.parse_args()
    _split_callmods_file(args.callmods_file)


if __name__ == '__main__':
    main()
