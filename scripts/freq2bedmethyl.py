#! /usr/bin/python
import argparse
import os


chromname_map_arab = {"NC_003070.9": "Chr1",
                      "NC_003071.7": "Chr2",
                      "NC_003074.8": "Chr3",
                      "NC_003075.7": "Chr4",
                      "NC_003076.8": "Chr5",
                      "NC_037304.1": "ChrM",
                      "NC_000932.1": "ChrC"}


def convert_dp_rmet_file2bedmethyl(args):
    freqinfo = dict()
    with open(args.freqfile, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            chrom = words[0]
            pos = int(words[1])
            strand = words[2]
            pos_in_strand = words[3]
            methy_prob = float(words[4])
            unmethy_prob = float(words[5])
            methy_cov = int(words[6])
            unmethy_cov = int(words[7])
            cov = int(words[8])
            rmet = float(words[9])
            kmer = words[10]

            if args.conv_chrom:
                chrom = chromname_map_arab[chrom]
            mkey = (chrom, pos, strand)
            if cov >= args.covcf:
                freqinfo[mkey] = (cov, rmet)
    fkeys = freqinfo.keys()
    if args.sort:
        fkeys = sorted(list(fkeys))
    fname, fext = os.path.splitext(args.freqfile)
    wfile = args.wfile if args.wfile is not None else fname + ".bed"
    wf = open(wfile, "w")
    for fkey in fkeys:
        chrom, pos, strand = fkey
        cov, rmet = freqinfo[fkey]
        wf.write("\t".join([chrom, str(pos), str(pos + 1), ".", str(cov), strand,
                            str(pos), str(pos + 1), "0,0,0", str(cov),
                            str(int(round(rmet * 100, 0)))]) + "\n")
    wf.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freqfile", type=str, action="store", required=True,
                        help="deepsignal-plant freq file")
    parser.add_argument("--covcf", type=int, required=False, default=1,
                        help="")
    parser.add_argument("--wfile", type=str, required=False, default=None)
    parser.add_argument("--conv_chrom", action='store_true', default=False,
                        help="convert chrom names of arab")
    parser.add_argument('--sort', action='store_true', default=False, help="sort items in the result")

    args = parser.parse_args()
    convert_dp_rmet_file2bedmethyl(args)


if __name__ == '__main__':
    main()
