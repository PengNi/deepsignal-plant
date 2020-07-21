#!/usr/bin/python
import argparse
import pickle


def read_posfile(posfile):
    poses = []
    with open(posfile, "r") as rf:
        for line in rf:
            words = line.strip().split("\t")
            chrom, pos, strand = words[0], int(words[1]), words[2]
            poses.append((chrom, pos, strand))
    return poses


def map_genomeloc2transcriptloc(args):
    poses = read_posfile(args.posfile)
    posidx2txposes = dict()
    for i in range(0, len(poses)):
        posidx2txposes[i] = []

    tx2mapinfo = pickle.load(open(args.tx2gn_pkl, "rb"))
    for txname in tx2mapinfo.keys():
        txid, txlen, tx_chr, tx_strand, txlocs = tx2mapinfo[txname]
        txlocset = set(txlocs)
        txloc2coordinate = dict()
        for idx in range(0, len(txlocs)):
            txloc2coordinate[txlocs[idx]] = idx
        for idx in range(0, len(poses)):
            chrom, pos, strand = poses[idx]
            if chrom == tx_chr and strand == tx_strand and pos in txlocset:
                transloc = txloc2coordinate[pos]
                posidx2txposes[idx].append((txid, transloc, "+"))

    wfile = args.posfile + ".tx2gn.pos.info"
    wf = open(wfile, "w")
    mapped_cnt = 0
    print("===unmapped poses:")
    for posidx in posidx2txposes.keys():
        if len(posidx2txposes[posidx]) > 0:
            mapped_cnt += 1
            for tranpos in posidx2txposes[posidx]:
                wf.write("\t".join(list(map(str, tranpos)) + list(map(str, poses[posidx]))) + "\n")
        else:
            chrom, pos, strand = poses[posidx]
            print("{}\t{}\t{}".format(chrom, pos, strand))
    wf.close()
    print("===================")
    print("pos: {}, mapped_pos: {}".format(len(poses), mapped_cnt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posfile", type=str, required=True, help="genome pos file: chrom, pos, strand")
    parser.add_argument("--tx2gn_pkl", type=str, required=True, help="transcript2genome pickle")

    args = parser.parse_args()
    map_genomeloc2transcriptloc(args)


if __name__ == '__main__':
    main()
