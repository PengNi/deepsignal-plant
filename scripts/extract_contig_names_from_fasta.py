import argparse


def get_contignams_from_genome_fasta(genomefa, outfile):
    contigs = []
    with open(genomefa, "r") as rf:
        for line in rf:
            if line.startswith(">"):
                contigname = line.strip()[1:].split(' ')[0]
                contigs.append(contigname)
    wfile = outfile if outfile is not None else genomefa + ".contig_names.txt"
    wf = open(wfile, "w")
    for contigname in contigs:
        wf.write(contigname + "\n")
    wf.flush()
    wf.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", "-r", type=str, required=True,
                        help="genome reference, .fasta or .fa")
    parser.add_argument("--out", "-o", type=str, required=False, default=None,
                        help="output file path")

    args = parser.parse_args()
    get_contignams_from_genome_fasta(args.ref, args.out)


if __name__ == '__main__':
    main()
