#!/usr/bin/python
import argparse
from gff_reader import GFF3


transcript_gff_chrname2genome_name = {"1": "1",
                                      "2": "2",
                                      "3": "3",
                                      "4": "4",
                                      "5": "5",
                                      "Mt": "Mt",
                                      "Pt": "Pt",
                                      }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gff3", type=str, required=True, help="gff3 from ensembl")
    # parser.add_argument("--wfile", type=str, required=True, help="output file")

    args = parser.parse_args()
    print("reding gff3..")
    gff3 = GFF3(args.gff3)
    print("set relations..")
    print("get and write mapped coordinates..")
    wfile = args.gff3 + ".tx2gn_coords.pkl"
    trans_locinfo = gff3.save_coordinates_mapping(wfile)


def test():
    import pickle
    basepairs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}

    def complement_seq(dnaseq):
        rdnaseq = dnaseq[::-1]
        comseq = ''
        try:
            comseq = ''.join([basepairs[x] for x in rdnaseq])
        except Exception:
            print('something wrong in the dna sequence.')
        return comseq

    def complement_seq_noreverse(dnaseq):
        # rdnaseq = dnaseq[::-1]
        comseq = ''
        try:
            comseq = ''.join([basepairs[x] for x in dnaseq])
        except Exception:
            print('something wrong in the dna sequence.')
        return comseq

    class DNAReference:
        def __init__(self, reffile):
            self._contignames = []
            self._contigs = {}  # contigname 2 contigseq
            with open(reffile, 'r') as rf:
                contigname = ''
                contigseq = ''
                for line in rf:
                    if line.startswith('>'):
                        if contigname != '' and contigseq != '':
                            self._contigs[contigname] = contigseq
                            self._contignames.append(contigname)
                        contigname = line.strip()[1:].split(' ')[0]
                        contigseq = ''
                    else:
                        # turn to upper case
                        contigseq += line.strip().upper()
                self._contigs[contigname] = contigseq
                self._contignames.append(contigname)

        def getcontigs(self):
            return self._contigs

        def getcontignames(self):
            return self._contignames

    cdna_ref = DNAReference("Arabidopsis_thaliana.TAIR10.cdna.all.fa")
    cdna_contigs = cdna_ref.getcontigs()
    dna_ref = DNAReference("Arabidopsis_thaliana.TAIR10.dna.toplevel.fa")
    dna_contigs = dna_ref.getcontigs()
    print("cnda contigs: {}".format(len(cdna_contigs)))

    tx2mapinfo = pickle.load(open("Arabidopsis_thaliana.TAIR10.47.gff3.tx2gn_coords.pkl", "rb"))
    total, checked, mapping, seq_map = 0, 0, 0, 0
    for tx in tx2mapinfo.keys():
        txid, txlen, txchrom, txstrand, txcoords = tx2mapinfo[tx]
        if txid in cdna_contigs.keys():
            seq_cdna = cdna_contigs[txid]
            cdna_seq_len = len(seq_cdna)
            if txlen == cdna_seq_len:
                mapping += 1
                if transcript_gff_chrname2genome_name[txchrom] in dna_contigs.keys():
                    dna_chrom_seq = dna_contigs[transcript_gff_chrname2genome_name[txchrom]]
                    seq_from_dna = "".join([dna_chrom_seq[i] for i in txcoords])
                    if txstrand == "-":
                        seq_from_dna = complement_seq_noreverse(seq_from_dna)
                    if seq_from_dna == seq_cdna:
                        seq_map += 1
                    else:
                        print("=========================================")
                        print(txid, txlen, txchrom, txstrand, len(dna_chrom_seq))
                        # print("==\n{}".format(seq_cdna))
                        # print("==\n{}".format(seq_from_dna))
                else:
                    print("not in genome contigs: {}, {}, {}, {}".format(txid, txlen, txchrom, txstrand))
            checked += 1
        total += 1
    print("total: {}, check: {}, len_mapped: {}, seq_mapped: {}".format(total, checked, mapping, seq_map))


if __name__ == '__main__':
    main()
    # test()
