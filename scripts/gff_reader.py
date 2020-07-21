# gff class
import pickle


def extract_region_by_attri(gff3_eles, attri_name, attri_val):
    outstrs = []
    for gff_ele in gff3_eles:
        if attri_name in gff_ele.get_attr_keys() and gff_ele.get_attrs()[attri_name] == attri_val:
            outstrs.append(gff_ele.print_str())
    print("extract {} regions by {} ({})".format(len(outstrs), attri_name, attri_val))
    return outstrs


def get_kinds_of_a_attri(gff3_eles, attri_name):
    attri_kinds = set()
    for gff_ele in gff3_eles:
        if attri_name in gff_ele.get_attr_keys():
            attri_kinds.add(gff_ele.get_attrs()[attri_name])
    return attri_kinds


class GFF3Element:
    def __init__(self, fields):
        # "chromosome", "source", "feature", "start", "end",
        #              "score", "strand", "phase", "attributes"
        self._chromosome = fields[0]
        self._source = fields[1]
        self._feature = fields[2]
        self._start = int(fields[3]) - 1  # turn to 0-based
        self._end = int(fields[4])
        self._score = fields[5]
        self._strand = fields[6]
        self._phase = fields[7]
        self._attributes = fields[8]

        self._set_gene_attrs()

        # only for ensembl gff3
        self._ensemblid = None
        self._parent = None
        self._rank = None

        self._set_relationinfo()

    def _set_gene_attrs(self):
        self._attrs = dict()
        for attr_kv in self._attributes.strip().split(";"):
            if attr_kv != "":
                attr = attr_kv.strip().split("=")
                self._attrs[attr[0]] = attr[1]
        self._attr_keys = set(self._attrs.keys())

    def _set_relationinfo(self):
        if "ID" in self._attr_keys:
            self._ensemblid = self._attrs["ID"].strip().split(":")[1]
        elif "Name" in self._attr_keys:
            self._ensemblid = self._attrs["Name"]
        if "Parent" in self._attr_keys:
            self._parent = self._attrs["Parent"]
        if "rank" in self._attr_keys:
            self._rank = int(self._attrs["rank"])

    def get_chromosome(self):
        return self._chromosome

    def get_source(self):
        return self._source

    def get_feature(self):
        return self._feature

    def get_start(self):
        return self._start

    def get_end(self):
        return self._end

    def get_score(self):
        return self._score

    def get_strand(self):
        return self._strand

    def get_phase(self):
        return self._phase

    def get_attributes(self):
        return self._attributes

    def get_attrs(self):
        return self._attrs

    def get_attr_keys(self):
        return self._attr_keys

    def get_id(self):
        if "ID" in self._attr_keys:
            return self._attrs["ID"]
        elif "Name" in self._attr_keys:
            return self._attrs["Name"]
        return None

    def get_ensemblid(self):
        return self._ensemblid

    def get_parent(self):
        return self._parent

    def get_rank(self):
        return self._rank

    def print_str(self):
        # id, chrom, start(0-based), end(0-based), strand, feature, attributes
        return "\t".join([self.get_id(), self.get_chromosome(), str(self.get_start()),
                          str(self.get_end()), self.get_strand(), self.get_feature(),
                          self.get_attributes()])


class GFF3:
    def __init__(self, filepath):
        self.eles = []
        self._features = set()
        self._gt2idx = dict()  # gene or transcript id 2 index in self.eles, only for ensembl gff3
        with open(filepath, "r") as rf:
            for line in rf:
                if not line.startswith("#"):
                    words = line.strip().split("\t")
                    gffele = GFF3Element(words)
                    self.eles.append(gffele)
                    self._features.add(gffele.get_feature())
                    if gffele.get_id() is not None:
                        if gffele.get_id().startswith("transcript") or gffele.get_id().startswith("gene"):
                            self._gt2idx[gffele.get_id()] = len(self.eles) - 1

        self._parent2exonidx = dict()
        for eleidx in range(0, len(self.eles)):
            gffele = self.eles[eleidx]
            parentid = gffele.get_parent()
            if gffele.get_feature() == "exon" and parentid is not None:
                if parentid not in self._parent2exonidx.keys():
                    self._parent2exonidx[parentid] = []
                self._parent2exonidx[parentid].append(eleidx)

    def get_eles(self):
        return self.eles

    def get_features(self):
        return self._features

    def get_gt2idx(self):
        return self._gt2idx

    def get_parent2exonidx(self):
        return self._parent2exonidx

    def map_coordinates_transcript2genome(self, transcriptid):
        transcript_ele = self.eles[self._gt2idx[transcriptid]]
        exonidxs = self._parent2exonidx[transcriptid]
        exonid_ranks = []
        for exonidx in exonidxs:
            exon_ele = self.eles[exonidx]
            exonid_ranks.append((exonidx, exon_ele.get_rank()))
        exonid_ranks = sorted(exonid_ranks, key=lambda x: x[1])

        strand = transcript_ele.get_strand()
        transcript_len = 0
        transcript_loc_in_genome = []
        for exonidx, rank in exonid_ranks:
            exon_ele = self.eles[exonidx]
            genome_start, genome_end = exon_ele.get_start(), exon_ele.get_end()
            genome_locs = [i for i in range(genome_start, genome_end)]
            if strand == "-":
                genome_locs = genome_locs[::-1]
            transcript_loc_in_genome += genome_locs
            exon_len = genome_end - genome_start
            transcript_len += exon_len
        return transcript_ele.get_ensemblid(), transcript_len, transcript_ele.get_chromosome(), \
            strand, transcript_loc_in_genome

    def save_coordinates_mapping(self, pkl_path):
        transcriptid2posinfo = dict()
        for eid in self._parent2exonidx.keys():
            posinfo = self.map_coordinates_transcript2genome(eid)
            if posinfo is not None:
                transcriptid2posinfo[eid] = posinfo
        print("mapped {} transcript coordinates..".format(len(transcriptid2posinfo)))
        pickle.dump(transcriptid2posinfo, open(pkl_path, "wb"))
        return transcriptid2posinfo
