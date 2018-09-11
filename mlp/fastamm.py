import math
import re

class FastaMM:
    def __init__(self, fasta, segmentLength, windowLen, K):
        self.fasta = fasta
        self.segmentLength = segmentLength
        self.windowLen = windowLen
        self.K = K

    def init(self):
        self.genomeDict = self._readGenome(self.fasta)
        self.table = self._segmentify(self.genomeDict)


    def allClassificationJob(self):
        for k,v in self.table.items():
            segment = self.genomeDict[v["name"]][  v["start"] : v["end"]  ]
            yield k, segment

    def writeMeta(self, directoryName):
        with open(directoryName+"/classify_detail.log", "w") as fout:
            fout.write("segmentLength: %d\n" % self.segmentLength)
            fout.write("windowLength: %d\n" % self.windowLen)
            fout.write("numClasses: %d\n" % len(self.table))
            fout.write("K: %d\n" % self.K)
            fout.write("endMeta\n")
            for k,v in self.table.items():
                fout.write("%d$$$%s$$$%d$$$%d\n" % (k, v['name'], v['start'], v['end']))

    def _readGenome(self, fasta):
        dictionary = {}
        with open(fasta, "r") as fastafile:
            for line in fastafile:
                if line.startswith(">"):
                    key = line.strip()
                    dictionary[key] = ""
                else:
                    dictionary[key] = dictionary[key] + line.strip()
        return dictionary

    def _segmentify(self, genomeDictionary):
        meta = {}
        for key,genome in genomeDictionary.items():
            numSegments, segments = self.__makeSegments(genome, self.segmentLength, self.windowLen)
            for segment in segments:
                each = {
                    "name": key,
                    "start": segment[0],
                    "end": segment[1]
                }
                meta[len(meta)] = each
        return meta

    def ___makeSegments(self, genome, segmentLength, windowLen):
        """
        Divides the genome into segmentLength ± (windowLen/2).
        Returns (start,end) of segments as list and total of those segments
        """
        genomeLength = len(genome)
        numSegments = math.ceil(float(genomeLength) / float(segmentLength))
        segments = []
        for i in range(numSegments):
            start = max(0, i * segmentLength - math.floor(windowLen / 2.0))
            end = min((i + 1) * segmentLength + math.ceil(windowLen / 2.0) - 1, genomeLength)
            segment = (start, end)
            segments.append(segment)
        return numSegments, segments

    def __makeSegments(self, genome, segmentLength, windowLen):
        """
        Given one contig of genome, it splits the contig into contigs whenever there's a padding ("NNNN+") >= windowLen.
        For each of those splitted contigs, it calls ___makeSegments to divide them into segments of length segmentLength ± (windowLen/2).
        """
        padding = "N"*windowLen + "+"
        unpadded_segments = []
        start = 0
        for r in [(m.start(),m.end()) for m in re.finditer(padding, genome)]:
            unpadded_segments.append((start, r[0]))
            start = r[1]
        unpadded_segments.append((start, len(genome)))

        total_segments = 0
        segments = []
        for r_unpadded in unpadded_segments:
            segment = genome[r_unpadded[0]:r_unpadded[1]]
            n, this_segments = self.___makeSegments(segment, segmentLength, windowLen)
            segments += [(start+r_unpadded[0], end+r_unpadded[0]) for (start,end) in this_segments]
            total_segments += n
        return total_segments, segments



if __name__=="__main__":
    m = FastaMM("sample_classification_run/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa", 10000, 200, 7)
    m.init()
    # for x in m.allClassificationJob():
    #     print(x)
    m.writeMeta("sample_classification_run/")