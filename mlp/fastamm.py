import math

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

    def __makeSegments(self, genome, segmentLength, windowLen):
        genomeLength = len(genome)
        numSegments = math.ceil(float(genomeLength) / float(segmentLength))
        segments = []
        for i in range(numSegments):
            start = max(0, i * segmentLength - math.floor(windowLen / 2.0))
            end = min((i + 1) * segmentLength + math.ceil(windowLen / 2.0) - 1, genomeLength)
            segment = (start, end)
            segments.append(segment)
        return numSegments, segments




if __name__=="__main__":
    m = FastaMM("sample_classification_run/sequence.fasta", 10000, 200)
    m.init()
    for x in m.allClassificationJob():
        print(x)
    m.writeMeta("sample_classification_run/")