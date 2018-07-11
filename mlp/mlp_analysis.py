import argparse
import re
import math


def main():
    fin = open(FLAGS.data_dir +"/predictedLocations", "r")
    total_data = 0
    predicted_count = 0
    correct_prediction = 0
    distance = 0
    distances = []
    for line in fin.readlines():
        splits = line.split("$$")
        true_position = int(splits[1])
        if true_position != -123:
            total_data += 1
        if len(splits) == 3:
            predicted_locations = splits[2].split(",")
            if len(predicted_locations) >= 2:
                predicted_count += 1
                highest_score = predicted_locations[-2]
                pred = [float(x) for x in re.split("\(|\)", highest_score) if len(x.strip()) > 0]
                location = int(pred[0])
                threshold = pred[1]
                diff = abs(location - true_position)
                distances.append(diff)
                distance += diff
                if diff <= 10:
                    correct_prediction += 1
    print("average distance", distance/total_data)
    print("correct", correct_prediction, "total", total_data)
    recall = correct_prediction/total_data
    precision = correct_prediction/predicted_count
    print("Recall", recall, "precision", precision, "sensitivity", predicted_count/total_data, "F1", 2*recall*precision/(recall+precision))

class OriginalRead:
    def __init__(self):
        self.contig = ""
        self.read1_start = -1
        self.read2_start = -1
        self.read1_forward = False
        self.read2_forward = False
        self.read1_rand = False
        self.read2_rand = False

    @staticmethod
    def fromString(sam_col1):
        splits = sam_col1.split("_")
        splits = list(reversed(splits))
        read = OriginalRead()
        read.contig = "_".join(reversed(splits[9:]))
        read.read1_start = int(splits[8])
        read.read2_start = int(splits[7])
        read.read1_forward = splits[6] == "0"
        read.read2_forward = splits[5] == "0"
        read.read1_rand = splits[4] == "1"
        read.read2_rand = splits[3] == "1"
        return read

    def __str__(self):
        return "%s r1:(start:%d forward:%r rand:%r) r2:(start:%d forward:%r rand:%r)" % (self.contig, self.read1_start, self.read1_forward, self.read1_rand, self.read2_start, self.read2_forward, self.read2_rand)

class PredictedRead:
    def __init__(self):
        self.contig = ""
        self.read1_start = -1
        self.read1_forward = False
        self.read1_unmapped = False

    @staticmethod
    def fromSplits(splits):
        read = PredictedRead()
        read.contig = splits[2]
        read.read1_start = int(splits[3])
        flags = int(splits[1])
        read.read1_unmapped = flags & 4 > 0
        read.read1_forward = flags & 16 == 0
        return read

    def __str__(self):
        return "%s r1:(start: %d forward: %r unmapped: %r)" % (self.contig, self.read1_start, self.read1_forward, self.read1_unmapped)


def analysis():
    fin = open(FLAGS.sam, "r")
    total_data, predicted_count, correct_prediction = 0,0,0
    for line in fin.readlines():
        if not line.startswith("@"):
            total_data += 1
            splits = line.strip().split("\t")
            original_read = OriginalRead.fromString(splits[0])
            # print(original_read, splits[0])
            predicted_read = PredictedRead.fromSplits(splits)
            # print(predicted_read, line)
            if not predicted_read.read1_unmapped:
                predicted_count += 1
                if abs(predicted_read.read1_start - original_read.read1_start) <= FLAGS.threshold and predicted_read.read1_forward == original_read.read1_forward:
                    correct_prediction += 1

    print("correct", correct_prediction, "predicted_count", predicted_count, "total", total_data)
    recall = correct_prediction/total_data
    precision = correct_prediction/predicted_count
    print("Recall", recall, "precision", precision, "sensitivity", predicted_count/total_data, "F1", 2*recall*precision/(recall+precision))



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--sam",
        type=str,
        default="sample_classification_run/alignment.sam",
        help="directory of predictedLocations file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=10,
        help="distance threshold")

    FLAGS, unparsed = parser.parse_known_args()

    analysis()