import argparse
import re


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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="sample_classification_run/",
        help="directory of predictedLocations file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="jaccard threshold")

    FLAGS, unparsed = parser.parse_known_args()

    main()