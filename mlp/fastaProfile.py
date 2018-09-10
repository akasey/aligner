import re
import argparse
import numpy as np

def count_errors(cigar):
    errors = 0
    for elements in re.findall(r'(\d+)([A-Z=X]{1})', cigar):
        if elements[1] != 'M' and elements[1] != '=':
            errors = errors + int(elements[0])
    return errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam", type=str,
                        help="TrueAlignment sam file")
    args = parser.parse_args()

    read_length = []
    error_percentage = []
    with open(args.sam, "r") as sam:
        for line in sam.readlines():
            if not line.startswith("@"):
                fields = line.strip().split("\t")
                cigar = fields[5]
                read = fields[9]
                error_count = count_errors(cigar)
                error_percentage.append(error_count/len(read))
                read_length.append(len(read))
                # print(error_count, cigar, read)

    read_length = np.array(read_length)
    error_percentage = np.array(error_percentage)
    print("Read Length: ", np.average(read_length), np.min(read_length), np.max(read_length))
    print("Error: ", np.average(error_percentage), np.min(error_percentage), np.max(error_percentage))