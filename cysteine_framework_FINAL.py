# November 13, 2023 -- Mariam Gelashvili
# Cysteine Framework ID

#!/usr/bin/env python3
import regex as re
import csv
import sys
import itertools


def get_frame(path_to_file):
    seq = []
    raw_frame = []
    to_remove = "-"
    with open(path_to_file, 'r') as indata:
        for line in csv.reader(indata, delimiter=','):
            str_seq = line[0]
            seq.append(str_seq)
            x = re.sub("[ARNDEQGHILKMFPSTWYV]", '-', str_seq)
            y = ''.join(k if k in to_remove else "".join(v) for k,v in itertools.groupby(x, lambda c: c))
            raw_frame.append(y)
    return seq, raw_frame


def fix_frames(pre_frames):
    filtered_frames = []
    for element in pre_frames:
        if element[0] == '-' and element[-1] == '-':
            filtered_frames.append(element[1:-1])
        else:
            if element[0] == '-':
                filtered_frames.append(element[1:])
            elif element[-1] == '-':
                filtered_frames.append(element[:-1])
            else:
                filtered_frames.append(element)
    return filtered_frames


def match_framework(seq_frame):
    frame_number = []
    frame_dict = {1: 'CC-C-C',
                  2: 'CCC-C-C-C',
                  3: 'CC-C-C-CC',
                  4: 'CC-C-C-C-C',
                  5: 'CC-CC',
                  '6 and 7': 'C-C-CC-C-C',
                  8: 'C-C-C-C-C-C-C-C-C-C',
                  9: 'C-C-C-C-C-C',
                  11: 'C-C-CC-CC-C-C',
                  12: 'C-C-C-C-CC-C-C',
                  13: 'C-C-C-CC-C-C-C',
                  14: 'C-C-C-C',
                  15: 'C-C-CC-C-C-C-C',
                  16: 'C-C-CC',
                  17: 'C-C-CC-C-CC-C',
                  18: 'C-C-CC-CC',
                  19: 'C-C-C-CCC-C-C-C-C',
                  20: 'C-CC-C-CC-C-C-C-C',
                  21: 'CC-C-C-C-CC-C-C-C',
                  22: 'C-C-C-C-C-C-C-C',
                  23: 'C-C-C-CC-C',
                  24: 'C-CC-C',
                  25: 'C-C-C-C-CC',
                  26: 'C-C-C-C-CC-CC',
                  27: 'C-C-C-CCC-C-C',
                  28: 'C-C-C-CC-C-C-C-C-C',
                  29: 'CCC-C-CC-C-C',
                  30: 'C-C-CCC-C-C-C-CC',
                  32: 'C-CC-C-C-C',
                  33: 'C-C-C-C-C-C-C-C-C-C-C-C',
                  }
    # 10 is not included becuase it has [PO], it will show up under framework 5 with the above dictionary
    # there is no famework 31 listed on conoserver
    for element in seq_frame:
        if element in frame_dict.values():
            for k, v in frame_dict.items():
                if element == v:
                    frame_number.append(k)
        else:
            frame_number.append('novel')
    return frame_number


def main():
    sequences, corder = get_frame(sys.argv[1])
    fin_corder = fix_frames(corder)
    num_match = match_framework(fin_corder)
    data_to_file = zip(sequences, fin_corder, num_match)

    with open(r"../../Downloads/VenomToxinDatabase/intermediateFiles/output.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        for sequence, frame, nomer in data_to_file:
            writer.writerow([sequence, frame, nomer])
        csvfile.close()

if __name__ == "__main__":
    main()
