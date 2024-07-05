
import argparse
import glob
import os
import random


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_path", default='<path-to-the-original-split>', help="file directory containing already splitted train, test and dev"
    )
    parser.add_argument(
        "--dest", default="<path-of-destination>", type=str, help="output directory"
    )
    parser.add_argument(
        "--ext", default="mp3", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    return parser


def main(args):
   
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    dir_path = os.path.realpath(args.split_path)
    clips_path = os.path.realpath('<path-of-audio-directory>')

    with open(os.path.join(args.split_path, 'train.tsv'), 'r') as read_train_f, \
         open(os.path.join(args.split_path, 'dev.tsv'), 'r') as read_dev_f, \
         open(os.path.join(args.dest, "train.tsv"), "w") as train_f, \
         open(os.path.join(args.dest, "valid.tsv"), "w") as dev_f:
            print(clips_path, file=train_f)
            print(clips_path, file=dev_f)

            train_labs = set()
            valid_labs = set()
            for idx, line in enumerate(read_train_f):
                if idx:
                    l = line.split('\t')
                    print("{}\t{}".format(l[1].strip(), l[3].strip()), file=train_f)
                    train_labs.add(l[3].strip())
                if idx > 10:
                    break
                            
            for idx, line in enumerate(read_dev_f):
                if idx:
                    l = line.split('\t')
                    print("{}\t{}".format(l[1].strip(), l[3].strip()), file=dev_f)
                    valid_labs.add(l[3].strip())
                if idx>10:
                    break

            print("Train labels: {}".format(train_labs))
            print("Valid labels: {}".format(valid_labs))

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
