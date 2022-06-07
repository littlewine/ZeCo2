from argparse import ArgumentParser
import os
def main(args):

    output_filepath=f"{args.collection_filepath}.intmapping"

    if os.path.exists(output_filepath):
        os.remove(output_filepath)
        print(f"removed {output_filepath}")

    with open(args.collection_filepath) as f:
        for line_idx, line in enumerate(f):
            intid, _ = line.strip().split('\t')
            with open(output_filepath, 'a') as writer:
                writer.write(f"{intid}\t{args.prefix}_{intid}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=".")

    parser.add_argument('--collection_filepath', required=True, type=str)
    parser.add_argument('--prefix', required=True, type=str)

    args = parser.parse_args()
    main(args)
