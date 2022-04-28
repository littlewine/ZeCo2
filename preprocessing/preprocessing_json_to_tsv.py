from argparse import ArgumentParser
import fileinput
import json
import os

def main(args):
    fp = fileinput.input(args.filepath_input)

    filepath_output_collection = args.filepath_input.replace('.json','.tsv.int')
    filepath_output_mapping = args.filepath_input.replace('.json','.tsv.intmapping')

    if os.path.exists(filepath_output_collection):
        os.remove(filepath_output_collection)
        print(f"Removed {filepath_output_collection}")
    if os.path.exists(filepath_output_mapping):
        os.remove(filepath_output_mapping)
        print(f"Removed {filepath_output_mapping}")

    i=0
    for l in fp:
        line = l.strip()
        ljson = json.loads(line)

        if ('contents' not in ljson.keys()) or ('id' not in ljson.keys()):
            print(f'\n\n****\n\nerror in line {i}: \n\n'
                  f'{line}\n\n****')
        # Write collection file
        docidint=i
        docidint=str(docidint).zfill(9) # pad with 0s and convert to str

        # Write collection
        with open(filepath_output_collection, 'a') as f:
            f.write(f"{docidint}\t{ljson['contents'].strip()}\n")

        # Write mapping
        with open(filepath_output_mapping, 'a') as f:
            f.write(f"{docidint}\t{ljson['id'].strip()}\n")

    # increase i
        i+=1
    # TODO: remove from file lines not starting with 9 digits (or replace \n with whitespace)
    # from bash:
    # sed -n '/^[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].*/p' {filepath.tsv} > {filepath.tsv.corrected}

if __name__ == "__main__":
    parser = ArgumentParser(description="Convert json (CAR) format to ColBERT tsv format")

    # parser.add_argument('--nthreads', dest='nthreads', default=28, type=int)
    parser.add_argument('--filepath_input', dest='filepath_input', required=True, type=str)
    parser.add_argument("--debug", default=False, required=False, action="store_true", help="print stuff")
    args = parser.parse_args()
    main(args)
