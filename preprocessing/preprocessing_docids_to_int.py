import fileinput
from argparse import ArgumentParser
# from multiprocessing import Pool
import os

# def process_page(inp):
#     # do processing
#     return (docid, title, url, passages)

def main(args):
    # filepath_ids_str = '/ivi/ilps/personal/akrasak/data/collections/collection_cast21_inc.tsv'
    filepath_output_ids_int = args.filepath_input+".int"
    filepath_output_mapping = args.filepath_input+".intmapping"

    if os.path.exists(filepath_output_ids_int):
        os.remove(filepath_output_ids_int)
    if os.path.exists(filepath_output_mapping):
        os.remove(filepath_output_mapping)

    # p = Pool(args.nthreads)
    # Collection = p.map(process_page, zip(process_page_params, RawCollection))

    fp = fileinput.input(args.filepath_input)
    for l in fp:
        line = l.strip()
        linesplt = line.split('\t')

        doc_id_original = linesplt[0].strip()
        doc_id_new = str(fp.filelineno()-1).zfill(9)

        if args.debug:
            print(f"Converting {doc_id_original} -> {doc_id_new}")

        with open(filepath_output_ids_int,'a') as writer:
            writer.write(f"{doc_id_new}\t{linesplt[1].strip()}\n")
        with open(filepath_output_mapping,'a') as writer:
            writer.write(f"{doc_id_new}\t{doc_id_original}\n")

if __name__ == "__main__":
    parser = ArgumentParser(description="docs2passages.")

    parser.add_argument('--nthreads', dest='nthreads', default=28, type=int)
    parser.add_argument('--filepath_input', dest='filepath_input', required=True, type=str)
    parser.add_argument("--debug", default=False, required=False, action="store_true", help="print stuff")
    args = parser.parse_args()
    main(args)
