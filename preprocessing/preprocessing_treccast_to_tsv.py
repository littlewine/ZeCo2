from bs4 import BeautifulSoup
from transformers import BertTokenizer
import re
import argparse
import os
import json
import pandas as pd
import shutil

def main(args):

    # Define paths
    if args.filepath_output is None:
        args.filepath_output = args.filepath_trecweb.replace("trecweb", "tsv")

    if args.filepath_trecweb==args.filepath_output:
        raise ValueError("output path should be != than original collection path ")

    if os.path.exists(args.filepath_output):
        os.remove(args.filepath_output)
        print(f"removed already existing tsv @ {args.filepath_output}")

    # copy files to local worker node
    print("copying files to slurm worker node")

    worker_storage_path = '/ssdstore/antonis/'
    args.filepath_trecweb_worker = os.path.join(worker_storage_path, args.filepath_trecweb.split('/')[-1])
    args.filepath_output_worker = os.path.join(worker_storage_path, args.filepath_output.split('/')[-1])

    if not os.path.exists(worker_storage_path):
        print(f"Creating dir {worker_storage_path}")
        os.mkdir(worker_storage_path)
    if os.path.exists(args.filepath_trecweb_worker): # delete previous filepath to ensure data integrity
        os.remove(args.filepath_trecweb_worker)
    shutil.copy(args.filepath_trecweb, args.filepath_trecweb_worker)

    # Initialize files, dicts etc.
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    lengths = dict()
    txt = ''
    re_docid = "<DOCNO>(.*)</DOCNO>"
    re_doctitle = "<TITLE>(.*)</TITLE>"
    re_passage_id = "<passage id=(\d+)>"

    with open(args.filepath_trecweb, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("<DOC>"):
                doc_id,doc_title = None, None
                passage = dict()
            elif line.startswith("<DOCNO>"):
                doc_id = re.search(re_docid,line).groups()[0]
            elif line.startswith("<TITLE>"):
                try:
                    doc_title = re.search(re_doctitle, line).groups()[0]
                except:
                    print(f"error in docid {doc_id}, skipping doc")
                    continue
                # count title length
                lengths[doc_id] = len(tokenizer.encode(doc_title))

            elif line.startswith("<passage id"):
                passage_id = re.search(re_passage_id,line).groups()[0]
                doc_passage_id = f"{doc_id}-{passage_id}"

                next_line = next(f)
                next_line = next_line.strip()
                while next_line != "</passage>": # loop lines until you find </passage> tag and concat to txt
                    if txt.strip().endswith('.'):
                        txt += f" {next_line.strip()}"
                    else:
                        txt += f". {next_line.strip()}"
                    next_line = next(f)
                    next_line = next_line.strip()

                #
                passage[doc_passage_id] = txt
                txt = ''

                # # Old parsing way (not breaking, but taking first line only)
                # # parse passage text
                # passage[doc_passage_id] = next_line.strip()
                # # find end tag / raise error if not in next line
                # next_line = next(f)
                # try:
                #     assert next_line.strip()=="</passage>"
                # except AssertionError:
                #     print(f"multiple lines inside <passage> tags in doc {doc_id}, passage {passage_id}")
                #     print('line: ',next_line)

            elif line.startswith("</DOC>"): # end of document tag
                with open(args.filepath_output_worker, 'a') as writer:
                    for id, p in passage.items():
                        if doc_title.strip().endswith('.'):
                            passage_with_title = f" {doc_title} {p}"
                        else:
                            passage_with_title = f" {doc_title} {p}"

                        writer.write(f"{id} \t {passage_with_title}\n")

                        # count total passage tokens
                        lengths[doc_passage_id] = len(tokenizer.encode(passage_with_title))

    # Write token statistics
    with open(args.filepath_output_worker + ".lengths", 'w') as file:
        file.write(json.dumps(lengths))  # use `json.loads` to do the reverse

    # print max length or stats of how many above 128/512
    ser_len = pd.Series(lengths)
    print(f"Passages with more than 180 tokens: {(ser_len>180).sum()} ({(ser_len>180).sum()*100/len(ser_len)}%)")
    print(f"Passages with more than 256 tokens: {(ser_len>256).sum()} ({(ser_len>256).sum()*100/len(ser_len)}%)")
    print(f"Passages with more than 512 tokens: {(ser_len>512).sum()} ({(ser_len>512).sum()*100/len(ser_len)}%)")

    # copy files back to normal storage
    shutil.copy(args.filepath_output_worker, args.filepath_output)

    print("Please delete ")

    return args.filepath_output, args.filepath_output_worker

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath_trecweb",
                        type=str,
                        required=True,
                        help="input path")

    parser.add_argument("--filepath_output",
                        type=str,
                        required=False,
                        default=None,
                        help="output path")

    args = parser.parse_args()
    main(args)
