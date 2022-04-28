from ColBERT.paths import path_queries
import json
import ast
from collections import defaultdict
from transformers import BertTokenizerFast
import os
from argparse import ArgumentParser
import re
import pandas as pd

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever

def main(args):
    raw_queries_dict = pd.read_csv(path_queries['cast19']['raw'], sep='\t', header=None, index_col=0).to_dict()[1]
    full_conv_queries_dict = dict()
    queries = dict()

    tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')

    output_path = path_queries['cast19']['full_conv']

    # get conversations, canonical answers & lengths

    if os.path.exists(output_path):
        os.remove(output_path)
        print(f'removed previous file @ {output_path}')

    conv_ids = set()
    for qid,utterance in raw_queries_dict.items():
        conv_id,turn_id = qid.split('_')
        conv_id = int(conv_id)
        turn_id = int(turn_id)

        conv_ids.add(conv_id)
        # full_conv_queries_dict[qid] = ''
        # for previous_turn_id in range(1,turn_id+1):
        #     if args.debug:
        #         print(f"Constructing query {qid}")

        # Do it recursively
        if turn_id==1:
            full_conv_queries_dict[qid] = raw_queries_dict[qid]
        else:
            previous_qid = f"{conv_id}_{turn_id-1}"
            #TODO: optionally add query token from colbert? (will appear double)
            full_conv_queries_dict[qid] = full_conv_queries_dict[previous_qid] + ' [SEP] ' + raw_queries_dict[qid]

    #TODO: write out
    #TODO: + check if I have underscore in ColBERT qids

    # Count length and cut?
    max_conv_len = 0
    for qid, query_str in full_conv_queries_dict.items():
        max_conv_len = max(max_conv_len, len(tokenizer.encode(query_str)))
        if len(tokenizer.encode(query_str)) > args.max_len:
            print(f"Query id = {qid} has a total (conv) length {len(tokenizer.encode(query_str))} > {args.max_len}!")
            if args.debug:
                print('\nQuery before trimming:\t', query_str, '\n')
    print(f"Max conv length: {max_conv_len}")

    # Write out
    queries_series = pd.Series(full_conv_queries_dict)
    queries_series.to_csv(output_path, header=None, sep='\t')
    print(f"wrote to {output_path}")
    print("Finished")

if __name__ == "__main__":
    parser = ArgumentParser(description="preprocessing queries.")

    parser.add_argument('--max_len', dest='max_len', default=512, type=int)
    # parser.add_argument('--filepath_output', dest='filepath_output', required=True, type=str)
    parser.add_argument("--debug", default=False, required=False, action="store_true", help="print stuff")

    args = parser.parse_args()
    main(args)
