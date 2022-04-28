from ColBERT.paths import path_queries_21
import json
import ast
from collections import defaultdict
from transformers import BertTokenizerFast
import os
from argparse import ArgumentParser
import re

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever

def main(args):

    with open(path_queries_21['json']) as f:
        data = ast.literal_eval(f.read())

    queries = dict()
    tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')

    output_path = path_queries_21['queries_with_canonical']
    if args.max_len<512:
        output_path = path_queries_21[f'queries_with_canonical_{args.max_len}']

    if args.only_hist:
        output_path = path_queries_21[f'queries_history_only']

    elif args.only_canonical:
        output_path = path_queries_21[f'queries_canonical_last']

    # get conversations, canonical answers & lengths

    if os.path.exists(output_path):
        os.remove(output_path)
        print(f'removed previous file @ {output_path}')
    for entry in data:
        conv_id = entry['number']
        queries[conv_id] = dict()
        for turn in entry['turn']:
            turn_id = turn['number']

            queries[conv_id][turn_id] = {
                'raw_utterance': turn['raw_utterance'],
                'passage_answer': turn['passage'],
            }
            # Add previous passage answers
            if turn_id>1:
                queries[conv_id][turn_id]['previous_answer'] = queries[conv_id][turn_id-1]['passage_answer']
                queries[conv_id][turn_id]['previous_answer_len'] = len(tokenizer.encode(queries[conv_id][turn_id-1]['passage_answer']))

                # print(f"length of canonical (previous) answer passage @ qid {conv_id}-{turn_id} : {queries[conv_id][turn_id]['previous_answer_len']}")

    # construct queries for ColBERT
    for conv_id, q in queries.items():
        for turn_id, turn in q.items():
            qid = f"{conv_id}{turn_id}"

            # encode all
            previous_history = [queries[conv_id][previous_turn_id]['raw_utterance'] for previous_turn_id in range(1,turn_id)]

            previous_history_str = ' [SEP] '.join(previous_history)

            all_query_str = previous_history_str
            if turn_id>1:
                all_query_str += ' [SEP] ' + queries[conv_id][turn_id]['previous_answer']
            all_query_str += ' [SEP] ' + queries[conv_id][turn_id]['raw_utterance']
            # fix whitespaces
            all_query_str = all_query_str.strip()
            all_query_str = remove_prefix(all_query_str,'[SEP]')
            all_query_str = all_query_str.strip()

            if args.only_hist:
                all_query_str = previous_history_str
                if previous_history_str!='':
                    all_query_str += ' [SEP] '
                all_query_str += queries[conv_id][turn_id]['raw_utterance']

            elif args.only_canonical: # previous turn + [SEP] + canonical_response + [SEP] + current turn
                if turn_id>1:
                    all_query_str = queries[conv_id][1]['raw_utterance'] + ' [SEP] ' # turn 1
                    all_query_str += queries[conv_id][turn_id-1]['raw_utterance'] + ' [SEP] ' #turn n-1
                    all_query_str += queries[conv_id][turn_id]['previous_answer'] #canonical n-1
                    all_query_str += ' [SEP] '
                else:
                    all_query_str = ''
                all_query_str += queries[conv_id][turn_id]['raw_utterance']

            if len(tokenizer.encode(all_query_str))>args.max_len:
                print(f"Query id = {qid} has a total (conv) length {len(tokenizer.encode(all_query_str))} > {args.max_len}!")
                if args.debug:
                    print('\nQuery before trimming:\t',all_query_str,'\n')

                # Cut stuff
                #if canonical response too big:
                canonical_cut_ratio = 3/2 # if canonical response > 2/3 of arg length cut this
                canonical_response = queries[conv_id][turn_id]['previous_answer']
                if len(tokenizer.encode(canonical_response))> canonical_cut_ratio * args.max_len:
                    print(len(tokenizer.encode(canonical_response))) #debug
                    # cut canonical response
                    sentence_split_tokens = [1012, 1029, 999] # . ?
                    sentence_separators = set(sentence_split_tokens) & set(lst)
                    if len(sentence_separators) > 2: #common tokens TODO: occurences
                        # get first  sentence
                        first_sep_idx = min([lst.index(x) for x in sentence_separators])
                        new_canonical = tokenizer.decode(lst[1:first_sep_idx+1])

                        # get last sentence
                        if lst[-2] in sentence_separators: # remove last .,!,? if there is one
                            print(lst.pop(-2))

                            last_sep_idx = min([lst[::-1].index(x) for x in sentence_separators])
                            last_sep_token = lst[last_sep_idx]
                            last_sep_idx = -(len(lst)- lst[::-1].index(last_sep_token))
                            new_canonical += tokenizer.decode(lst[1+last_sep_idx:])

                            # replacement
                            all_query_str = all_query_str.replace(canonical_response,new_canonical)
                        else:
                            # or remove canonical response altogether
                            all_query_str = all_query_str.replace(canonical_response, ' ')

                # if cutting the canonical is not enough, start removing turns, from turn 2 onwards.
                while len(tokenizer.encode(all_query_str))>args.max_len: #
                    to_replace = re.search(r"\[SEP].*?\[SEP].?", all_query_str).group() # matches the 2nd+ turn
                    all_query_str = all_query_str.replace(to_replace, ' [SEP] ')

                if args.debug:
                    print('\nNew query:\t',all_query_str,'\n')


            with open(output_path,'a') as f:
                f.write(f"{qid} \t {all_query_str} \n")

    # # Cut query lengths @ 384
    # import pandas as pd
    # q = pd.read_csv(output_path, header=None, sep='\t', names = ['qid','q'])
    # q['len'] = q.q.apply(lambda x: len(tokenizer.encode(x)))
    # q[(q.len>384)]


if __name__ == "__main__":
    parser = ArgumentParser(description="preprocessing queries.")

    parser.add_argument('--max_len', dest='max_len', default=512, type=int)
    # parser.add_argument('--filepath_input', dest='filepath_input', required=True, type=str)
    parser.add_argument("--debug", default=False, required=False, action="store_true", help="print stuff")

    parser.add_argument("--only_hist", default=False, action="store_true", help="write only previous turns in Q")
    parser.add_argument("--only_canonical", default=False, action="store_true", help="write only canonical prev answer + last turn")

    args = parser.parse_args()
    main(args)
