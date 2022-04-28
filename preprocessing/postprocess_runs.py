import pandas as pd
import os
from argparse import ArgumentParser
import pickle

def convert_ids(filepath_run, filepath_docid_mapping, dataset, path_qid_mapping):
    """Convert a run to proper cast submission format
    docids are mapped through a mapping file and the docid_prefix is appended
    qids are turned from int to string (split in topic_len 2 or 3 with underscore)
    """
    run = pd.read_csv(filepath_run,sep='\t',
                      names=['qid_old', 'docid_old', 'r', 'score'],
                      dtype=str)
    run['score'] = run['score'].astype(float)
    mapping_d = pd.read_csv(
        filepath_docid_mapping,sep='\t', squeeze=True, header=None, index_col=0).to_dict()

    if dataset=='cast19':
        run['qid'] = run.qid_old.apply(lambda x: str(x[:2])+'_'+str(x[2:]))
    elif dataset=='cast21':
        run['qid'] = run.qid_old.apply(lambda x: str(x[:3])+'_'+str(x[3:]))
    elif dataset=='cast20':
        with open(path_qid_mapping, 'rb') as f:
            qid_mapping = pickle.load(f)
        run['qid'] = run.qid_old.apply(lambda x: qid_mapping[x])

    run['docid'] = run.docid_old.astype(int).apply(lambda x: mapping_d[x] if x in mapping_d.keys() else -1)

    # sanity checks
    missing_qids_in_mapfile = (run.docid==-1).sum()
    if missing_qids_in_mapfile>0:
        print(f"Warning: {missing_qids_in_mapfile} ids ({missing_qids_in_mapfile/len(run)*100} %) "
              f"of run ({filepath_run}) were not found "
              f"in mapping file ({filepath_docid_mapping})")

    return run[['qid','docid','score']]

def main(args):
    # Convert ids to from dictionary
    assert len(args.run) == len(args.mapping)

    # Merge runs from different indexes
    runs = []
    for i in range(len(args.run)):
        tmp = convert_ids(filepath_run=args.run[i],
                          filepath_docid_mapping=args.mapping[i],
                          dataset=args.dataset,
                          path_qid_mapping=args.path_qid_mapping if args.dataset=='cast20' else None
                          )
        runs.append(tmp)
    run = pd.concat(runs)

    # checks
    nr_unknown_ids = (run['docid'] == -1).sum()
    print(
        f"Warning: in total {nr_unknown_ids} document ids could not be mapped ({nr_unknown_ids / len(run) * 100}%). \n  Removing . . .")
    run = run[run['docid'] != -1]

    # turn passages to documents:
    if args.passage2doc:
        run['docid'] = run.docid.apply(lambda x: x.split("-")[0])
        run.drop_duplicates(subset=['qid','docid'], inplace=True)

    # groupby and rearrange ranks!
    run['rank'] = run.groupby('qid')['score'].rank(ascending=False, method='first').astype(int)
    run = run.sort_values(['qid','rank'])
    # cut retrieved docs to 1000 max
    run = run[run['rank']<=args.topk]

    # Stats
    run['collection_origin'] = run.docid.apply(lambda x:x[:4])
    print("\n Stats: \n retrieved docs from collections:")
    print(run.collection_origin.value_counts())
    run.drop('collection_origin', inplace=True, axis=1)

    # write to file
    run['Q0'] = 'Q0'
    run['run_id'] = args.run_id

    run = run[['qid','Q0', 'docid', 'rank', 'score', 'run_id']]
    run.to_csv(args.filepath_output, header=None, sep=' ', index=False)
    return args.filepath_output

if __name__ == "__main__":
    parser = ArgumentParser(description="postprocessing runs.")

    # parser.add_argument('--max_len', required=True, default=512, type=str)
    parser.add_argument('--run', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--mapping', nargs='+', help='<Required> Set flag', required=True)
    # Usage: python postprocess_runs.py -run run_MARCO.tsv run_CAR.tsv -mapping map_MARCO.tsv map_CAR.tsv

    parser.add_argument('--filepath_output', required=True, type=str)
    parser.add_argument('--path_qid_mapping', required=False, type=str)
    parser.add_argument('--run_id', required=True, default='Naxos', type=str)
    parser.add_argument('--topk', required=False, default=1000, type=int)
    parser.add_argument('--dataset', required=True, type=str,
                        choices=['cast19', 'cast20', 'cast21'],
                        help='For mapping int qids -> qids')
    parser.add_argument('--passage2doc', default=False, action="store_true",
                        required=False, help='Turn passages to documents (delete last -X from docids)')

    # parser.add_argument("--debug", default=False, required=False, action="store_true", help="print stuff")
    args = parser.parse_args()
    main(args)
