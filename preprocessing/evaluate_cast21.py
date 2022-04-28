import pandas as pd
import argparse
import subprocess
import os
import io
from pprint import pprint

qrel_path_MARCO = '/ivi/ilps/personal/akrasak/data/cqa-rewrite/qrels/2021qrels_MARCO.txt'
path_treceval = '/home/akrasak/anserini/tools/eval/trec_eval.9.0.4/trec_eval'
path_mapping_MARCO = '/ivi/ilps/personal/akrasak/data/collections/cast21sep/msmarco-docs.tsv.intmapping'

def main(args, prefix=False):

    evaluate_on_doclevel = True
    topk=1000
    underscore_pos_qid = 3
    # Fix marco doc names

    new_path_ranking = args.path_ranking + '.trecrun'
    if os.path.exists(new_path_ranking):
        print(f"Overwriting {new_path_ranking} (with corrected query ids)")
    ranking = pd.read_csv(args.path_ranking, sep='\t', names=['qid','docid','rank', 'score'])

    # read mapping
    mapping_d = pd.read_csv(path_mapping_MARCO,sep='\t',
                            squeeze=True, header=None, index_col=0)
    mapping_d.index = mapping_d.index.astype(int)
    mapping_d = mapping_d.to_dict()

    # Logging
    print("Nr of digits (of qids) in rankfiles")
    print(pd.Series(ranking.qid.unique()).astype(str).apply(lambda x: len(x)).value_counts())

    # construct new rankfile

    ranking['new_qid'] = ranking.qid.apply(lambda x: str(x)[:underscore_pos_qid] + '_' + str(x)[underscore_pos_qid:])

    ranking['new_docid'] = ranking.docid.apply(lambda x: mapping_d[int(x)])

    if evaluate_on_doclevel:
        # Delete passage indicator, deduplicate and compute order again
        ranking['new_docid'] = ranking.new_docid.apply(lambda x: x.split("-")[0])

        # deduplicate
        # ranking = ranking.sort_values('score', ascending=False).drop_duplicates(subset=['new_docid', 'new_qid'], keep='last')

        ranking.drop_duplicates(subset=['new_docid','new_qid'],
                                inplace=True)

        ranking['rank'] = ranking.groupby('new_qid')['score'].rank(ascending=False, method='first').astype(int)
        ranking = ranking.sort_values(['qid','rank'])

    # # cut retrieved docs to 1000 max
    # ranking = ranking[ranking['rank']<=topk]

    ranking['Q0'] = "Q0"
    ranking['run'] = "run"

    # write out ranking
    if ranking.score.isna().all():
        print("No logged scores found. Filling with dummy scores")
        ranking.score = -ranking['rank'].astype(float)

    ranking[['new_qid', "Q0", 'new_docid', 'rank', 'score', 'run']].to_csv(new_path_ranking,
                                                                           sep='\t', index=False, header=None)

    print(f"Evaluating runfile {new_path_ranking} with {ranking.qid.nunique()} unique query ids")

    # Run treceval

    command = [f"{path_treceval}",
               '-m', 'recall.1000',
               '-m', 'map',
               '-m', 'recip_rank',
               '-m', 'ndcg_cut.3',

               '-c',
               # '-q',
               qrel_path_MARCO,
               new_path_ranking,
               ]

    # partial evaluation
    qrel_df = pd.read_csv(qrel_path_MARCO, header=None, names=['qid', 'Q0', 'docid', 'rel'], sep=' ')
    if ranking.qid.nunique()!= qrel_df.qid.nunique():
        print("***** Some queries missing. Doing partial evaluation *****")
        # command.insert(-2,'-q')

    process = subprocess.run(command, capture_output=True, encoding='utf-8')
    if len(process.stderr) > 1:
        print(process.stderr)
        exit(1)
    else:
        print(process.stdout)
        results = pd.read_csv(io.StringIO(process.stdout), sep="\t", header=None, names=['metric', 'q', 'value'])
        results['metric'] = results.metric.str.strip() # remove stupid whitespace
        results_dict = results.set_index('metric')['value'].to_dict()
        if prefix:
            results_dict = {f"{prefix}_{k}": v for k, v in results_dict.items()}

    # print per turn information
    Q_results = results[results.q!='all']
    Q_results['turn'] = Q_results.q.apply(lambda x: x.split('_')[-1])
    Q_results.groupby(['turn', 'metric'])
    print("\n\t\t * Results per turn * ")
    print(Q_results[Q_results.metric.str.startswith('ndcg_cut_3')].groupby('turn').mean().sort_values('turn'))

    #exit
    pprint(results_dict)
    return results_dict, process.stdout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_ranking",
                        type=str,
                        required=True,
                        help="ranking path from ColBERT")

    args = parser.parse_args()
    main(args)
