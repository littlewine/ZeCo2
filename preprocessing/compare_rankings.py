import argparse
import pandas as pd
import os
import pytrec_eval
from collections import Counter

def read_pytreceval_run_dict(filepath_run):
    """Reads a trec eval file into a pytrec_eval compatible dictionary"""
    run = pd.read_csv(filepath_run,
                      names=['qid', 'Q0', 'docid', 'r', 'score'],
                      sep=' ', index_col=False)
    df = run.set_index(['qid','docid'])['score']
    run_dict = {level: df.xs(level).to_dict() for level in df.index.levels[0]}
    return run_dict

def read_pytreceval_qrel_dict(filepath_qrel):
    """Reads a trec eval file into a pytrec_eval compatible dictionary"""
    qrel = pd.read_csv(filepath_qrel,
                      names=['qid', 'Q0', 'docid', 'rel'],
                      sep=' ', index_col=False)
    df = qrel.set_index(['qid','docid'])['rel']
    qrel_dict = {level: df.xs(level).to_dict() for level in df.index.levels[0]}
    return qrel_dict

def compute_delta_performance(filepath_ranking1, filepath_ranking2, filepath_qrel, metric = 'map'):

    # Read to pytrec formats
    run1 = read_pytreceval_run_dict(filepath_ranking1)
    run2 = read_pytreceval_run_dict(filepath_ranking2)
    assert run1.keys() == run2.keys()

    qrel = read_pytreceval_qrel_dict(filepath_qrel)

    # Checks
    qids_retrieved = set(run1.keys())
    qids_eval = set(qrel.keys())
    print(f"# evaluation qids = {len(qids_eval)} \n"
          f"# ranking qids = {len(qids_retrieved)}")

    missing_qids = qids_eval - qids_retrieved
    assert len(missing_qids)==0, f"{len(missing_qids)} evaluation qids are missing from runfiles:\n" \
                                f"{missing_qids}"

    # pytrec eval
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {metric})

    results1 = evaluator.evaluate(run1)
    results1 = {k: v[metric] for k, v in results1.items()}
    r1 = Counter(results1)

    results2 = evaluator.evaluate(run2)
    results2 = {k: v[metric] for k, v in results2.items()}
    r2 = Counter(results2)

    r1.subtract(r2)
    delta_perf = dict(r1)

    return delta_perf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath_ranking1",
                        type=str,
                        required=True,
                        help="filepath to evaluate")

    parser.add_argument("--filepath_ranking2",
                        type=str,
                        required=True,
                        help="filepath to evaluate")

    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        choices=['cast19','cast20','cast21'],
                        help="qrel filepath")

    parser.add_argument("--metric",
                        type=str,
                        required=False,
                        default='map',
                        help="qrel filepath")

    args = parser.parse_args()
    path_project_data = '/ivi/ilps/personal/akrasak/data/0sConvDR'
    path_qrels = {
        'cast19': os.path.join(path_project_data, 'qrels', '2019qrels.txt'),
        'cast19MARCO': os.path.join(path_project_data, 'qrels', '2019qrels_MARCO.txt'),
        'cast20': os.path.join(path_project_data, 'qrels', '2020qrels.txt'),
        'cast21': os.path.join(path_project_data, 'qrels', 'qrels-docs.2021.txt')
    }

    compute_delta_performance(args.filepath_ranking1, args.filepath_ranking2, path_qrels[args.dataset], args.metric)
