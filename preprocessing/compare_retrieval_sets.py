from argparse import ArgumentParser
import pandas as pd

def main(args):
    # Capture dataset
    if 'cast21' in args.filepath_unordered[0]:
        topic_len = 3
    elif 'cast19' in args.filepath_unordered[0]:
        topic_len = 2

    # Read the runs
    runs = dict()
    for i, path_run in enumerate(args.filepath_unordered):
        runs[i] = pd.read_csv(path_run, sep='\t', names=['qid','docid','rel']).groupby('qid')['docid'].apply(set).to_dict()

    # Measure nr of queries & retrieved docs
    for i, path_run in enumerate(args.filepath_unordered):

        print(f"Run {i}: \t {len(runs[i].keys())} qids \t {sum([len(runs[i][qid]) for qid in runs[i].keys()])} retrieved docs")

    # Subtract sets
    diff01 = [runs[0][qid] - runs[1][qid] for qid in runs[0].keys()]
    diff10 = [runs[1][qid] - runs[0][qid] for qid in runs[0].keys()]
    common01 = [runs[0][qid].intersection(runs[1][qid]) for qid in runs[0].keys()]

    set_diff = pd.DataFrame(index=runs[0].keys())
    set_diff["0-1"] = diff01
    set_diff["1-0"] = diff10
    set_diff['nr_non_common'] = set_diff.applymap(len).sum(1)

    set_diff['nr_common'] = [len(x) for x in common01]
    set_diff['pct_common'] = set_diff.nr_common / (set_diff.nr_common + set_diff.nr_non_common)
    set_diff.sort_values("pct_common", inplace=True)
    set_diff.drop(['0-1','1-0'], axis=1, inplace=True)

    print(f"Found {set_diff['nr_non_common'].sum()} documents that were not present in the "
          f"other runs retrieval set")

    if set_diff['nr_non_common'].sum()>0:
        print("20 most different queries and sets:")
        print(set_diff[['nr_non_common','nr_common']].head(20))

    # Macro averaged
    print(f"Macro averaged overlap of document (per query):   {set_diff.pct_common.mean()*100:.2f} % ")
    # # Micro averaged percentage of common queries:
    # print(f"Micro averaged percentage of common docs in queries:    "
    #       f"{set_diff['nr_non_common'].sum()/set_diff['nr_common'].sum()*100:.2f}")

    # Per turn analysis
    per_turn = set_diff.pct_common.reset_index()
    per_turn['qid'] = per_turn['index'].astype(str)
    per_turn['qid'] = per_turn['qid'].apply(lambda x: x[:topic_len]+"_"+x[topic_len:])
    per_turn.drop('index', axis=1, inplace=True)
    per_turn['turn'] = per_turn['qid'].apply(lambda  x: x.split("_")[-1]).astype(int)
    print("Percentage of common documents per turn:")
    print(per_turn.groupby('turn')['pct_common'].mean())

    print(runs[0].head())


    return



if __name__ == "__main__":
    parser = ArgumentParser(description="Compare two unordered sets of documents")

    parser.add_argument('--filepath_unordered', nargs='+', help='<Required> Set flag', required=True)

    args = parser.parse_args()
    main(args)