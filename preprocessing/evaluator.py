from paths import *
import argparse
import subprocess
import pandas as pd
import io

def main(filepath_ranking, filepath_qrel, prefix=False):
    # Evaluation (ranking)
    command = [f"{path_treceval}",
               '-m', 'recall.1000',
               '-m', 'recall.100',
               '-m', 'map',
               '-m', 'recip_rank',
               '-m', 'ndcg_cut.3',

               '-c',
               # '-q',
               filepath_qrel,
               filepath_ranking,
               ]

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
        return results_dict, process.stdout

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath_ranking",
                        type=str,
                        required=True,
                        help="filepath to evaluate")

    parser.add_argument("--filepath_qrel",
                        type=str,
                        required=True,
                        help="qrel filepath")
    args = parser.parse_args()
    main(args.filepath_ranking, args.filepath_qrel)
