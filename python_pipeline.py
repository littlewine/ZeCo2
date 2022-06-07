import os, psutil, sys
import random
from colbert.utils.parser import Arguments
from paths import *

from colbert.ranking.retrieval import retrieve
from colbert.ranking.batch_retrieval import batch_retrieve
from colbert.utils.runs import Run
from colbert.evaluation.loaders import load_colbert, load_qrels, load_queries, load_topK_pids
from colbert.indexing.faiss import get_faiss_index_name

from colbert.ranking.reranking import rerank
from colbert.ranking.batch_reranking import batch_rerank

from argparse import Namespace
from preprocessing.postprocess_runs import main as postprocess_runs
from preprocessing.evaluator import main as evaluator

from copy import deepcopy
from pprint import pprint

config = {"cast19": {'collections': ['MSMARCO.L2.32x200k.180len',
                                     'CAR.FirstP.L2.32x200k.180len'],
                     'collection_mappings': [path_collection_mappings['marcoP'],
                                             path_collection_mappings['CAR']],
                     'qrel': path_qrels['cast19'],
                     },
                    "cast20": {'collections': ['MSMARCO.L2.32x200k.180len',
                                     'CAR.FirstP.L2.32x200k.180len'],
                     'collection_mappings': [path_collection_mappings['marcoP'],
                                             path_collection_mappings['CAR']],
                     'qrel': path_qrels['cast20'],
                     },
          "cast21": {'collections': ["kilt.FirstP.L2.32x200k.180len",
                                     "wapo.FirstP.L2.32x200k.180len",
                                     "marco.FirstP.L2.32x200k.180len"],
                     'collection_mappings': [path_collection_mappings['KILT'],
                                             path_collection_mappings['WAPO'],
                                             path_collection_mappings['marcoD']],
                     'qrel': path_qrels['cast21'],
                     }
          }

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def retrieve_step(args):
    # Run retrieval
    random.seed(12345)
    args.depth = args.depth if args.depth > 0 else None

    if args.part_range:
        part_offset, part_endpos = map(int, args.part_range.split('..'))
        args.part_range = range(part_offset, part_endpos)

    with Run.context():
        args.colbert, args.checkpoint = load_colbert(args)
        args.qrels = load_qrels(args.qrels)
        args.queries = load_queries(args.queries)

        args.index_path = os.path.join(args.index_root, args.index_name)

        if args.faiss_name is not None:
            args.faiss_index_path = os.path.join(args.index_path, args.faiss_name)
        else:
            args.faiss_index_path = os.path.join(args.index_path, get_faiss_index_name(args))

        if args.batch:
            return batch_retrieve(args)
        else:
            return retrieve(args)

def rerank_step(args):
    # Run reranking

    if args.part_range:
        part_offset, part_endpos = map(int, args.part_range.split('..'))
        args.part_range = range(part_offset, part_endpos)

    with Run.context():
        if not isinstance(args.checkpoint,str):
            args.checkpoint = path_colbert_checkpoint
        args.colbert, args.checkpoint = load_colbert(args)

        args.queries = load_queries(args.queries)
        args.qrels = load_qrels(args.qrels)
        args.topK_pids, args.qrels = load_topK_pids(args.topK, qrels=args.qrels)

        args.index_path = os.path.join(args.index_root, args.index_name)

        if args.batch:
            return batch_rerank(args)
        else:
            return rerank(args)

def add_static_args(args):
    # Add arguments that are static in our experiments
    args.amp = True
    args.doc_maxlen = 180
    args.mask_punctuation = True
    args.nprobe = 32
    args.partitions = 32768
    args.bsize = 1
    args.checkpoint = path_colbert_checkpoint
    args.index_root = path_index_root
    # args.root = os.getcwd() # get the default (experiments/)
    args.batch = True

    if args.setting=='raw':
        args.query_maxlen=32
        args.queries = path_queries[args.dataset]['raw']
        args.mask_method = None
    elif args.setting=='ZeCo2':
        args.query_maxlen = 256
        args.queries = path_queries[args.dataset]['full_conv']
        args.mask_method = 'ZeCo2'
    elif args.setting=='allHistory':
        args.query_maxlen = 256
        args.queries = path_queries[args.dataset]['full_conv']
        args.mask_method = None
    elif args.setting=='human':
        args.query_maxlen = 32
        args.queries = path_queries[args.dataset]['human']
        args.mask_method = None

    # To add from parsing
    #     query_maxlen
    #     queries
    #     mask_method
    #     experiment
    #     index_name
    #     run?
    return args

def get_retriever_args(global_args):
    retriever_args = deepcopy(global_args)
    retriever_args.faiss_depth = 1024
    retriever_args.retrieve_only = True
    # add here "initial" args?

    return retriever_args

def get_reranker_args(global_args, path_topK):
    reranker_args = deepcopy(global_args)

    reranker_args.log_scores = True
    reranker_args.topK = path_topK

    return reranker_args


if __name__ == "__main__":
    parser_global = Arguments(description='End-to-end retrieval and ranking with ColBERT.')

    # # Retriever args
    # parser_global.add_model_parameters()
    # parser_global.add_model_inference_parameters()
    # # parser_global.add_ranking_input() # included in add reranking input!
    # parser_global.add_retrieval_input()

    # get retriever defaults
    parser_global.add_argument('--faiss_name', dest='faiss_name', default=None, type=str)
    parser_global.add_argument('--part-range', dest='part_range', default=None, type=str)
    parser_global.add_argument('--batch', dest='batch', default=False, action='store_true')
    parser_global.add_argument('--depth', dest='depth', default=1000, type=int)
    parser_global.add_argument('--similarity', dest='similarity', default='cosine', choices=['cosine', 'l2'])
    parser_global.add_argument('--dim', dest='dim', default=128, type=int)
    parser_global.add_argument('--collection', dest='collection', default=None)
    parser_global.add_argument('--qrels', dest='qrels', default=None)

    # get reranker defaults
    parser_global.add_argument('--shortcircuit', dest='shortcircuit', default=False, action='store_true')

    # # Reranker args
    parser_global.add_argument('--step', dest='step', default=1, type=int)
    # parser_global.add_argument('--log-scores', dest='log_scores', default=False, action='store_true')
    #
    # parser_global.add_reranking_input()

    # My args
    parser_global.add_argument('--debug', default=False, required=False, action="store_true", help='debugging flag' )
    # parser_global.add_argument('--mask_method', default=None, required=False,
    #                     choices = [None,'ZeCo2'],
    #                     help='Do matching only on specific tokens')

    # parser_global.add_argument('--query_maxlen', dest='query_maxlen', default=32, type=int)
    # parser_global.add_argument('--queries', dest='queries', default=None)
    parser_global.add_argument('--overwrite_rundir', dest='overwrite_rundir', default=False,
                               action="store_true", help='do not ask for confirmation to overwrite run directory')
    # parser_global.add_argument('--experiment', dest='experiment', default='dirty')
    # parser_global.add_argument('--index_name', dest='index_name', required=True)
    # parser_global.add_argument('--run', dest='run', default=Run.name)
    parser_global.add_argument('--dataset', dest='dataset',
                               choices=['cast19', 'cast20','cast21'], required=True)
    parser_global.add_argument('--setting', dest='setting',
                               choices=['raw', 'ZeCo2', 'allHistory','human'], required=True)
    parser_global.add_argument('--nr_expansion_tokens', dest='nr_expansion_tokens',
                               default=10, type=int)
    parser_global.add_argument('--add_CLSQ_tokens', dest='add_CLSQ_tokens', default=False,
                               action="store_true", help='Include CLS and Q token in score matching function')

    # Parse experiment arguments
    global_args = parser_global.parse()
    global_args = add_static_args(global_args) # extend them with static values
    print('args_global: ',global_args)

    # For each collection in Year:
    retrieval_lists = []
    reranking_lists = []

    if not global_args.debug:
        collections = config[global_args.dataset]['collections']
        mappings = config[global_args.dataset]['collection_mappings']
    else:
        collections = ['MSMARCO.L2.32x200k.180len.small', 'wapo.FirstP.L2.32x200k.180len']
        mappings = [path_collection_mappings['marcoP'], path_collection_mappings['WAPO']]

    for collection in collections:
        # change collection
        global_args.index_name = collection
        print(f"Running on {collection}")

        # Retrieve step
        args_retriever = get_retriever_args(global_args)
        print('args_retriever: ',args_retriever)

        print(f"\n\nBefore retrieval step. Using {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} mb RAM.\n\n ")

        path_topK = retrieve_step(args_retriever)
        retrieval_lists.append(path_topK)
        print("Finished 1st stage ranking, results @ ",path_topK)
        print(f"\n\nAfter retrieval step. Using {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} mb RAM.\n\n ")
        del args_retriever

        # Rerank step
        args_reranker = get_reranker_args(global_args, path_topK)

        print(f"\n\nBefore reranking step. Using {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} mb RAM.\n\n ")

        path_reranked = rerank_step(args_reranker)

        print(f"\n\nAfter reranking step. Using {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} mb RAM.\n\n ")

        reranking_lists.append(path_reranked)
        print(f"Finished ranking for {global_args.index_name}")
        del args_reranker

    assert len(retrieval_lists)==len(reranking_lists)
    print("\n\n********\nFinished retrieval/reranking\n********\n\n")

    # Run postprocessing

    args_postprocessor = Namespace()
    args_postprocessor.run = reranking_lists
    args_postprocessor.mapping = mappings
    args_postprocessor.dataset = global_args.dataset
    args_postprocessor.run_id = 'postprocessed_run'
    args_postprocessor.filepath_output = os.path.join(
        os.path.dirname(path_topK), f'{args_postprocessor.run_id}.trecrun'
    )
    args_postprocessor.topk=1000

    if global_args.dataset=='cast21':
        args_postprocessor.passage2doc=True
    else:
        args_postprocessor.passage2doc=False

    if global_args.dataset=='cast20':
        args_postprocessor.path_qid_mapping = path_queries['cast20']['qid_mapping']

    path_processed_run = postprocess_runs(args_postprocessor)

    # Run evaluation

    results_dict = evaluator(filepath_ranking = path_processed_run,
              filepath_qrel = config[global_args.dataset]['qrel'])

    pprint(results_dict)
