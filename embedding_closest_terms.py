"""extract query analysis table showing
embedding change of "it" from CAsT'19 """
import pickle

from colbert.utils.parser import Arguments
from paths import *
from colbert.evaluation.loaders import load_colbert, load_queries
from colbert.modeling.tokenization.query_tokenization import QueryTokenizer
from colbert.modeling.inference import ModelInference
from python_pipeline import add_static_args
import pandas as pd
import torch
import numpy as np
from token_embedding_change import query_length
from preprocessing.compare_rankings import compute_delta_performance

def find_closest_embedding(term_embed, seq_embed, skip_identical=True, verbose=False):
    """Finds the position where the term embedding is closest
    to the token embedding of a sequence of embeddings"""
    term_embed = term_embed.unsqueeze(0).expand(seq_embed.shape[0], -1)
    cosine_sim = np.array((term_embed * seq_embed).sum(-1).to('cpu'))

    nr_identical_embeddings = (cosine_sim>.999).sum()
    if skip_identical and nr_identical_embeddings>0:
        if verbose:
            print(f"Skipping {nr_identical_embeddings} identical embedding during token matching")
        cosine_sim = np.where(cosine_sim>.999, 0, cosine_sim)

    return (np.argmax(cosine_sim), np.round(np.max(cosine_sim),3))

def print_token_in_context(sequence_ids,position, decode_fn, n_words=5):
    start = max(position-n_words, 0)
    s = f"{decode_fn(sequence_ids[start:position].tolist())}"
    s += " *"
    s += f"{decode_fn([sequence_ids[position].item()]).upper()}"
    s += f"* "
    s += f"{decode_fn(sequence_ids[position+1:position+n_words+1].tolist())}"
    return s

if __name__ == "__main__":
    parser = Arguments(description='End-to-end retrieval and ranking with ColBERT.')

    parser.add_argument('--dataset', dest='dataset',
                           choices=['cast19', 'cast20', 'cast21'], required=True)

    args = parser.parse()

    # add other needed args
    args.doc_maxlen = 180
    args.mask_punctuation = True
    args.nprobe = 32
    args.partitions = 32768
    args.bsize = 1
    args.checkpoint = path_colbert_checkpoint
    args.index_root = path_index_root
    args.batch = True
    args.query_maxlen = 257
    args.dim = 128
    args.similarity = 'cosine'


    args.colbert, args.checkpoint = load_colbert(args)

    queries_full = pd.Series(load_queries(path_queries[args.dataset]['full_conv']))
    queries_lastturn = pd.Series(load_queries(path_queries[args.dataset]['raw']))
    queries_human = pd.Series(load_queries(path_queries[args.dataset]['human']))
    queries = queries_full.to_frame(name='full').join(queries_lastturn.to_frame(name='lastturn')).join(queries_human.to_frame(name='human'))

    inference = ModelInference(colbert=args.colbert,
                               add_CLSQ_tokens=False, nr_expansion_tokens=0,
                               mask_method=None
                               )

    inference_lastturn = ModelInference(colbert=args.colbert,
                               add_CLSQ_tokens=False, nr_expansion_tokens=0,
                               mask_method='last_turn'
                               )

    Q_full = inference.queryFromText(list(queries.full),bsize=512)
    ids_full, _ = inference.query_tokenizer.tensorize(list(queries.full), nr_expansion_tokens=0)
    ids_lastturn, _ = inference.query_tokenizer.tensorize(list(queries.lastturn), nr_expansion_tokens=0)
    ids_human, _ = inference.query_tokenizer.tensorize(list(queries.human), nr_expansion_tokens=0)

    # Find added/removed terms (human - raw)
    # human_added_terms = [set(x)-set(y) for x,y in zip(ids_human.tolist(), ids_lastturn.tolist()) ]
    human_added_terms_from_hist = [(set(x)-set(y)).intersection(z)
                                   for x,y,z in zip(ids_human.tolist(), ids_lastturn.tolist(), ids_full.tolist() ) ]
    # human_removed_terms = [set(y)-set(x) for x,y in zip(ids_human.tolist(), ids_lastturn.tolist()) ]
    #
    # indices_it_simple_corref = np.array([x=={2009} for x in human_removed_terms]) * np.array([len(y)==1 for y in human_added_terms])


    Q_raw = inference.queryFromText(list(queries.lastturn), bsize=512)
    Q_ctx = inference_lastturn.queryFromText(list(queries.full), bsize=512)
    nonz_tok_raw = query_length(Q_raw)
    nonz_tok_ctx = query_length(Q_ctx)
    assert torch.all(torch.eq(
        nonz_tok_raw , nonz_tok_ctx))
    if not torch.all(torch.eq(
            nonz_tok_raw , nonz_tok_ctx)): # query mismatch
        mismatches = torch.nonzero(torch.logical_not(torch.eq(nonz_tok_raw, nonz_tok_ctx)), as_tuple=True)[0].tolist()

    token_id = 2009 #it
    token_positions = torch.nonzero(ids_lastturn==token_id,).tolist()
    closest_matches = dict()

    for qid, token_pos in token_positions:
        query_id = queries.index[qid]
        closest_matches[query_id] = dict()
        token_vector_raw = Q_raw[qid, token_pos]
        token_vector_ctx = Q_ctx[qid, token_pos]
        cosine_raw_ctx = (token_vector_raw * token_vector_ctx).sum().item()
        sep_pos = torch.nonzero(ids_full[qid] == 102, as_tuple=True)[0].tolist()

        # Identify previous history
        if len(sep_pos)<2: # skip first turn
            continue
        pos_last_turn_start = sep_pos[-2]
        Q_hist = Q_full[qid,:pos_last_turn_start]
        ids_history = ids_full[qid,:pos_last_turn_start]

        # print("History:\t", inference.bert_tokenizer.decode(ids_history.tolist()))
        # print("Last turn:\t", inference.bert_tokenizer.decode(ids_lastturn[qid].tolist()).replace("[PAD] ",''))
        closest_matches[query_id]['history'] = inference.bert_tokenizer.decode(ids_history.tolist())
        closest_matches[query_id]['last_turn'] = inference.bert_tokenizer.decode(ids_lastturn[qid].tolist()).replace("[PAD] ",'')
        closest_matches[query_id]['human'] = queries.iloc[qid]['human']

        closest_pos_raw, sim_raw = find_closest_embedding(token_vector_raw, Q_hist, skip_identical=False)
        closest_pos_ctx, sim_ctx = find_closest_embedding(token_vector_ctx, Q_hist, skip_identical=False ,verbose=True)

        closest_matches[query_id]['query'] = print_token_in_context(ids_lastturn[qid], token_pos,inference.bert_tokenizer.decode)
        closest_matches[query_id]['cosine_raw_ctx'] = np.round(cosine_raw_ctx,2)

        # print("query:\t ",print_token_in_context(ids_lastturn[qid], token_pos,inference.bert_tokenizer.decode),
        #       f"\t cosine_raw_ctx = {np.round(cosine_raw_ctx,2)}"
        #       )
        # print("human resolution:\t ",queries.iloc[qid]['human'])

        # print(f"Closest match to raw ({sim_raw:.2f}): "
        #       f"{print_token_in_context(ids_history, closest_pos_raw,inference.bert_tokenizer.decode)}")

        closest_matches[query_id]['raw_token'] = print_token_in_context(ids_history, closest_pos_raw,inference.bert_tokenizer.decode)
        closest_matches[query_id]['raw_sim'] = sim_raw

        # print(f"Closest match to ctx ({sim_ctx:.2f}): "
        #       f"{print_token_in_context(ids_history, closest_pos_ctx,inference.bert_tokenizer.decode)}")

        closest_matches[query_id]['ctx_token'] = print_token_in_context(ids_history, closest_pos_ctx,inference.bert_tokenizer.decode)
        closest_matches[query_id]['ctx_sim'] = sim_ctx

        # print("\n*****\n")

        # break

    closest_matches_table = pd.DataFrame(closest_matches).T
    # closest_matches_table.reset_index(inplace=True)
    # closest_matches_table.rename({'index':'qid_old'}, axis=1, inplace=True)
    closest_matches_table['qid_old'] = closest_matches_table.index.astype(str)
    if args.dataset=='cast19':
        closest_matches_table['qid'] = closest_matches_table.qid_old.apply(lambda x: str(x[:2])+'_'+str(x[2:]))
    elif args.dataset=='cast21':
        closest_matches_table['qid'] = closest_matches_table.qid_old.apply(lambda x: str(x[:3])+'_'+str(x[3:]))
    elif args.dataset=='cast20':
        with open(path_queries['cast20']['qid_mapping'], 'rb') as f:
            qid_mapping = pickle.load(f)
        closest_matches_table['qid'] = closest_matches_table.qid_old.apply(lambda x: qid_mapping[x])
    closest_matches_table.drop('qid_old',axis=1, inplace=True)
    closest_matches_table.set_index('qid', inplace=True)

    # Add delta performance per query
    # for metric in ['map','recall', 'recip_rank']
    Dperf = compute_delta_performance(path_rankings_noexp[args.dataset]['ctx'],
                              path_rankings_noexp[args.dataset]['raw'],
                              path_qrels[args.dataset],
                              metric = 'recall_1000')
    Dperf = pd.Series(Dperf,name='Drecall')

    print(Dperf.mean())

    closest_matches_table = closest_matches_table.join(Dperf)
    closest_matches_table.dropna(inplace=True) # drop qids without judgements

    closest_matches_table.to_csv("tables/closest_term_match_it.csv",sep='\t')