from ColBERT.preprocessing.compare_rankings import compute_delta_performance
from colbert.utils.parser import Arguments
from paths import *
from colbert.evaluation.loaders import load_colbert, load_queries
from colbert.modeling.tokenization.query_tokenization import QueryTokenizer
from colbert.modeling.inference import ModelInference
from python_pipeline import add_static_args
import pandas as pd
import torch
import numpy as np
from utils import qid_to_str
from scipy import stats

def print_token_change(cosine_sim, qid, input_ids, tokenizer):
    tokens = tokenizer.batch_decode([[x] for x in input_ids[qid]])
    change = [np.round(1-x,2) for x in cosine_sim[qid].tolist()]
    s = [f"{token} ({change[i]}) \t" for i,token in enumerate(tokens) if token!='[PAD]']
    print(''.join(s))

def query_length(Q_vector):
    return (Q_vector[:, :, 0] != 0).sum(1)

def print_query_change(Q_similarity, input_ids_q, tokenizer):
    Q_similarity, input_ids_q = Q_similarity.tolist(), input_ids_q.tolist()
    for i, token_id in enumerate(input_ids_q):
        if token_id==0:
            break
        print(Q_similarity[i], tokenizer.decode([token_id]))

def get_avg_token_changes(args, min_occurences, return_cosine_matrix=False):
    """Get a matrix of token changes """
    queries_full = pd.Series(load_queries(path_queries[args.dataset]['full_conv']))
    queries_lastturn = pd.Series(load_queries(path_queries[args.dataset]['raw']))
    queries = queries_full.to_frame(name='full').join(queries_lastturn.to_frame(name='lastturn'))

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

    Q_raw = inference.queryFromText(list(queries.lastturn), bsize=512)
    Q_ctx = inference_lastturn.queryFromText(list(queries.full), bsize=512)
    nonz_tok_raw = query_length(Q_raw)
    nonz_tok_ctx = query_length(Q_ctx)
    assert torch.all(torch.eq(
        nonz_tok_raw , nonz_tok_ctx))
    if not torch.all(torch.eq(
            nonz_tok_raw , nonz_tok_ctx)): # query mismatch
        mismatches = torch.nonzero(torch.logical_not(torch.eq(nonz_tok_raw, nonz_tok_ctx)), as_tuple=True)[0].tolist()

    cosine_sim = Q_raw * Q_ctx
    nonz_tok_cosine = query_length(cosine_sim)
    # Aggregate per query
    cosine_sim = cosine_sim.sum(-1)
    if return_cosine_matrix:
        return cosine_sim
    assert torch.all(torch.eq(nonz_tok_raw,
                              nonz_tok_cosine))

    # print(f"Cosine distance length: {query_length(cosine_sim)}")
    cosine_sim = torch.where(cosine_sim > 0, cosine_sim, torch.ones_like(cosine_sim))

    # Get token ids
    ids_lastturn, _ = inference.query_tokenizer.tensorize(list(queries.lastturn), nr_expansion_tokens=0)
    if not inference.add_CLSQ_tokens:
        ids_lastturn = ids_lastturn[:,2:]
    non_empty_mask = ids_lastturn.abs().sum(dim=0).bool()
    ids_lastturn = ids_lastturn[:, non_empty_mask]

    # Measure how much each token changed in dataset
    frequent_tokens = pd.DataFrame(ids_lastturn.numpy()).stack().value_counts()
    frequent_tokens = frequent_tokens.to_frame(name='number')
    frequent_tokens.reset_index(inplace=True)
    frequent_tokens.rename({'index':'token_id'}, axis=1, inplace=True)

    frequent_tokens = frequent_tokens[frequent_tokens.number>=min_occurences]
    frequent_tokens['token'] = frequent_tokens.token_id.apply(lambda x: inference.bert_tokenizer.decode([x]))

    # Find token positions
    avg_token_sim = dict()
    for token_id in frequent_tokens.token_id:
        token_mask = (ids_lastturn == token_id)
        token_sim = cosine_sim*token_mask.to('cuda')
        avg_token_sim[token_id] = (token_sim.sum()/token_mask.sum()).item()

    frequent_tokens['mean_change'] = frequent_tokens.token_id.apply(lambda x:
                                                                      1-avg_token_sim[x])
    # print(frequent_tokens.sort_values("mean_change", ascending=False))

    return frequent_tokens

if __name__ == "__main__":
    parser = Arguments(description='End-to-end retrieval and ranking with ColBERT.')

    parser.add_argument('--dataset', dest='dataset',
                        choices=['cast19', 'cast20', 'cast21', 'all'], required=True)

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

    if args.dataset!='all':
        # get average token changes per dataset
        get_avg_token_changes(args, min_occurences=1)

        # get embedding change metric per query
        print("**** \n Compute correlation between max embedding change & Drecall \n****")
        cosine_sim = get_avg_token_changes(args, min_occurences=1, return_cosine_matrix=True)
        cosine_sim = torch.where(cosine_sim > 0, cosine_sim, torch.ones_like(cosine_sim))
        max_change_query,_ = cosine_sim.min(axis=1)
        max_change_query = 1-max_change_query
        max_change_query = pd.Series(max_change_query.tolist(),
                                     index=qid_to_str(queries_full.index,args.dataset),
                                     name='max_embed_change').round(3)
        # get Delta performance
        Dperf = compute_delta_performance(path_rankings_noexp[args.dataset]['ctx'],
                                          path_rankings_noexp[args.dataset]['raw'],
                                          path_qrels[args.dataset],
                                          metric = 'recall_1000')
        Dperf = pd.Series(Dperf,name='Drecall')

        max_change_query = max_change_query.to_frame().join(Dperf).dropna()

        # compute Pearson corr coef
        drop_first_turns=False
        if drop_first_turns:
            max_change_query = max_change_query[~max_change_query.index.str.endswith("_1")]
        cor_coef, p_val = stats.pearsonr(max_change_query['max_embed_change'],max_change_query['Drecall'])
        print(f"Correlation coefficient: {cor_coef:.3f}\n"
              f"p-value: {p_val}")

    else:
        frequent_tokens_all_years = []
        for d in ['cast19','cast20','cast21']:
            args.dataset = d
            frequent_tokens_all_years.append(get_avg_token_changes(args, min_occurences=1).set_index("token_id"))
        pd.concat(frequent_tokens_all_years, axis=1)

        print("s")

        # Combine metrics
        frequent_tokens = pd.concat(frequent_tokens_all_years, axis=1)
        frequent_tokens.fillna(0, inplace=True)
        frequent_tokens['total_count'] = frequent_tokens.number.sum(1)
        frequent_tokens['total_change'] = (np.array(frequent_tokens.mean_change)*np.array(frequent_tokens.number)).sum(1)
        frequent_tokens.drop(['mean_change', 'number'], axis=1, inplace=True)
        frequent_tokens.drop(0, inplace=True)


        macro_avg = frequent_tokens.avg_change.mean()
        print(f"Macro averaged token change: {macro_avg}")

        micro_avg = frequent_tokens.total_change.sum() / frequent_tokens.total_count.sum()
        print(f"Micro averaged token change: {micro_avg}")


        table = frequent_tokens[frequent_tokens.total_count > 50].sort_values('avg_change', ascending=False)
        table['word'] = table['token'].iloc[:,0]
        table.drop(['token','total_change'], axis=1, inplace=True)
        table.total_count = table.total_count.astype(int)

        # print(table[['word', 'total_count', 'avg_change']].set_index('word').round(3).to_latex())
