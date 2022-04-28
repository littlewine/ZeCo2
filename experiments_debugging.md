## Start new server

!!! Be careful where to start. in old or new directory?
```
source ~/.bash_profile
conda activate colbert
cd /home/akrasak/colbertdebug/ColBERT
export PYTHONPATH=$PYTHONPATH:/home/akrasak/colbertdebug/ColBERT

```

*****   Retrieval.   *****
```
old_retrieve_rank='/home/akrasak/cqa-rewrite/ColBERT/cast19-raw/retrieve.py/2021-06-11_16.30.27/unordered.tsv'
old_retrieve_args='/home/akrasak/cqa-rewrite/ColBERT/cast19-raw/retrieve.py/2021-06-11_16.30.27/logs/args.json'

new_retrieve_rank='/home/akrasak/cqa-rewrite/ColBERT/cast19-raw/retrieve.py/2021-10-07_12.20.16/unordered.tsv'
new_retrieve_args='/home/akrasak/cqa-rewrite/ColBERT/cast19-raw/retrieve.py/2021-10-07_12.20.16/logs/args.json'
```



exact args given:

```
/home/akrasak/cqa-rewrite/ColBERT/colbert/retrieve.py 


source ~/.bash_profile
conda activate colbert
cd ~/cqa-rewrite/ColBERT/
export PYTHONPATH=$PYTHONPATH:$HOME/cqa-rewrite/ColBERT

CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=6 \
python -m colbert.retrieve --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --nprobe 32 --partitions 32768 --faiss_depth 1024 --index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ --index_name MSMARCO.L2.32x200k.180len --checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn --root /home/akrasak/cqa-rewrite/ColBERT/ --experiment cast19-raw --queries /home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/1_Original.tsv.raw.queries --batch --retrieve_only
```

*****   Reranking.   *****
```
old_retrieve_rank='/home/akrasak/cqa-rewrite/ColBERT/cast19-raw/rerank.py/2021-06-11_16.35.05/ranking.tsv'
old_retrieve_args='/home/akrasak/cqa-rewrite/ColBERT/cast19-raw/rerank.py/2021-06-11_16.35.05/logs/args.json'
```

```
CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=6 \
python -m colbert.rerank --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --partitions 32768 \
--index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ --index_name MSMARCO.L2.32x200k.180len \
--checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn \
--root /home/akrasak/cqa-rewrite/ColBERT/ --experiment cast19-raw \
--queries /home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/1_Original.tsv.raw.queries \
--batch --log-scores --topk /home/akrasak/cqa-rewrite/ColBERT/cast19-raw/retrieve.py/2021-06-11_16.30.27/unordered.tsv

rerank with old retrieval set	:	/home/akrasak/cqa-rewrite/ColBERT/cast19-raw/rerank.py/2021-10-08_13.40.29/ranking.tsv
{'map': 0.0324,
 'ndcg_cut_3': 0.0472,
 'recall_1000': 0.2282,
 'recip_rank': 0.1168}
```
```
CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=6 \
python -m colbert.rerank --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --partitions 32768 --index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ --index_name MSMARCO.L2.32x200k.180len --checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn --root /home/akrasak/cqa-rewrite/ColBERT/ --experiment cast19-raw --queries /home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/1_Original.tsv.raw.queries --batch --log-scores --topk /home/akrasak/cqa-rewrite/ColBERT/cast19-raw/retrieve.py/2021-10-07_12.20.16/unordered.tsv

rerank with new retrieval set	:	/home/akrasak/cqa-rewrite/ColBERT/cast19-raw/rerank.py/2021-10-08_14.20.17/ranking.tsv

{'map': 0.0205,
 'ndcg_cut_3': 0.0324,
 'recall_1000': 0.0979,
 'recip_rank': 0.072}
```


***
## Changing nr of tokens to 0:
```
python -m colbert.rerank --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --partitions 32768 --index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ --index_name MSMARCO.L2.32x200k.180len --checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn --root /home/akrasak/cqa-rewrite/ColBERT/ --experiment cast19-raw --queries /home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/1_Original.tsv.raw.queries --batch --log-scores --topk /home/akrasak/cqa-rewrite/ColBERT/cast19-raw/retrieve.py/2021-06-11_16.30.27/unordered.tsv --nr_expansion_tokens 0


/home/akrasak/cqa-rewrite/ColBERT/cast19-raw/rerank.py/2021-10-11_15.21.46/ranking.tsv
{'map': 0.0324,
 'ndcg_cut_3': 0.0472,
 'recall_1000': 0.2282,
 'recip_rank': 0.1168}
```

No effect, since code is not affected when not `last_turn`

## Reverting to previous commits

### Start new server
```
source ~/.bash_profile
conda activate colbert
cd 
export PYTHONPATH=$PYTHONPATH:/home/akrasak/colbertdebug/ColBERT

```

### Run reranking step with old (June) unordered set on (CAsT19)
```
path_topk='/home/akrasak/cqa-rewrite/ColBERT/cast19-raw/retrieve.py/2021-06-11_16.30.27/unordered.tsv'

CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=6 \
python -m colbert.rerank --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --partitions 32768 \
--index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ --index_name MSMARCO.L2.32x200k.180len \
--checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn \
--root /home/akrasak/colbertdebug/ColBERT --experiment cast19-raw \
--queries /home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/1_Original.tsv.raw.queries \
--batch --log-scores --topk $path_topk
```

### Evaluate on CAsT19

```
python evaluate_cast.py --path_ranking 
```

* Commit `383ab2cac1831f08d448c5f8fc3100c41795ff4f` (last_summer_commit)

Results: `/home/akrasak/colbertdebug/ColBERT/cast19-raw/rerank.py/last_summer_commit/ranking.tsv`
```
{'map': 0.0324,
 'ndcg_cut_3': 0.0472,
 'recall_1000': 0.2282,
 'recip_rank': 0.1168}
```

!! Queries seem to be missing in some runs

Results (with masked last turn): `/home/akrasak/colbertdebug/ColBERT/cast19-raw/rerank.py/last_summer_commit/ranking.tsv`
```
{'map': 0.0324,
 'ndcg_cut_3': 0.0472,
 'recall_1000': 0.2282,
 'recip_rank': 0.1168}
```

Same commit but with human query:

```
CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=6 \
python -m colbert.rerank --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --partitions 32768 \
--index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ --index_name MSMARCO.L2.32x200k.180len \
--checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn \
--root /home/akrasak/colbertdebug/ColBERT --experiment cast19-human \
--queries /home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/10_Human.tsv.queries \
--batch --log-scores --topk $path_topk

```


Results: `/home/akrasak/colbertdebug/ColBERT/cast19-human/rerank.py/last_summer_commit/ranking.tsv`
```
{'map': 0.0388,
 'ndcg_cut_3': 0.0663,
 'recall_1000': 0.2538,
 'recip_rank': 0.1472}
```

* git checkout -b zero_commit efaabb0f8731c7d96a9fe109a125357a9232f7a7

```
CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=6 \
python -m colbert.rerank --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --partitions 32768 \
--index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ --index_name MSMARCO.L2.32x200k.180len \
--checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn \
--root /home/akrasak/colbertdebug/ColBERT --experiment cast19-raw \
--queries /home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/1_Original.tsv.raw.queries \
--batch --log-scores --topk $path_topk
```

Results: `/home/akrasak/colbertdebug/ColBERT/cast19-human/rerank.py/zero_commit/ranking.tsv`


```
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py \
--path_ranking /home/akrasak/colbertdebug/ColBERT/cast19-raw/rerank.py/zero_commit/ranking.tsv

{'map': 0.1826,
 'ndcg_cut_3': 0.2583,
 'recall_1000': 0.4515,
 'recip_rank': 0.4285}
```


* git checkout -b last_summer_commit 383ab2cac1831f08d448c5f8fc3100c41795ff4f

Trying to replicate `/home/akrasak/cqa-rewrite/ColBERT/cast19-zeroshot-lastturn/rerank.py/2021-07-19_19.02.43/ranking.tsv`

```
CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=6 \
python -m colbert.rerank --amp --doc_maxlen 180 \
--query_maxlen 256 --mask-punctuation --bsize 64 --partitions 32768 \
--index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ \
--index_name MSMARCO.L2.32x200k.180len \
--checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn \
--root /home/akrasak/colbertdebug/ColBERT --experiment cast19-zeroshot-lastturn \
--queries /home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/1_Original.tsv.full.raw.queries \
--log-scores --topk /home/akrasak/cqa-rewrite/ColBERT/cast19-raw/retrieve.py/2021-06-11_16.30.27/unordered.tsv \
--mask_method last_turn --run last_summer_commit_topK_raw
```

!! this ^^^ is missing `--batch` and is retrieving from cast19-raw topK!

```
add --batch and --run last_summer_commit_topK_raw_batch


python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py --path_ranking /home/akrasak/colbertdebug/ColBERT/cast19-zeroshot-lastturn/rerank.py/last_summer_commit_topK_raw_batch/ranking.tsv

```

```
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py --path_ranking /home/akrasak/colbertdebug/ColBERT/cast19-zeroshot-lastturn/rerank.py/last_summer_commit_topK_raw/ranking.tsv


No logged scores found. Filling with dummy scores
{'map': 0.1672,
 'ndcg_cut_3': 0.252,
 'recall_1000': 0.4743,
 'recip_rank': 0.4787}
```

* git checkout -b 19Jul 2839bb172f8e74f89f7990b1e1c40c2a80d4923d

```
CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=6 \
python -m colbert.rerank --amp --doc_maxlen 180 \
--query_maxlen 256 --mask-punctuation --bsize 64 --partitions 32768 \
--index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ \
--index_name MSMARCO.L2.32x200k.180len \
--checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn \
--root /home/akrasak/colbertdebug/ColBERT --experiment cast19-zeroshot-lastturn \
--queries /home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/1_Original.tsv.full.raw.queries \
--log-scores --topk /home/akrasak/cqa-rewrite/ColBERT/cast19-raw/retrieve.py/2021-06-11_16.30.27/unordered.tsv \
--mask_method last_turn --run 19Jul_topK_raw

```

```
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py --path_ranking /home/akrasak/colbertdebug/ColBERT/cast19-zeroshot-lastturn/rerank.py/19Jul_topK_raw/ranking.tsv

No logged scores found, logging with dummy scores

{'map': 0.1757,
 'ndcg_cut_3': 0.2539,
 'recall_1000': 0.4802,
 'recip_rank': 0.4546}
```

```
Same, but with --batch:

CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=6 \
python -m colbert.rerank --amp --doc_maxlen 180 \
--query_maxlen 256 --mask-punctuation --bsize 64 --partitions 32768 \
--index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ \
--index_name MSMARCO.L2.32x200k.180len \
--checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn \
--root /home/akrasak/colbertdebug/ColBERT --experiment cast19-zeroshot-lastturn \
--queries /home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/1_Original.tsv.full.raw.queries \
--topk /home/akrasak/cqa-rewrite/ColBERT/cast19-raw/retrieve.py/2021-06-11_16.30.27/unordered.tsv \
--mask_method last_turn --run 19Jul_topK_raw --batch --log-scores 

```

```
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py --path_ranking /home/akrasak/colbertdebug/ColBERT/cast19-zeroshot-lastturn/rerank.py/19Jul_topK_raw_batch/ranking.tsv

***** Some queries missing. Doing partial evaluation *****

{'map': 0.1722,
 'ndcg_cut_3': 0.249,
 'recall_1000': 0.4787,
 'recip_rank': 0.4634}

```

## Try to reproduce 22July run (last logged cast19 result in excel sheet) with different logs
args:
`/home/akrasak/cqa-rewrite/ColBERT/cast19-zeroshot-lastturn/rerank.py/2021-07-22_20.20.26/ranking.tsv`

```

CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=6 \
python -m colbert.rerank \
--amp --doc_maxlen 180 --mask-punctuation --bsize 1 \
--partitions 32768 --index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ \
--index_name MSMARCO.L2.32x200k.180len \
--checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn \
--root /home/akrasak/colbertdebug/ColBERT --experiment last_column_results \
--queries /home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/1_Original.tsv.full.raw.queries \
--batch --log-scores \
--topk /home/akrasak/cqa-rewrite/ColBERT/cast19-zeroshot-lastturn/retrieve.py/2021-07-21_11.32.20/unordered.tsv \
--mask_method last_turn --query_maxlen 384 \
--run 

```

* 19Jul
```
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py \
--path_ranking /home/akrasak/colbertdebug/ColBERT/last_column_results/rerank.py/19Jul/ranking.tsv

partial evaluation

{'map': 0.1791,
 'ndcg_cut_3': 0.2254,
 'recall_1000': 0.6621,
 'recip_rank': 0.4042}

```

* 20 Jul 

```
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py \
--path_ranking /home/akrasak/colbertdebug/ColBERT/last_column_results/rerank.py/20Jul/ranking.tsv

partial evaluation

{'map': 0.1791,
 'ndcg_cut_3': 0.2254,
 'recall_1000': 0.6621,
 'recip_rank': 0.4042}
```

* last_summer_commit
```
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py --path_ranking /home/akrasak/colbertdebug/ColBERT/last_column_results/rerank.py/last_summer_commit/ranking.tsv

partial evaluation

{'map': 0.119,
 'ndcg_cut_3': 0.1671,
 'recall_1000': 0.5328,
 'recip_rank': 0.3164}
```

* master branch (18-Oct)

```
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py --path_ranking /home/akrasak/colbertdebug/ColBERT/last_column_results/rerank.py/master/ranking.tsv

partial evaluation

{'map': 0.119,
 'ndcg_cut_3': 0.1671,
 'recall_1000': 0.5328,
 'recip_rank': 0.3164}

```

* master branch (18-Oct) + expansion_tokens=0 + Qlen=256

```
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py --path_ranking \
/home/akrasak/colbertdebug/ColBERT/last_column_results/rerank.py/master_noExp256/ranking.tsv

partial evaluation

{'map': 0.1048,
 'ndcg_cut_3': 0.1614,
 'recall_1000': 0.5171,
 'recip_rank': 0.2907}
```


```
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py --path_ranking \
/home/akrasak/colbertdebug/ColBERT/last_column_results/rerank.py/master_256/ranking.tsv

partial evaluation

{'map': 0.1048,
 'ndcg_cut_3': 0.1614,
 'recall_1000': 0.5171,
 'recip_rank': 0.2907}

```

* 15 Aug
```
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py --path_ranking \
/home/akrasak/colbertdebug/ColBERT/last_column_results/rerank.py/15Aug/ranking.tsv

partial evaluation

{'map': 0.119,
 'ndcg_cut_3': 0.1671,
 'recall_1000': 0.5328,
 'recip_rank': 0.3164}
```

* 16 Aug (52ccf582d651a2deda3356c337b061406ebbd623)

```

```

## Run CAsT19-raw run (params from decent column in excel)

```
CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=6 \
python -m colbert.rerank \
--amp --doc_maxlen 180 --mask-punctuation --bsize 256 --partitions 32768 \
--index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ \
--index_name MSMARCO.L2.32x200k.180len \
--checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn \
--root /home/akrasak/colbertdebug/ColBERT --experiment first_column_results \
--queries /home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/1_Original.tsv.raw.queries \
--batch --log-scores \
--topk /home/akrasak/cqa-rewrite/ColBERT/cast19-raw/retrieve.py/2021-06-11_16.30.27/unordered.tsv \
--run 
```

* zero-commit

```
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py --path_ranking \
/home/akrasak/colbertdebug/ColBERT/first_column_results/rerank.py/zero_commit/ranking.tsv

partial evaluation

{'map': 0.1826,
 'ndcg_cut_3': 0.2583,
 'recall_1000': 0.4515,
 'recip_rank': 0.4285}
```


* 19Jul

```
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py --path_ranking \
/home/akrasak/colbertdebug/ColBERT/first_column_results/rerank.py/19Jul/ranking.tsv

partial evaluation

{'map': 0.1826,
 'ndcg_cut_3': 0.2583,
 'recall_1000': 0.4515,
 'recip_rank': 0.4285}

```

* last_summer_commit

```
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py --path_ranking \
/home/akrasak/colbertdebug/ColBERT/first_column_results/rerank.py/last_summer_commit/ranking.tsv

partial evaluation 

{'map': 0.0324,
 'ndcg_cut_3': 0.0472,
 'recall_1000': 0.2282,
 'recip_rank': 0.1168}
```

* master

```
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py --path_ranking \
/home/akrasak/colbertdebug/ColBERT/first_column_results/rerank.py/master/ranking.tsv

partial evaluation

{'map': 0.0324,
 'ndcg_cut_3': 0.0472,
 'recall_1000': 0.2282,
 'recip_rank': 0.1168}
```

* 23 July

``` 
Traceback (most recent call last):
  File "/home/akrasak/anaconda3/envs/colbert/lib/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/home/akrasak/anaconda3/envs/colbert/lib/python3.7/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/home/akrasak/colbertdebug/ColBERT/colbert/rerank.py", line 55, in <module>
    main()
  File "/home/akrasak/colbertdebug/ColBERT/colbert/rerank.py", line 49, in main
    batch_rerank(args)
  File "/home/akrasak/colbertdebug/ColBERT/colbert/ranking/batch_reranking.py", line 85, in batch_rerank
    all_query_embeddings = inference.queryFromText(queries_in_order, bsize=512, to_cpu=True)
  File "/home/akrasak/colbertdebug/ColBERT/colbert/modeling/inference.py", line 36, in queryFromText
    batches = self.query_tokenizer.tensorize(queries, bsize=bsize, query_truncation=query_truncation, nr_expansion_tokens=nr_expansion_tokens)
```

* 20 July

``` 
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py --path_ranking \
/home/akrasak/colbertdebug/ColBERT/first_column_results/rerank.py/20Jul/ranking.tsv

partial evaluation

{'map': 0.1826,
 'ndcg_cut_3': 0.2583,
 'recall_1000': 0.4515,
 'recip_rank': 0.4285}

```


* 23 July 
```
    ids[(ids == 0).nonzero()[0][:nr_expansion_tokens]] = 0
IndexError: index 0 is out of bounds for dimension 0 with size 0
```

* 14 Aug (59b8143dc9)
I think that's where out of bounds error is fixed

```
  File "/home/akrasak/colbertdebug/ColBERT/colbert/modeling/tokenization/query_tokenization.py", line 69, in tensorize
    ids[:,start_pad_idx:start_pad_idx+nr_expansion_tokens] = self.mask_token_id
TypeError: can only concatenate list (not "int") to list

```

* 15 Aug (7b989ee0c)
Still working on index out of bounds error
```
python /home/akrasak/cqa-rewrite/ColBERT/evaluate_cast.py --path_ranking \
/home/akrasak/colbertdebug/ColBERT/first_column_results/rerank.py/15Aug/ranking.tsv

partial evaluation

{'map': 0.0324,
 'ndcg_cut_3': 0.0472,
 'recall_1000': 0.2282,
 'recip_rank': 0.1168}

```

## Run CAsT21 in different timestamps ?
How to spot code issues? 
- via evaluation? (should avoid running 3 indices & run over). can I do partial/unfair evaluation, in only 1 index? 
Results wont be comparable to trec
- via comparing runs (how? absolute comparison seems harsh. Rank correlation?)

### CAsT21 run @ MARCO (marco_lastturnmask)

```
CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=6 \
python -m colbert.rerank \
--amp --doc_maxlen 180 --mask-punctuation --bsize 1 --partitions 32768 \
--index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ --index_name marco.FirstP.L2.32x200k.180len \
--checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn \
--root /home/akrasak/colbertdebug/ColBERT/ \
--experiment cast21-marco_lastturnmask \
--topk /home/akrasak/cqa-rewrite/ColBERT/cast21-marco_lastturnmask/retrieve.py/2021-08-18_11.49.47/unordered.tsv \
--queries /ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_with_canonical_256.tsv \
--batch --log-scores --query_maxlen 256 --mask_method last_turn \
--run 
```


```
python /home/akrasak/cqa-rewrite/ColBERT/preprocessing/evaluate_cast21.py \
--path_ranking  /home/akrasak/cqa-rewrite/ColBERT/cast21-marco_lastturnmask/rerank.py/2021-08-18_12.18.34/ranking.tsv
```


```
Submitted runs:

/ivi/ilps/personal/akrasak/data/cast21_submission/queries_hist_canonical_256.trecrun
/ivi/ilps/personal/akrasak/data/cast21_submission/histonly.trecrun
```
```
qrel_path='/ivi/ilps/personal/akrasak/data/cqa-rewrite/qrels/qrels-docs.2021.txt'
```

```

path_ranking='/home/akrasak/cqa-rewrite/ColBERT/cast21-marco_lastturnmask/rerank.py/2021-08-18_12.18.34/ranking.tsv'

trec_eval -m recall.1000 -m map -m recip_rank  -m ndcg_cut.3 -c  $qrel_path $path_ranking
```

# First stage ranking comparison

## Retrieve code that produces ranking 

```
CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=6 \
python -m colbert.retrieve \
--amp --doc_maxlen 180 --query_maxlen 256 --mask-punctuation --bsize 1 \
--queries /ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_with_canonical_256.tsv \
--nprobe 32 --partitions 32768 --faiss_depth 1024 --index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ \
--index_name wapo.FirstP.L2.32x200k.180len --checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn \
--root /home/akrasak/colbertdebug/ColBERT/ \
--experiment cast21-wapo_lastturnmask \
--mask_method last_turn 

```

Only works in small indexes (ie. WAPO), because I cannot load it to RAM.

## zeroshot-lastturn-masked (CAsT19)
```
CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=6 \
python -m colbert.retrieve \
--amp --doc_maxlen 180 --mask-punctuation --bsize 1 --nprobe 32 --partitions 32768 \
--faiss_depth 1024 --index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ \
--index_name MSMARCO.L2.32x200k.180len --checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn \
--root /home/akrasak/colbertdebug/ColBERT/ --experiment cast19-zeroshot-lastturn-retrieve \
--queries /home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/1_Original.tsv.full.raw.queries \
--mask_method last_turn --query_maxlen 512 \
--retrieve_only --batch \
--run 
```
* 20Jul
* last_summer_commit
* zero_commit (without mask_method)


## CAsT19-raw
```
CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=6 \
python -m colbert.retrieve \
--amp --doc_maxlen 180 --mask-punctuation --bsize 256 --nprobe 32 --partitions 32768 \
--faiss_depth 1024 --index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ \
--index_name MSMARCO.L2.32x200k.180len --checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn \
--root /home/akrasak/colbertdebug/ColBERT/ --experiment cast19-raw-retrieve \
--queries /home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/1_Original.tsv.raw.queries \
--batch --retrieve_only \
--run

```

TODO:
- run wapo last turn mask (reranking) on cn111
- run zero_commit (retrieve)
- run CAsT19-raw (retrieve)
