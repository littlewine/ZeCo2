Start server (p40 needed only when query length is big)

`srun -c 24 --mem=230gb -p gpu --gres=gpu:p40:4 --time=48:00:00 --exclude=ilps-cn108  --pty bash`


```
source ~/.bash_profile
conda activate colbert
cd ~/cqa-rewrite/ColBERT/
export PYTHONPATH=$PYTHONPATH:$HOME/cqa-rewrite/ColBERT
```

```
# Define common paths
path_root='/home/akrasak/cqa-rewrite/ColBERT/'
path_model_checkpoint='/ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn'
path_indexes='/ivi/ilps/personal/akrasak/data/faiss_indexes/'

# Define queries
path_queries_fullconv='/home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/1_Original.tsv.full.raw.queries'
path_queries_raw='/home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/1_Original.tsv.raw.queries'
path_queries_human='/home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/10_Human.tsv.queries'
path_queries_human_BOW='/home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/9_Human_BoW_Q.tsv.queries'
path_queries_quretec='/home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/5_QuReTeC_Q.tsv.queries'

# Define indexes
index_name_CAR='CAR.FirstP.L2.32x200k.180len'
index_name_MARCOPassage='MSMARCO.L2.32x200k.180len'
```


TODO: fix path_queries_human and quretec 

# CAsT19 experiments

## 1. Zero-shot model on CAsT19 (submitted) `[cast19-zeroshot-masked]`
Params:
```
path_queries=$path_queries_cast19_fullconv
exp_name='cast19-zeroshot-masked'
mask_method_param=' --mask_method last_turn '
```

`index_name=$index_name_CAR`
`index_name=$index_name_MARCOPassage`

* First-stage ranking
```
echo "Experiment details:" $exp_name $path_queries
CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=6 \
python -m colbert.retrieve \
--amp --doc_maxlen 180 --query_maxlen 256 \
--mask-punctuation --bsize 1 \
--nprobe 32 --partitions 32768 --faiss_depth 1024 \
--index_root $path_indexes --index_name $index_name \
--checkpoint $path_model_checkpoint \
--root $path_root --experiment $exp_name --queries $path_queries \
--batch --retrieve_only \
--run $exp_name-${index_name:0:5} \
$mask_method_param
```
Results:

`filepath_retrieve_output='/home/akrasak/cqa-rewrite/ColBERT/cast19-zeroshot-masked/retrieve.py/cast19-zeroshot-masked-MSMAR/unordered.tsv'`

`filepath_retrieve_output='/home/akrasak/cqa-rewrite/ColBERT/cast19-zeroshot-masked/retrieve.py/cast19-zeroshot-masked-CAR.F/unordered.tsv'`



* Re-ranking

```
CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=6 \
python -m colbert.rerank \
--amp --doc_maxlen 180 --mask-punctuation --bsize 256 \
--partitions 32768 --query_maxlen 256 \
--index_root $path_indexes --index_name $index_name \
--checkpoint $path_model_checkpoint \
--root $path_root --experiment $exp_name --queries $path_queries \
--batch --log-scores \
 $mask_method_param \
--run $exp_name-${index_name:0:5} \
--topk $filepath_retrieve_output

```
Results:
`filepath_rerank_output='/home/akrasak/cqa-rewrite/ColBERT/cast19-zeroshot-masked/rerank.py/cast19-zeroshot-masked-MSMAR/ranking.tsv'`

`filepath_rerank_output='/home/akrasak/cqa-rewrite/ColBERT/cast19-zeroshot-masked/rerank.py/cast19-zeroshot-masked-CAR.F/ranking.tsv'`

...



## 2. With raw utterances (baseline) `[cast19-raw]`
Params:
```
path_queries=$path_queries_raw
exp_name='cast19-raw'
mask_method_param=' '

```

`index_name=$index_name_MARCOPassage`
`index_name=$index_name_CAR`

Results:

`filepath_retrieve_output='/home/akrasak/cqa-rewrite/ColBERT/cast19-raw/retrieve.py/cast19-raw-MSMAR/unordered.tsv'`

`filepath_retrieve_output='/home/akrasak/cqa-rewrite/ColBERT/cast19-raw/retrieve.py/cast19-raw-CAR.F/unordered.tsv'`


`filepath_rerank_output='/home/akrasak/cqa-rewrite/ColBERT/cast19-raw/rerank.py/cast19-raw-MSMAR/ranking.tsv'`

`filepath_rerank_output='/home/akrasak/cqa-rewrite/ColBERT/cast19-raw/rerank.py/cast19-raw-CAR.F/ranking.tsv'`

## 3. With human utterances (oracle) `[cast19-human]`
Params:
```
path_queries=$path_queries_human
exp_name='cast19-human'
mask_method_param=' '

```

`index_name=$index_name_MARCOPassage`
`index_name=$index_name_CAR`


`filepath_retrieve_output='/home/akrasak/cqa-rewrite/ColBERT/cast19-human/retrieve.py/cast19-human-MSMAR/unordered.tsv'`

`filepath_retrieve_output='/home/akrasak/cqa-rewrite/ColBERT/cast19-human/retrieve.py/cast19-human-CAR.F/unordered.tsv'`


`filepath_rerank_output='/home/akrasak/cqa-rewrite/ColBERT/cast19-human/rerank.py/cast19-human-MSMAR/ranking.tsv'`

`filepath_rerank_output='/home/akrasak/cqa-rewrite/ColBERT/cast19-human/rerank.py/cast19-human-CAR.F/ranking.tsv'`


## 4. With human BOW utterances (oracle) `[cast19-humanBOW]`
Params:
```
path_queries=$path_queries_human_BOW
exp_name='cast19-humanBOW'
mask_method_param=' '

```

`index_name=$index_name_MARCOPassage`
`index_name=$index_name_CAR`


`filepath_retrieve_output='/home/akrasak/cqa-rewrite/ColBERT/cast19-humanBOW/retrieve.py/cast19-humanBOW-MSMAR/unordered.tsv'`

`filepath_retrieve_output='/home/akrasak/cqa-rewrite/ColBERT/cast19-humanBOW/retrieve.py/cast19-humanBOW-CAR.F/unordered.tsv'`


`filepath_rerank_output='/home/akrasak/cqa-rewrite/ColBERT/cast19-humanBOW/rerank.py/cast19-humanBOW-MSMAR/ranking.tsv'`

`filepath_rerank_output='/home/akrasak/cqa-rewrite/ColBERT/cast19-humanBOW/rerank.py/cast19-humanBOW-CAR.F/ranking.tsv'`


# Postprocess runs
```
filepath_MARCOP_mapping='/ivi/ilps/personal/akrasak/data/collections/msmarco-passage/collection.tsv.intmapping'
filepath_CAR_mapping='/ivi/ilps/personal/akrasak/data/collections/car-wiki2020-01-01/Car_collection.tsv.intmapping'

filepath_rerank_output_MARCOP=''
filepath_rerank_output_CAR=''

python preprocessing/postprocess_runs.py --run $filepath_rerank_output_MARCOP $filepath_rerank_output_CAR --mapping $filepath_MARCOP_mapping $filepath_CAR_mapping 

```

```

python preprocessing/postprocess_runs.py \
--run /home/akrasak/cqa-rewrite/ColBERT/cast19-zeroshot-masked/rerank.py/cast19-zeroshot-masked-MSMAR/ranking.tsv \
/home/akrasak/cqa-rewrite/ColBERT/cast19-zeroshot-masked/rerank.py/cast19-zeroshot-masked-CAR.F/ranking.tsv \
--mapping $filepath_MARCOP_mapping $filepath_CAR_mapping \
--filepath_output /home/akrasak/cqa-rewrite/ColBERT/cast19-zeroshot-masked/rerank.py/ranking.tsv \
--run_id cast19-zeroshot-masked --topk 1000 --topic_len 2

python preprocessing/postprocess_runs.py \
--run /home/akrasak/cqa-rewrite/ColBERT/cast19-raw/rerank.py/cast19-raw-MSMAR/ranking.tsv \
/home/akrasak/cqa-rewrite/ColBERT/cast19-raw/rerank.py/cast19-raw-CAR.F/ranking.tsv \
--mapping $filepath_MARCOP_mapping $filepath_CAR_mapping \
--filepath_output /home/akrasak/cqa-rewrite/ColBERT/cast19-raw/rerank.py/ranking.tsv \
--run_id cast19-raw --topk 1000 --topic_len 2

python preprocessing/postprocess_runs.py \
--run /home/akrasak/cqa-rewrite/ColBERT/cast19-human/rerank.py/cast19-human-MSMAR/ranking.tsv \
/home/akrasak/cqa-rewrite/ColBERT/cast19-human/rerank.py/cast19-human-CAR.F/ranking.tsv \
--mapping $filepath_MARCOP_mapping $filepath_CAR_mapping \
--filepath_output /home/akrasak/cqa-rewrite/ColBERT/cast19-human/rerank.py/ranking.tsv \
--run_id cast19-human --topk 1000 --topic_len 2

python preprocessing/postprocess_runs.py \
--run /home/akrasak/cqa-rewrite/ColBERT/cast19-humanBOW/rerank.py/cast19-humanBOW-MSMAR/ranking.tsv \
/home/akrasak/cqa-rewrite/ColBERT/cast19-humanBOW/rerank.py/cast19-humanBOW-CAR.F/ranking.tsv \
--mapping $filepath_MARCOP_mapping $filepath_CAR_mapping \
--filepath_output /home/akrasak/cqa-rewrite/ColBERT/cast19-humanBOW/rerank.py/ranking.tsv \
--run_id cast19-humanBOW --topk 1000 --topic_len 2

```

# Evaluation

```

path_qrel='/ivi/ilps/personal/akrasak/data/cqa-rewrite/qrels/2019qrels.txt'
quretec_run='/home/akrasak/cqa-rewrite/cast_evaluation/runs/2019/52_Quretec_Q_reranked.txt'

ranking='/home/akrasak/cqa-rewrite/ColBERT/cast19-human/rerank.py/ranking.tsv'

trec_eval -m recall.1000 -m map -m recip_rank -m ndcg_cut.3 \
-c -q $path_qrel $ranking 
```


