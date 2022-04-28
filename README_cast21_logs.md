cast 21 Run
```
srun -c 24 --mem=218gb -p  gpu --gres=gpu:pascalxp:4 --exclude=ilps-cn108 --time=100:00:00 --pty bash

srun -c 42 --mem=720gb -p  cpu --time=300:00:00 --pty bash


source ~/.bash_profile
conda activate colbert
cd ~/cqa-rewrite/ColBERT/
pwd


CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=6

******

path_queries='/ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_with_canonical_384.tsv'
path_queries='/ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_with_canonical_256.tsv'
path_queries='/ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_history_only.tsv'

index_name='wapo.FirstP.L2.32x200k.180len'
exp_name='cast21-wapo_lastturnmask-histonly'

index_name='kilt.FirstP.L2.32x200k.180len'
exp_name='cast21-kilt_lastturnmask-histonly' 

index_name='marco.FirstP.L2.32x200k.180len' 
exp_name='cast21-marco_lastturnmask-histonly' 

********************

path_queries='/ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_canonical_last.tsv'

index_name='wapo.FirstP.L2.32x200k.180len'
exp_name='cast21-wapo_lastturnmask-canonical'
unordered_path='/home/akrasak/cqa-rewrite/ColBERT/cast21-wapo_lastturnmask-canonical/retrieve.py/2021-08-19_10.55.57/unordered.tsv'

index_name='kilt.FirstP.L2.32x200k.180len'
exp_name='cast21-kilt_lastturnmask-canonical' 
unordered_path=

index_name='marco.FirstP.L2.32x200k.180len' 
exp_name='cast21-marco_lastturnmask-canonical' 
unordered_path='/home/akrasak/cqa-rewrite/ColBERT/cast21-marco_lastturnmask-canonical/retrieve.py/2021-08-19_11.22.35/unordered.tsv'

path_queries='/ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_canonical_last.tsv'

********************

python -m colbert.retrieve \
--amp --doc_maxlen 180  --query_maxlen 254 --mask-punctuation --bsize 1 \
--queries $path_queries \
--nprobe 32 --partitions 32768 --faiss_depth 1024 \
--index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ --index_name $index_name \
--checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn \
--batch --retrieve_only \
--root /home/akrasak/cqa-rewrite/ColBERT/ --experiment $exp_name --mask_method last_turn

echo "1st stage ranking results saved to:"
echo $retrieve_filepath

******

unordered_path=$'/home/akrasak/cqa-rewrite/ColBERT/cast21-kilt_lastturnmask/retrieve.py/2021-08-18_10.00.13/unordered.tsv'

unordered_path=$'/home/akrasak/cqa-rewrite/ColBERT/cast21-wapo_lastturnmask/retrieve.py/2021-08-17_13.26.11/ranking.tsv'

unordered_path=$'/home/akrasak/cqa-rewrite/ColBERT/cast21-wapo_lastturnmask-histonly/retrieve.py/2021-08-19_09.56.47/unordered.tsv'

python -m colbert.rerank \
--amp --doc_maxlen 180 --mask-punctuation --bsize 1 --partitions 32768 \
--index_root /ivi/ilps/personal/akrasak/data/faiss_indexes/ \
--index_name $index_name --checkpoint /ivi/ilps/personal/akrasak/data/models/colbert-400000.dnn \
--root  /home/akrasak/cqa-rewrite/ColBERT/ \
--experiment $exp_name \
 --topk $unordered_path \
 --queries $path_queries \
 --batch --log-scores --query_maxlen 254 --mask_method last_turn


*/home/akrasak/cqa-rewrite/cast_evaluation/rewrites/2019/1_Original.tsv.full.raw.queries
```

---

Logs

```


******

	256 WAPO:

- retrieve: 
/home/akrasak/cqa-rewrite/ColBERT/cast21-wapo_lastturnmask/retrieve.py/2021-08-17_13.26.11/ranking.tsv

-rerank:
/home/akrasak/cqa-rewrite/ColBERT/cast21-wapo_lastturnmask/rerank.py/2021-08-18_19.50.47/ranking.tsv

384 WAPO:
/home/akrasak/cqa-rewrite/ColBERT/cast21-wapo_lastturnmask/retrieve.py/2021-08-18_15.44.56/unordered.tsv


************************

	256 KILT:
- retrieve:
/home/akrasak/cqa-rewrite/ColBERT/cast21-kilt_lastturnmask/retrieve.py/2021-08-18_10.00.13/unordered.tsv

- rerank:
/home/akrasak/cqa-rewrite/ColBERT/cast21-kilt_lastturnmask/rerank.py/2021-08-18_11.01.31/ranking.tsv

	384 KILT:
retrieve:
/home/akrasak/cqa-rewrite/ColBERT/cast21-kilt_lastturnmask/retrieve.py/2021-08-18_16.05.55/unordered.tsv

************************

	256 MARCO:

- retrieve:
/home/akrasak/cqa-rewrite/ColBERT/cast21-marco_lastturnmask/retrieve.py/2021-08-18_11.49.47/unordered.tsv

- rerank:
/home/akrasak/cqa-rewrite/ColBERT/cast21-marco_lastturnmask/rerank.py/2021-08-18_12.18.34/ranking.tsv

	384 MARCO:
- retrieve:
/home/akrasak/cqa-rewrite/ColBERT/cast21-marco_lastturnmask/retrieve.py/2021-08-18_16.25.52/unordered.tsv


************************************************************************************************

```

```
$mapping_kilt='/ivi/ilps/personal/akrasak/data/collections/cast21sep/kilt_knowledgesource.tsv.intmapping'

$mapping_marco='/ivi/ilps/personal/akrasak/data/collections/cast21sep/msmarco-docs.tsv.intmapping'

$mapping_wapo='/ivi/ilps/personal/akrasak/data/collections/cast21sep/wapo.tsv_incomplete.intmapping'
```
---
postprocessing

```

python postprocess_runs.py \
--run \
/home/akrasak/cqa-rewrite/ColBERT/cast21-kilt_lastturnmask/rerank.py/2021-08-18_11.01.31/ranking.tsv \
/home/akrasak/cqa-rewrite/ColBERT/cast21-wapo_lastturnmask/rerank.py/2021-08-18_19.50.47/ranking.tsv \
/home/akrasak/cqa-rewrite/ColBERT/cast21-marco_lastturnmask/rerank.py/2021-08-18_12.18.34/ranking.tsv \
--mapping \
/ivi/ilps/personal/akrasak/data/collections/cast21sep/kilt_knowledgesource.tsv.intmapping \
/ivi/ilps/personal/akrasak/data/collections/cast21sep/wapo.tsv_incomplete.intmapping \
/ivi/ilps/personal/akrasak/data/collections/cast21sep/msmarco-docs.tsv.intmapping \
--filepath_output /ivi/ilps/personal/akrasak/data/cast21_submission/queries_hist_canonical_256.trecrun \
--run_id astypalaia256 \
--topic_len 3


--run /home/akrasak/cqa-rewrite/ColBERT/cast21-kilt_lastturnmask/rerank.py/2021-08-18_11.01.31/ranking.tsv /home/akrasak/cqa-rewrite/ColBERT/cast21-wapo_lastturnmask/rerank.py/2021-08-17_13.38.26/ranking.tsv --mapping /ivi/ilps/personal/akrasak/data/collections/cast21sep/kilt_knowledgesource.tsv.intmapping /ivi/ilps/personal/akrasak/data/collections/cast21sep/wapo.tsv_incomplete.intmapping --filepath_output /ivi/ilps/personal/akrasak/data/cast21_submission/test_wapo_kilt.trecrun --run_id test_wapo_kilt --topic_len 3
```

---

History only runs

```
path_queries='/ivi/ilps/personal/akrasak/data/treccastweb/2021/queries_history_only.tsv


* wapo * 
unordered_path='/home/akrasak/cqa-rewrite/ColBERT/cast21-wapo_lastturnmask-histonly/retrieve.py/2021-08-19_09.56.47/unordered.tsv'

/home/akrasak/cqa-rewrite/ColBERT/cast21-wapo_lastturnmask-histonly/rerank.py/2021-08-19_10.05.23/ranking.tsv



* kilt * 

unordered_path='/home/akrasak/cqa-rewrite/ColBERT/cast21-kilt_lastturnmask-histonly/retrieve.py/2021-08-19_10.23.20/unordered.tsv'

/home/akrasak/cqa-rewrite/ColBERT/cast21-kilt_lastturnmask-histonly/rerank.py/2021-08-19_12.05.02/ranking.tsv

* marco *

unordered_path='/home/akrasak/cqa-rewrite/ColBERT/cast21-marco_lastturnmask-histonly/retrieve.py/2021-08-19_10.34.41/unordered.tsv'

/home/akrasak/cqa-rewrite/ColBERT/cast21-marco_lastturnmask-histonly/retrieve.py/2021-08-19_10.34.41/unordered.tsv

/home/akrasak/cqa-rewrite/ColBERT/cast21-marco_lastturnmask-histonly/rerank.py/2021-08-19_10.49.46/ranking.tsv

```

---
Canonical only runs

```

/home/akrasak/cqa-rewrite/ColBERT/cast21-wapo_lastturnmask-canonical/rerank.py/2021-08-19_12.59.44/ranking.tsv
/home/akrasak/cqa-rewrite/ColBERT/cast21-marco_lastturnmask-canonical/rerank.py/2021-08-19_13.10.46/ranking.tsv
/home/akrasak/cqa-rewrite/ColBERT/cast21-kilt_lastturnmask-histonly/retrieve.py/2021-08-19_11.00.01/unordered.tsv

*****

python postprocess_runs.py \
--run \
/home/akrasak/cqa-rewrite/ColBERT/cast21-wapo_lastturnmask-histonly/rerank.py/2021-08-19_10.05.23/ranking.tsv \
/home/akrasak/cqa-rewrite/ColBERT/cast21-marco_lastturnmask-histonly/rerank.py/2021-08-19_10.49.46/ranking.tsv \
/home/akrasak/cqa-rewrite/ColBERT/cast21-kilt_lastturnmask-histonly/rerank.py/2021-08-19_12.05.02/ranking.tsv \
--mapping \
/ivi/ilps/personal/akrasak/data/collections/cast21sep/wapo.tsv_incomplete.intmapping \
/ivi/ilps/personal/akrasak/data/collections/cast21sep/msmarco-docs.tsv.intmapping \
/ivi/ilps/personal/akrasak/data/collections/cast21sep/kilt_knowledgesource.tsv.intmapping \
--filepath_output /ivi/ilps/personal/akrasak/data/cast21_submission/historyonlyKILT.trecrun \
--run_id historyonlyKILT \
--topk 200 \
--topic_len 3

```