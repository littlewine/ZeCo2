This folder provides various helpers for collection preprocessing, indexing, 
as well as postprocessing runfiles and evaluating (mostly CAsT related issues).

# Collection/indexing tools:

Colbert indexing requires documents to be in tab-separated files, see here `../README_ColBERT.md/#`
To use the provided pipeline, you need a faiss generated index, along with a mapping to 


```
/data/collection_samples/
```

## Converting trecweb files to tsv:
``` 
preprocessing_treccast_to_tsv.py
collection_preprocess.py
```
* Check tsv validity:
`check_tsv.py`

* Create collection mapping (to map doc `12837 -> MARCO_12837` since all docids have to be integers):
` append_collection_prefix.py`

For CAsT'21 the collection was created from the trecweb files, provided by the track organizers.

## Query preprocessing
`preprocessing_queries*.py`

Creates the conversational queries. 

(Not needed since the queries are already provided.)

# Post-processing/evaluation tools:
```
postprocess_runs.py 
evaluator.py
```
The postprocessing script is required to run the automated pipeline. 

Specifically it:
* fixes document and query ids (eg doc `12837 -> MARCO_12837`, query `1011->101_1`). requires certain mappings (replace paths at `paths.py`)
* merges rankings from different collections/indexes into one and 
* converts to the standard `trec_eval` format

## Analysis tools:
```
compare_rankings.py
compare_retrieval_sets.py
```
Some of those are needed for the analysis section of the paper.