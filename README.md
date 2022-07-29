Code for reproducing the SIGIR '22 paper:
# Zero-shot Query Contextualization for Conversational Search



<p align="center">
  <img align="center" src="ZeCo2.pdf" />
</p>
<p align="center">
  <b>:</b> ZeCo<sup>2</sup> contextualizes the user question within the conversation history, but restrict the matching only between question and potential answer.
</p>


* [**ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**](https://arxiv.org/abs/2004.12832) (SIGIR'20).


----

## Reproducing results:

1. Install colbert 
2. Download colbert model checkpoint & update various paths (see the #TODO @ `paths.py` )
3. Corpus indexing: create FAISS indexes using a ColBERT model (see [index your collection](README_ColBERT.md#ColBERT Indexing). Note that preprocessing scripts are available in `/preprocessing`. To make use our pipeline for retrieval and evaluation, you need to convert query&passage ids to integers and retain a mapping file (`.intmapping`) before indexing (see the [preprocessing README](preprocessing/README.md) and [examples](data/collection_samples)).
4. Retrieve & rerank using the available pipeline:

`python_pipeline.py --setting ZeCo2 --dataset cast19`

### Paper analysis section:
The two scripts used to reproduce the analysis section of the paper are:

```
token_embedding_change.py
embedding_closest_terms.py
```

You can already run the analysis since the final rankings are provided under `data/rankings/`

----

For ColBERT-related questions, instructions, etc. please refer to the [original repository (forked from v0.2.0)](https://github.com/stanford-futuredata/ColBERT/tree/efaabb0f8731c7d96a9fe109a125357a9232f7a7) or `README_ColBERT.md`, or feel free to raise an issue!

----

