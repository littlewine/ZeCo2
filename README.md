# Zero-shot Query Contextualization for Conversational Search

### Code for reproducing the SIGIR '22 paper

<p align="center">
  <img align="center" src="ZeCo2.pdf" />
</p>
<p align="center">
  <b>:</b> ZeCo<sup>2</sup> contextualizes the user question within the conversation history, but restrict the matching only between question and potential answer.
</p>


* [**ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**](https://arxiv.org/abs/2004.12832) (SIGIR'20).


----

## How to reproduce results:

1. Install colbert 
2. Update various paths (eg. query, pretrained model path etc.) @ `paths.py`
3. Corpus indexing: use a ColBERT model checkpoint and see [index your collection](README_ColBERT.md#ColBERT Indexing). Note that preprocessing scripts are available in `/preprocessing`. To make use our pipeline for retrieval and evaluation, you need to convert passage ids to integers and retain a mapping file (`.intmapping`) before indexing (see `preprocessing/README.md`).
4. Retrieve & rerank using the available pipeline:

`python_pipeline.py --setting ZeCo2 --dataset cast19`

For ColBERT-related questions, instructions, etc. please refer to the [original repository (forked from v0.2.0)](https://github.com/stanford-futuredata/ColBERT/tree/efaabb0f8731c7d96a9fe109a125357a9232f7a7) or `README_ColBERT.md`, or feel free to raise an issue!
