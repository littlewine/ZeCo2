import torch

from colbert.modeling.colbert import ColBERT
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.utils.amp import MixedPrecisionManager
from colbert.parameters import DEVICE
from transformers import BertTokenizerFast
import numpy as np


class ModelInference():
    def __init__(self, colbert: ColBERT, amp=False, debug=False,
                 mask_method=None, add_CLSQ_tokens=True, nr_expansion_tokens=10,
                 add_SEP_tokens=True):
        assert colbert.training is False

        self.colbert = colbert
        self.query_tokenizer = QueryTokenizer(colbert.query_maxlen)
        self.doc_tokenizer = DocTokenizer(colbert.doc_maxlen)
        self.bert_tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')

        self.amp_manager = MixedPrecisionManager(amp)
        self.debug = debug
        self.mask_method = mask_method
        self.add_CLSQ_tokens = add_CLSQ_tokens
        self.nr_expansion_tokens = nr_expansion_tokens
        self.add_SEP_tokens = add_SEP_tokens

    def query(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                Q = self.colbert.query(*args, **kw_args)
                input_ids = args[0]
                if self.mask_method == 'ZeCo2':
                    Q = self.crop_query_tensor(Q, input_ids=input_ids)
                elif (self.mask_method is None):
                    # handle CLS,Q,[MASK] matching if query cropping is skipped,
                    skip_mask = (input_ids==self.bert_tokenizer.pad_token_id) #skip pads
                    if not self.add_CLSQ_tokens:
                        skip_mask += (input_ids == self.query_tokenizer.cls_token_id)
                        skip_mask += (input_ids == self.query_tokenizer.Q_marker_token_id)
                    if not self.add_SEP_tokens:
                        skip_mask += (input_ids == self.query_tokenizer.sep_token_id)

                    match_mask = torch.logical_not(skip_mask)
                    match_mask = match_mask.unsqueeze(2).expand(-1, -1, 128).cuda() # add 128 dimension
                    Q = Q*match_mask

                    # crop empty columns (if too many paddings in all queries)
                    non_empty_cols = torch.nonzero(Q.abs().sum(dim=0).bool().sum(1), as_tuple=True)[0]
                    Q = Q[:,non_empty_cols,:]

                print(f"self.mask_method: {self.mask_method}\n"
                      f"self.nr_expansion_tokens: {self.nr_expansion_tokens}\n"
                      f"self.add_CLSQ_tokens: {self.add_CLSQ_tokens}\n"
                      f"Q tensor shape:{Q.shape}\n")

                print(f"Match tokens for each query: {(Q[:,:,0]!=0).sum(1)}")
                return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                D = self.colbert.doc(*args, **kw_args)
                return D.cpu() if to_cpu else D

    def crop_query_tensor(self, Q, input_ids):
        # https://stackoverflow.com/questions/60891052/how-can-i-trim-a-tensor-based-on-a-mask-with-pytorch
        # https://pytorch.org/docs/stable/generated/torch.masked_select.html
        assert self.mask_method == 'ZeCo2'
        cropped_Q_tensors = []
        if self.debug:
            print("Final query tokens/inputIDs to be matched with docs:")

        # Find and crop last query terms
        for i, query in enumerate(input_ids):
            sep_positions = (query == self.bert_tokenizer.sep_token_id).nonzero(as_tuple=True)[-1].tolist()
            if len(sep_positions)<2:
                start_idx = 2 # TODO: solve issue here: first two tokens are cut
            else:
                start_idx = sep_positions[-2] + 1 # second from last sep position + 1
            try:
                end_idx = (query == 0).nonzero()[0].item() -1 # one before first pad token
            except:
                end_idx = Q.shape[1] -1
            if self.debug:
                final_query = query[start_idx:end_idx+1]
                print(query[start_idx], query[end_idx], final_query)
                print(self.bert_tokenizer.decode(final_query.tolist()))
            # concatenate Q_tensor to list
            cropped_Q_tensors.append(Q[i,start_idx:end_idx+1,:])

        last_turn_Q = torch.nn.utils.rnn.pad_sequence(cropped_Q_tensors, batch_first=True) # pad
        if self.add_CLSQ_tokens: #add CLS, Q tokens to matching
            final_Q = torch.cat([Q[:, 0:2, :], last_turn_Q], dim=1)
        else:
            final_Q = last_turn_Q
        # if normalize_final_Q:
        #     final_Q = torch.nn.functional.normalize(final_Q, p=2, dim=2)
        print(f"Queries to be matched (with expansion tokens):\n"
              f"last_turn_Q.shape: {last_turn_Q.shape}\n"
              f"final_Q.shape: {final_Q.shape}")
        return final_Q

    def queryFromText(self, queries, bsize=None, to_cpu=False):
        if bsize:
            batches = self.query_tokenizer.tensorize(queries, bsize=bsize,
                                                     nr_expansion_tokens=self.nr_expansion_tokens,
                                                     debug=self.debug)
            batches = [self.query(input_ids, attention_mask, to_cpu=to_cpu
                                  ) for input_ids, attention_mask in batches]

            batches = torch.cat(batches) # TODO: pad different lengths if many batches
            return batches

        input_ids, attention_mask = self.query_tokenizer.tensorize(queries,
                                                                   nr_expansion_tokens=self.nr_expansion_tokens,
                                                                   debug=self.debug)
        Q = self.query(input_ids, attention_mask)

        # Construct query matching mask
        if self.mask_method is None:
            if self.add_CLSQ_tokens:
                Q_mask = torch.ones_like(Q)
            else:
                Q_mask = torch.cat([torch.zeros_like(Q[:,:2,:]),
                           torch.ones_like(Q[:,2:,:])], dim=1)

        elif self.mask_method == 'ZeCo2':
            sep_positions = (input_ids == self.query_tokenizer.sep_token_id).nonzero(as_tuple=True)[-1].tolist()

            if len(sep_positions) < 2: # No need to mask, take all query
                Q_mask = torch.ones_like(Q)
            elif len(sep_positions) >= 2: # more than 1 turns present
                # apply mask between last two tokens:
                end_idx = sep_positions[-1]
                start_idx = sep_positions[-2]

                nr_first_tokens_match = 2 # Include CLS token + query token (id=1)
                if self.add_CLSQ_tokens:
                    cls_q_vector = torch.ones([Q.shape[0], nr_first_tokens_match, Q.shape[2] ]) # [ [CLS], [Q] ]
                else:
                    cls_q_vector = torch.zeros([Q.shape[0], nr_first_tokens_match, Q.shape[2]])  # [ [CLS], [Q] ]
                a = torch.zeros([Q.shape[0], start_idx-nr_first_tokens_match, Q.shape[2]]) # previous turns
                b = torch.ones([Q.shape[0], end_idx - start_idx, Q.shape[2]]) # last turn
                c = torch.ones([Q.shape[0], self.nr_expansion_tokens, Q.shape[2]]) # expansion tokens ([MASK] x nr_expansion_tokens )
                d = torch.zeros([Q.shape[0],
                                 self.colbert.query_maxlen-end_idx-self.nr_expansion_tokens ,
                                 Q.shape[2]]) # [MASK] -> [PAD] #TODO: this should ideally be replaced with [PAD] token

                Q_mask = torch.cat([cls_q_vector, a, b, c, d], dim=1).to(Q.device)

                if self.debug:
                    print('')
                    print(f"positions of SEP token for query {queries} : {sep_positions}")  # debug
                    print(f"Shape a: {a.shape}")  # debug
                    print(f"Shape b: {b.shape}")  # debug

                    print(f"Shape Q_mask: {Q_mask.shape}")  # debug

                    non_zero_tokens = Q_mask[0, :, 0].to('cpu')
                    non_masked_token_ids = (non_zero_tokens * input_ids.to('cpu'))[0]
                    non_masked_token_ids = [int(x) for x in non_masked_token_ids.tolist()]
                    non_masked_token_ids_wo_pads = [x for x in non_masked_token_ids if x != 103]
                    print(f"Tokens matched: {non_masked_token_ids_wo_pads}")
                    # print(f"Matching query on {len(non_masked_token_ids_wo_pads)} tokens ")  # debug
                    cont_Q = self.bert_tokenizer.decode(non_masked_token_ids)
                    print(f"Contextualized query to be matched (without masks):"
                          f"{cont_Q.replace('[MASK] ','')}")
                    print('')

        # Recalculate Q tensor
        try:
            assert (Q_mask.shape == Q.shape)
        except AssertionError:
            print('Masked query tensor does not match original Query tensor')
        Q_masked = torch.mul(Q, Q_mask)
        nonzero_pos = np.nonzero(Q_masked.sum(2).tolist()[0])[0]
        print(f"Will do matching on {len(nonzero_pos)} tokens")

        print(f"Match tokens:\t"
              f"{self.bert_tokenizer.decode(input_ids[0][nonzero_pos].tolist())}")

        if self.debug:
            if not torch.all(torch.eq(Q, Q_masked)).item(): #debug
                print("some tokens were masked (skipped) for matching") #debug
        return Q_masked

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False):
        if bsize:
            batches, reverse_indices = self.doc_tokenizer.tensorize(docs, bsize=bsize)

            batches = [self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)
                       for input_ids, attention_mask in batches]

            if keep_dims:
                D = _stack_3D_tensors(batches)
                return D[reverse_indices]

            D = [d for batch in batches for d in batch]
            return [D[idx] for idx in reverse_indices.tolist()]

        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims)

    def score(self, Q, D, mask=None, lengths=None, explain=False):
        if lengths is not None:
            assert mask is None, "don't supply both mask and lengths"

            mask = torch.arange(D.size(1), device=DEVICE) + 1
            mask = mask.unsqueeze(0) <= lengths.to(DEVICE).unsqueeze(-1)

        scores = (D @ Q)
        scores = scores if mask is None else scores * mask.unsqueeze(-1)
        scores = scores.max(1)

        if explain:
            assert False, "TODO"

        return scores.values.sum(-1).cpu()


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos

    return output
