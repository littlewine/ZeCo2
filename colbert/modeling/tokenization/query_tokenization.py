import torch

from transformers import BertTokenizerFast
from colbert.modeling.tokenization.utils import _split_into_batches


class QueryTokenizer():
    def __init__(self, query_maxlen):
        self.tok = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.query_maxlen = query_maxlen

        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tok.convert_tokens_to_ids('[unused0]')
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id

        assert self.Q_marker_token_id == 1 and self.mask_token_id == 103

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix + [self.mask_token] * (self.query_maxlen - (len(lst)+3)) for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix + [self.mask_token_id] * (self.query_maxlen - (len(lst)+3)) for lst in ids]

        return ids

    # def decode(self, input_ids): #TODO: delete or change
    #     # words = self.tok
    #     # return words
    #     # cut to self.query_maxlen
    #     pass

    def tensorize(self, batch_text, bsize=None, nr_expansion_tokens=10, debug=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [Q] marker
        batch_text = ['. ' + x for x in batch_text]

        obj = self.tok(batch_text, padding='max_length', truncation=True,
                       return_tensors='pt', max_length=self.query_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id

        pads = torch.cumsum(ids == 0, dim=1)

        expansion_mask = (0<pads) & (pads<=nr_expansion_tokens)
        pad_mask = pads>nr_expansion_tokens

        # ids[ids == 0] = self.mask_token_id # thats where all tokens where masked.

        print(f"Average # of [MASK] added per query (in query tensorization): {(expansion_mask.sum(1)*1.0).mean().item():.2f} ")
        print(f"Average # of [PAD] added per query (in query tensorization): {(pad_mask.sum(1)*1.0).mean().item():.1f} ")

        ids = ids + self.tok.mask_token_id * expansion_mask + self.tok.pad_token_id * pad_mask

        # attention mask is zero on the masked and padded tokens. How does this affect?
        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches

        return ids, mask
