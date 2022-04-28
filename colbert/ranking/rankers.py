import torch

from functools import partial

from colbert.ranking.index_part import IndexPart
from colbert.ranking.faiss_index import FaissIndex
from colbert.utils.utils import flatten, zipstar


class Ranker():
    def __init__(self, args, inference, faiss_depth=1024):
        self.inference = inference
        self.faiss_depth = faiss_depth
        self.debug = args.debug

        if faiss_depth is not None:
            self.faiss_index = FaissIndex(args.index_path, args.faiss_index_path, args.nprobe, part_range=args.part_range)
            self.retrieve = partial(self.faiss_index.retrieve, self.faiss_depth)

        self.index = IndexPart(args.index_path, dim=inference.colbert.dim, part_range=args.part_range, verbose=True)

    def encode(self, queries, mask_method=None): #TODO: make mask here!
        assert type(queries) in [list, tuple], type(queries)

        Q = self.inference.queryFromText(queries, bsize=512 if len(queries) > 512 else None,
                                         mask_method=mask_method)

        return Q

    def rank(self, Q, pids=None, Q_mask=None):
        pids = self.retrieve(Q, verbose=False)[0] if pids is None else pids

        if self.debug: # check if all docs in index
            pids = [pid for pid in pids if self.index.pid_in_range(pid)]
            print(f"Calculating scores for {len(pids)} passages")

        assert type(pids) in [list, tuple], type(pids)
        assert Q.size(0) == 1, (len(pids), Q.size())
        assert all(type(pid) is int for pid in pids)

        scores = []
        if len(pids) > 0:
            Q = Q.permute(0, 2, 1)
            scores = self.index.rank(Q, pids, Q_mask) #TODO: pass arg here (level = -2)

            scores_sorter = torch.tensor(scores).sort(descending=True)
            pids, scores = torch.tensor(pids)[scores_sorter.indices].tolist(), scores_sorter.values.tolist()

        return pids, scores
