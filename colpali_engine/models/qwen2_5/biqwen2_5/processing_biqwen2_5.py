from typing import List, Optional, Union

import torch
from transformers import BatchFeature

from colpali_engine.models.qwen2_5.colqwen2_5 import ColQwen2_5_Processor


class BiQwen2_5_Processor(ColQwen2_5_Processor):
    """
    Processor for ColQwen2.
    """

    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        """
        Process queries for ColQwen2.
        """
        if suffix is None:
            suffix = self.query_augmentation_token # we remove buffer tokens
        texts_query: List[str] = []

        for query in queries:
            query = self.query_prefix + query + suffix
            texts_query.append(query)

        batch_query = self(
            text=texts_query,
            return_tensors="pt",
            padding="longest",
        )

        return batch_query

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_single_vector(qs, ps, device=device)
