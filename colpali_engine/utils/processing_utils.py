from abc import ABC, abstractmethod
from collections import defaultdict
from email.policy import default
from typing import List, Optional, Tuple, Union

from matplotlib import pyplot as plt
import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature

from colpali_engine.utils.torch_utils import get_torch_device


class BaseVisualRetrieverProcessor(ABC):
    """
    Base class for visual retriever processors.
    """

    @abstractmethod
    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        pass

    @abstractmethod
    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        pass

    @abstractmethod
    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def score_single_vector(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the dot product score for the given single-vector query and passage embeddings.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        qs_stacked = torch.stack(qs).to(device)
        ps_stacked = torch.stack(ps).to(device)

        scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)
        assert scores.shape[0] == len(
            qs
        ), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @staticmethod
    def score_multi_vector(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
        dataset_name=None,
    ) -> torch.Tensor:
        """
        Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings (`ps`). For ColPali, a passage is the
        image of a document page.

        Because the embedding tensors are multi-vector and can thus have different shapes, they
        should be fed as:
        (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
        (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
            obtained by padding the list of tensors.

        Args:
            qs (`Union[torch.Tensor, List[torch.Tensor]`): Query embeddings.
            ps (`Union[torch.Tensor, List[torch.Tensor]`): Passage embeddings.
            batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
            device (`Union[str, torch.device]`, *optional*): Device to use for computation. If not
                provided, uses `get_torch_device("auto")`.

        Returns:
            `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
            tensor is saved on the "cpu" device.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")
        scores_list: List[torch.Tensor] = []
        counts = torch.zeros(1000)
        sums = torch.zeros(1000)
        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = qs[i : i + batch_size]
            qs_batch = [q.flip(dims=[0]) for q in qs_batch]
            qs_batch = torch.nn.utils.rnn.pad_sequence(
                qs_batch, batch_first=True, padding_value=0
            ).to(device)
            qs_batch = qs_batch.flip(dims=[1])
            for j in range(0, len(ps), batch_size):
                ps_batch = ps[j : j + batch_size]
                ps_batch = [p.flip(dims=[0]) for p in ps_batch]
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps_batch, batch_first=True, padding_value=0
                ).to(device)
                ps_batch = ps_batch.flip(dims=[1])
                scores = torch.einsum("bnd,csd->bcns", qs_batch, ps_batch)
                # scores_argmax = scores.argmax(dim=3).tolist()
                # scores_max = scores.max(dim=3)[0].tolist()
                score_sums = scores.abs().sum(dim=2).sum(dim=0).sum(dim=0)
                score_counts = scores.count_nonzero(dim=2).sum(dim=0).sum(dim=0)

                sums += torch.nn.functional.pad(
                    score_sums, (0, 1000 - score_sums.shape[0]), value=0.0
                ).cpu()
                counts += torch.nn.functional.pad(
                    score_counts, (0, 1000 - score_sums.shape[0]), value=0.0
                ).cpu()
                # nonzero_count = scores.reshape(
                #     (-1, scores.shape[2], scores.shape[3])
                # ).count_nonzero(dim=0)
                # print("nonzero_count", nonzero_count)
                # counts += torch.nn.functional.pad(
                #     nonzero_count,
                #     (0, 1000 - nonzero_count.shape[1], 0, 100 - nonzero_count.shape[0]),
                #     value=0.0,
                # ).cpu()
                # scores_sum = scores.abs().sum(0).sum(0)
                # print("sums", scores_sum)
                # sums += torch.nn.functional.pad(
                #     scores_sum,
                #     (0, 1000 - scores_sum.shape[1], 0, 100 - scores_sum.shape[0]),
                #     value=0.0,
                # ).cpu()

                scores_batch.append(scores.max(dim=3)[0].sum(dim=2))

            plt.figure()
            plt.plot(counts.tolist())
            plt.savefig(f"./all-counts-{dataset_name}.jpg")
            sums = sums / counts
            plt.figure()
            plt.plot(sums.tolist())
            plt.savefig(f"./all-averages-{dataset_name}.jpg")
            # sums = sums / counts
            # plt.figure()
            # plt.imshow(sums.tolist())
            # plt.savefig(f"./all-averages-{dataset_name}.png", format="png", dpi=1200)

            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(
            qs
        ), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @abstractmethod
    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int = 14,
        *args,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an
        image of size (height, width) with the given patch size.
        """
        pass
