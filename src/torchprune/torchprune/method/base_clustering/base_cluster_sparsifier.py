"""Module containing the sparsifier for low-rank compression."""

from abc import abstractmethod, ABC
import warnings
import torch

from ..base import BaseSparsifier
from .base_cluster_util import FoldScheme


class BaseClusterSparsifier(BaseSparsifier, ABC):
    """Base sparsifier for any type of cluster-decomposition-based compression."""

    def sparsify(self, rank_stats, arrangements=None):
        """Sparsify to the desired number of features (rank_j)."""
        # obtain current set of stats.
        rank_j, k_split, scheme_val = [int(stat.item()) for stat in rank_stats]
        scheme = FoldScheme(scheme_val)

        # making sure rank is at least 1
        rank_j = max(rank_j, 1)

        # recall kernel for later
        kernel_size = scheme.get_kernel(self._tensor)

        # now we sparsify
        # if the order is given we sparsify by it
        if arrangements is not None:
            weights_hat = self._recreate(
                self._tensor.detach(), rank_j, arrangements, scheme
            )
        else:
            # if we need to find the arrangements
            weights_hat = self._sparsify(
                self._tensor.detach(), rank_j, k_split, scheme
            )

        # finally we unfold the decomposed weights and keep the order of the input nodes
        weights_hat = [
            (scheme.unfold_decomposition(u, v, kernel_size), order)
            for u, v, order in weights_hat
        ]

        # check that embedding dimensions agree
        # sometimes factor module seems to mess up ...
        if any(u.shape[1] != v.shape[0] for u, v, order in weights_hat):
            warnings.warn("Skipping compression since embedding is wrong.")
            return []

        # check resulting number of weights in sparsification
        num_w_hats = sum(
            (u != 0.0).sum() + (v != 0.0).sum() for u, v, order in weights_hat
        )
        num_w = (self._tensor != 0.0).sum()

        # if it is not reducing number of parameters, set fake sparsification
        if num_w <= num_w_hats:
            return []

        return {"scheme": scheme, "weights_hat": weights_hat}

    @abstractmethod
    def _sparsify(self, tensor, rank_j, k_split, scheme):
        """Sparsify to the desired number of features (rank_j)."""

    def _recreate(self, tensor, rank_j, arrangements, scheme):
        # get the clusters as specified by the given arrangements
        tensor_clusters = [tensor[:, order] for order in arrangements]

        # fold tensor clusters
        tensor_clusters = [scheme.fold(chunk) for chunk in tensor_clusters]

        # compute clustered SVD
        svd_clusters = [torch.svd(chunk) for chunk in tensor_clusters]

        # change from A = U * D * V^T convention to A = U * V convention
        svd_clusters = [
            (u @ torch.sqrt(torch.diag(d)), torch.sqrt(torch.diag(d)) @ v.t())
            for u, d, v in svd_clusters
        ]

        # remove undesired parts of approximation for lower singular values
        return [(u[:, :rank_j], v[:rank_j, :], order) for (u, v), order in zip(svd_clusters, arrangements)]

class SequencialClusterSparsifier(BaseClusterSparsifier):
    """A cluster sparsifier that assumes the clusters are in sequence and have the same size."""

    def _sparsify(self, tensor, rank_j, k_split, scheme):
        """Sparsify to the desired number of features (rank_j)."""
        # get chunks from dim 1 (input dimension)
        tensor_chunks = torch.chunk(tensor, k_split, dim=1)
        orders = torch.chunk(torch.arange(tensor.shape[1]),k_split)

        # fold tensor chunks
        tensor_chunks = [scheme.fold(chunk) for chunk in tensor_chunks]

        # compute chunked SVD
        svd_chunks = [torch.svd(chunk) for chunk in tensor_chunks]

        # change from A = U * D * V^T convention to A = U * V convention
        svd_chunks = [
            (u @ torch.sqrt(torch.diag(d)), torch.sqrt(torch.diag(d)) @ v.t())
            for u, d, v in svd_chunks
        ]

        # remove undesired parts of approximation for lower singular values
        return [(u[:, :rank_j], v[:rank_j, :], order) for (u, v), order in zip(svd_chunks,orders)]
