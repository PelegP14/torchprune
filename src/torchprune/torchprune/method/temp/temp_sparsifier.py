import os.path

import torch
import numpy as np

from ..base_decompose import GroupedDecomposeSparsifier
from ..base_clustering import SequencialClusterSparsifier
from .temp_util import factor

class TempSparsifier(GroupedDecomposeSparsifier):
    """A Temp-based sparsifier for tensors (generalized to conv)."""

    @property
    def _em_steps(self):
        return 300

    @property
    def _num_init_em(self):
        return 10

    @property
    def can_improve_func(self):
        return factor.can_improve_max

    def _sparsify_with_temp(self, tensor, rank_j, k_split, scheme):
        """Sparsify to the desired number of features (rank_j) with Messi."""
        # get projective clustering
        # Convention:
        #  1. y = A^t * x
        #  2. A = UV
        #  3. y = V^T * (U^T * x)
        partition, list_u, list_v = factor.raw_messi(
            scheme.fold(tensor.detach()).t(),
            j=rank_j,
            k=k_split
        )
        u_stitched, v_stitched = factor.stitch(partition, list_u, list_v)

        # Convert it to pytorch/numpy convention ...
        # Pytorch convention:
        #  1. y = A * x
        #  2. y = U * (V * x)
        # weights_hat = [U, V]
        return [
            (
                v_stitched.t(),
                u_stitched.t(),
            )
        ]

    def _sparsify(self, tensor, rank_j, k_split, scheme):
        """Sparsify to the desired number of features (rank_j)."""
        # in some rare occassion Messi fails
        # --> then let's just use grouped projective sparsifier and stitch it
        print("sparsifying k={}, j={}".format(k_split,rank_j))
        try:
            return self._sparsify_with_temp(tensor, rank_j, k_split, scheme)
        except ValueError as e:
            weights_hat = super()._sparsify(tensor, rank_j, k_split, scheme)
            u_chunks, v_chunks = zip(*weights_hat)
            print(
                "Temp Sparsification failed."
                " Falling back to grouped decomposition sparsification"
            )
            return [(torch.cat(u_chunks, dim=1), torch.block_diag(*v_chunks))]

class TempMeanSparsifier(TempSparsifier):
    @property
    def can_improve_func(self):
        return factor.can_improve_mean

class TempMaxClusteringSparsifier(TempSparsifier):
    def _sparsify_with_temp(self, tensor, rank_j, k_split, scheme):
        """Sparsify to the desired number of features (rank_j) with Messi."""
        # get projective clustering
        # Convention:
        #  1. y = A^t * x
        #  2. A = UV
        #  3. y = V^T * (U^T * x)
        partition, list_u, list_v = factor.raw_messi_base(
            scheme.fold(tensor.detach()).t().cpu().numpy(),
            j=rank_j,
            k=k_split,
            steps=self._em_steps,
            NUM_INIT_FOR_EM=self._num_init_em,
            can_improve_func = self.can_improve_func
        )
        u_stitched, v_stitched = factor.stitch(partition, list_u, list_v)
        order = 'fro'
        base_messi_error = factor.base_messi_error(
            scheme.fold(tensor.detach()).t().cpu().numpy(),
            j=rank_j,
            k=k_split,
            partition=partition,
            order=order
        )
        smart_j_error = np.linalg.norm(scheme.fold(tensor.detach()).cpu().numpy()-(v_stitched.T @ u_stitched.T),ord=order)\
                        /np.linalg.norm(scheme.fold(tensor.detach()).cpu().numpy(),ord=order)
        # if base_messi_error < smart_j_error:
        #     for i in range(3):
        #         save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "smart_j_worse_{}.npy".format(i))
        #         if not os.path.exists(save_path):
        #             print("saving because worse")
        #             np.savez(
        #                 save_path,
        #                 A=scheme.fold(tensor.detach()).t().cpu().numpy(),
        #                 j=rank_j,
        #                 k=k_split,
        #                 partition=partition
        #             )
        #             break

        print("by changing j error changed from {} (static j={}) to {}".format(base_messi_error,rank_j,smart_j_error))
        if base_messi_error < smart_j_error:
            partition, list_u, list_v = factor.raw_messi_base(
                scheme.fold(tensor.detach()).t().cpu().numpy(),
                j=rank_j,
                k=k_split,
                steps=self._em_steps,
                NUM_INIT_FOR_EM=self._num_init_em,
                can_improve_func=self.can_improve_func
            )
        # print("alds bound was {}".format(factor.alds_bound(scheme.fold(tensor.detach()).cpu().numpy(),k_split,rank_j)))
        # Convert it to pytorch/numpy convention ...
        # Pytorch convention:
        #  1. y = A * x
        #  2. y = U * (V * x)
        # weights_hat = [U, V]
        return [
            (
                torch.tensor(v_stitched.T).float().to(tensor.device),
                torch.tensor(u_stitched.T).float().to(tensor.device),
            )
        ]

class TempMeanClusteringSparsifier(TempMaxClusteringSparsifier):
    @property
    def can_improve_func(self):
        return factor.can_improve_mean

class TempFrobeniusSparsifier(TempMaxClusteringSparsifier):
    @property
    def can_improve_func(self):
        return factor.can_improve_frobenius

class TempJOptSparsifier(TempSparsifier):
    def _sparsify_with_temp(self, tensor, rank_j, k_split, scheme):
        """Sparsify to the desired number of features (rank_j) with Messi."""
        # get projective clustering
        # Convention:
        #  1. y = A^t * x
        #  2. A = UV
        #  3. y = V^T * (U^T * x)
        partition, list_u, list_v = factor.raw_j_opt(
            scheme.fold(tensor.detach()).t(),
            j=rank_j,
            k=k_split,
            steps=self._em_steps,
            NUM_INIT_FOR_EM=self._num_init_em
        )
        u_stitched, v_stitched = factor.stitch(partition, list_u, list_v)

        # Convert it to pytorch/numpy convention ...
        # Pytorch convention:
        #  1. y = A * x
        #  2. y = U * (V * x)
        # weights_hat = [U, V]
        return [
            (
                v_stitched.t(),
                u_stitched.t(),
            )
        ]

class TempPickBestSparsifier(TempSparsifier):
    def _sparsify_with_temp(self, tensor, rank_j, k_split, scheme):
        """Sparsify to the desired number of features (rank_j) with Messi."""
        # get projective clustering
        # Convention:
        #  1. y = A^t * x
        #  2. A = UV
        #  3. y = V^T * (U^T * x)
        partition, list_u, list_v = factor.raw_best_pick(
            scheme.fold(tensor.detach()).t().cpu().numpy(),
            j=rank_j,
            k=k_split,
            steps=self._em_steps,
            NUM_INIT_FOR_EM=self._num_init_em
        )
        u_stitched, v_stitched = factor.stitch(partition, list_u, list_v)

        # Convert it to pytorch/numpy convention ...
        # Pytorch convention:
        #  1. y = A * x
        #  2. y = U * (V * x)
        # weights_hat = [U, V]
        return [
            (
                torch.tensor(v_stitched.T).float().to(tensor.device),
                torch.tensor(u_stitched.T).float().to(tensor.device),
            )
        ]

class TempSparsifierEfficient(SequencialClusterSparsifier):
    @property
    def _em_steps(self):
        return 300

    @property
    def _num_init_em(self):
        return 10

    @property
    def can_improve_func(self):
        return factor.can_improve_max

    def _sparsify_with_temp(self, tensor, rank_j, k_split, scheme):
        """Sparsify to the desired number of features (rank_j) with Messi."""
        # get projective clustering
        # Convention:
        #  1. y = A^t * x
        #  2. A = UV
        #  3. y = V^T * (U^T * x)
        partition, list_u, list_v, arrangements = factor.raw_j_opt_for_clustering(
            scheme.fold(tensor.detach()).t().cpu().numpy(),
            j=rank_j,
            k=k_split,
        )

        # Convert it to pytorch/numpy convention ...
        # Pytorch convention:
        #  1. y = A * x
        #  2. y = U * (V * x)
        # weights_hat = [U, V]
        return [
            (
                torch.tensor(v.T).float().to(tensor.device),
                torch.tensor(u.T).float().to(tensor.device),
                torch.tensor(arrangement).long().to(tensor.device)
            ) for u, v, arrangement in zip(list_u, list_v, arrangements)
        ]

    def _sparsify(self, tensor, rank_j, k_split, scheme):
        """Sparsify to the desired number of features (rank_j)."""
        # in some rare occassion Messi fails
        # --> then let's just use grouped projective sparsifier and stitch it
        print("sparsifying k={}, j={}".format(k_split, rank_j))
        try:
            return self._sparsify_with_temp(tensor, rank_j, k_split, scheme)
        except ValueError as e:
            weights_hat = super()._sparsify(tensor, rank_j, k_split, scheme)
            print(
                "Temp Sparsification failed."
                " Falling back to grouped decomposition sparsification"
            )
            return weights_hat