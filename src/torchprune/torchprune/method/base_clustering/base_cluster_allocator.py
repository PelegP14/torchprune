from abc import abstractmethod, ABC
import numpy as np
import torch

from .base_cluster_util import FoldScheme
from ..base_decompose import BaseDecomposeAllocator


class BaseClusterAllocator(BaseDecomposeAllocator):
    """The base allocator for decomposition-based compression."""

    @property
    def _folding_scheme_value(self):
        return FoldScheme.KERNEL_DECODE.value
