"""Module containing the ALDS net implementations."""


from ..base_clustering import (
    BaseClusterNet,
)
from  ..messi import (
    MessiClusterSparsifier
)
from .temp_allocator import (
    TempErrorAllocator,
    TempErrorIterativeAllocator,
)


class TempNet(BaseClusterNet):
    """ALDS pruning with k-SVD and iterative error-based allocation."""

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return TempErrorIterativeAllocator

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return MessiClusterSparsifier

    @property
    def _k_split(self):
        """Get number of k splits in each layer."""
        return 3

