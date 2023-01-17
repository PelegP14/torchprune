"""Module containing the MessiNet implementations."""

from ..base_decompose import BaseDecomposeNet
from ..base_clustering import BaseClusterNet
from .messi_allocator import MessiAllocator, MessiClusterAllocator
from .messi_sparsifier import MessiSparsifier, MessiClusterSparsifier


class MessiNet(BaseDecomposeNet):
    """Projective clustering for any weight layer."""

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return MessiAllocator

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return MessiSparsifier

    @property
    def _k_split(self):
        """Get number of k splits in each layer."""
        return 3


class MessiNet5(MessiNet):
    """Projective clustering for any weight layer with k=5."""

    @property
    def _k_split(self):
        """Get number of k splits in each layer."""
        return 5


class MessiNetEfficient(BaseClusterNet):
    """Projective clustering for any weight layer."""

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return MessiClusterAllocator

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return MessiClusterSparsifier

    @property
    def _k_split(self):
        """Get number of k splits in each layer."""
        return 3
