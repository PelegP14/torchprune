"""Module containing the ALDS net implementations."""


from ..base_clustering import (
    BaseClusterNet,
)
from ..base_decompose import (
    BaseDecomposeNet,
)
from  ..messi import (
    MessiClusterSparsifier,
    MessiSparsifier,
    MessiALDSSparsifier,
    MessiComparisonSparsifier
)
from  ..alds import (
    ALDSErrorIterativeAllocator
)
from .temp_allocator import (
    TempErrorAllocator,
    TempErrorIterativeAllocator,
    TempErrorIterativeAllocatorFrobenius,
    TempErrorIterativeAllocatorJOPT,
    TempErrorIterativeAllocatorPCwJOPT,
    TempErrorIterativeAllocatorUseBest,
    TempClusteringIterativeAllocator,
    TempErrorUseCoresetPC,
    TempErrorUseCoresetJOPT,
    TempErrorPracticalSpeedUp
)
from .temp_sparsifier import (
    TempSparsifier,
    TempMeanSparsifier,
    TempMaxClusteringSparsifier,
    TempMeanClusteringSparsifier,
    TempFrobeniusSparsifier,
    TempJOptSparsifier,
    TempPickBestSparsifier,
    TempSparsifierEfficient
)


class TempNet(BaseDecomposeNet):
    """ALDS pruning with k-SVD and iterative error-based allocation."""

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return TempErrorIterativeAllocator

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return MessiSparsifier

    @property
    def _k_split(self):
        """Get number of k splits in each layer."""
        return 3

class TempNetFrobenius(TempNet):
    """ALDS pruning with k-SVD and iterative error-based allocation."""

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return TempErrorIterativeAllocatorFrobenius

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return TempFrobeniusSparsifier

class TempNetALDSerror(BaseDecomposeNet):
    """ALDS pruning with k-SVD and iterative error-based allocation."""

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return ALDSErrorIterativeAllocator

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return MessiSparsifier

    @property
    def _k_split(self):
        """Get number of k splits in each layer."""
        return 3

class TempNetALDSerrorALDSsparsify(TempNetALDSerror):
    """ALDS pruning with k-SVD and iterative error-based allocation."""

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return MessiALDSSparsifier

class TempNetALDSerrorComparison(TempNetALDSerror):

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return MessiComparisonSparsifier

class TempNetALDSerrorMean(TempNetALDSerror):

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return TempMeanClusteringSparsifier

class TempNetALDSerrorMax(TempNetALDSerror):

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return TempMaxClusteringSparsifier

class TempNetALDSerrorFro(TempNetALDSerror):

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return TempFrobeniusSparsifier

class TempNetFrobeniusErrorMean(TempNetFrobenius):

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return TempMeanClusteringSparsifier

class TempNetFrobeniusErrorMax(TempNetFrobenius):

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return TempMaxClusteringSparsifier

class TempNetFrobeniusError(TempNetFrobenius):

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return MessiSparsifier

class ALDSerrorNoClusteringMax(TempNetALDSerror):

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return TempSparsifier

class TempNetALDSerrorJOpt(TempNetALDSerror):

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return TempJOptSparsifier

class TempNetJOpt(TempNet):

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return TempErrorIterativeAllocatorJOPT

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return TempJOptSparsifier


class TempNetPCwJOPT(TempNet):

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return TempErrorIterativeAllocatorPCwJOPT

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return TempMaxClusteringSparsifier

class TempNetUseBest(TempNetALDSerror):

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return TempPickBestSparsifier

class TempNetUniformSamplePC(TempNet):
    @property
    def _allocator_type(self):
        return TempErrorUseCoresetPC

class TempNetUniformSampleJOPT(TempNet):
    @property
    def _allocator_type(self):
        return TempErrorUseCoresetPC
class TempNetPracticalSpeedUpPC(TempNet):

    @property
    def _allocator_type(self):
        return TempErrorPracticalSpeedUp

class TempNetEfficient(BaseClusterNet):
    @property
    def _allocator_type(self):
        """Get allocator type."""
        return TempClusteringIterativeAllocator

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return TempSparsifierEfficient

    @property
    def _k_split(self):
        """Get number of k splits in each layer."""
        return 3

class TempNetPCEfficient(TempNetEfficient):

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return MessiClusterSparsifier


