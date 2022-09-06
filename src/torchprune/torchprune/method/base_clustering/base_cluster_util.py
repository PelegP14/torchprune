"""Some utility functions for decomposition pytorch modules."""
from abc import ABC, abstractmethod
from enum import Enum, unique
from typing import final
import torch
import torch.nn as nn

from ...util import tensor


def get_attr(obj, names):
    """Get attribute from state dict naming convention."""
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    """Set attribute based on state dict naming convention."""
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


class GroupedLinear(nn.Module):
    """A linear layer that supports channel clustering."""

    def __init__(self, arrangements, j_rank, bias=True):
        """Initialize a list of linear layers each corresponding to a cluster."""
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(len(arrangement),j_rank,bias) for arrangement in arrangements]
        )
        self.arrangements = arrangements
        self.j_rank = j_rank
        self.use_bias = bias

    def forward(self, x):
        out = [self.layers[i](x[:,self.arrangements[i]]) for i in range(len(self.arrangements))]
        out = torch.cat(out,dim=1)
        return out

    def set_weights(self,weights,bias):
        if self.use_bias:
            biases = torch.chunk(bias, len(self.arrangements))
            for i in range(len(self.layers)):
                self.layers[i].bias = biases[i]

        for i in range(len(self.layers)):
            self.layers[i].weight = weights[i]


class GroupedConv2d(nn.Module):
    """A Conv2d layer that supports channel clustering"""

    def __init__(self, arrangements, j_rank, kernel_size, stride, padding, dilation, padding_mode, bias=True):
        """Initialize a list of conv layers each corresponding to a cluster."""
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Conv2d(len(arrangement), j_rank, kernel_size, stride=stride, padding=padding
                       , dilation=dilation, bias=bias, padding_mode=padding_mode) for arrangement in arrangements]
        )
        self.arrangements = arrangements
        self.j_rank = j_rank
        self.use_bias = bias

    def forward(self, x):
        out = [self.layers[i](x[:,self.arrangements[i]]) for i in range(len(self.arrangements))]
        out = torch.cat(out,dim=1)
        return out

    def set_weights(self,weights,bias):
        if self.use_bias:
            biases = torch.chunk(bias, len(self.arrangements))
            for i in range(len(self.layers)):
                self.layers[i].bias = biases[i]

        for i in range(len(self.layers)):
            self.layers[i].weight = weights[i]


class ProjectedModule(nn.Module, ABC):
    """An embedded module that contains encoding and decoding."""

    @property
    @abstractmethod
    def _ungrouped_module_type(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def _grouped_module_type(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def _feature_name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def _weight_dim(self):
        raise NotImplementedError

    @abstractmethod
    def _get_init_kwargs(self, weights, arrangements):
        """Get init kwargs for module that can be inferred from weights."""
        raise NotImplementedError

    @abstractmethod
    def _get_ungrouped_init_kwargs(self, weight):
        """Get init kwargs for ungrouped module."""
        raise NotImplementedError

    def __init__(self, module_original, weights_hat):
        """Initialize with the original conv module and the new weight."""
        super().__init__()

        # check if module_original is a vanilla module
        assert isinstance(
            module_original, self._ungrouped_module_type
        ), f"Need an unmodified {self._ungrouped_module_type} for init."

        # check that we have just one group to start with
        try:
            assert module_original.groups == 1
        except AttributeError:
            pass

        # get scheme and weights
        scheme = weights_hat["scheme"]
        weights_hat = weights_hat["weights_hat"]

        # check new number of clusters
        k_splits = len(weights_hat)

        # check output dimensions
        out_f = getattr(module_original, f"out_{self._feature_name}")
        assert all(
            w_dec.shape[0] == out_f for w_dec, _, _ in weights_hat
        ), "Output feature dimension doesn't agree"

        # check dimensions of each tensor
        w_dim = self._weight_dim
        assert all(
            w_enc.dim() == w_dim and w_dec.dim() == w_dim
            for w_dec, w_enc, _ in weights_hat
        ), f"Enc/dec w's need to be {w_dim}-dimensional."

        # check that embedding dimensions agree
        assert all(
            w_dec.shape[1] == w_enc.shape[0] for w_dec, w_enc, _ in weights_hat
        ), "Embedding dimension must agree!"

        # retrieve decomposed init kwargs
        kwargs_dec, kwargs_enc = scheme.decompose_kwargs(module_original)

        # set new decoding
        w_dec = torch.cat([w_dec for w_dec, _, _ in weights_hat], dim=1)
        decoding = self._grouped_module_type(
            **self._get_init_kwargs((w_dec,), (torch.arange(w_dec.shape[0]))), **kwargs_dec
        )
        bias = module_original.bias
        decoding.set_weights((w_dec,), bias)

        # set new encoding
        w_enc = [w_enc for _, w_enc, _ in weights_hat]
        arrangements = [arrangement for _, _, arrangement in weights_hat]
        encoding = self._grouped_module_type(
            **self._get_init_kwargs(w_enc, arrangements),
            **kwargs_enc,
            bias=False,
        )
        encoding.set_weights(w_enc, None)

        self.encoding = encoding
        self.decoding = decoding

        # register arrangements as buffers
        self.register_buffer("k_splits", torch.tensor(k_splits))
        for i in range(k_splits):
            self.register_buffer(f"arrangement{i}", weights_hat[i][2])

        # register scheme enum value as buffer
        self.register_buffer("scheme_value", torch.tensor(scheme.value))

    def forward(self, x):
        """Forward as encoding+decoding."""
        return self.decoding(self.encoding(x))

    def get_original_module(self):
        """Return an "unprojected" version of the module."""
        encoding = self.encoding
        decoding = self.decoding

        # get scheme
        scheme = FoldScheme(self.scheme_value.item())

        # get resulting encoding and decoding weights in right shape
        weight_enc = torch.block_diag(
            *[
                scheme.fold(layer.weight)
                for layer in encoding.layers
            ]
        )
        # get the total arrangement and rearrange the weights back
        total_arrangement = torch.cat(encoding.arrangements)
        orig_arrangement = torch.sort(total_arrangement)[1]
        weight_enc = weight_enc[:,orig_arrangement]

        weight_dec = scheme.fold(decoding.weight)

        # retrieve module kwargs from scheme and kernel_size
        kwargs_original = scheme.compose_kwargs(decoding, encoding)
        try:
            k_size = kwargs_original["kernel_size"]
        except KeyError:
            k_size = ()

        # build original weights
        w_original = scheme.unfold(weight_dec @ weight_enc, k_size)

        # build original module
        kwargs_weight = self._get_ungrouped_init_kwargs(w_original)
        module_original = self._ungrouped_module_type(
            **kwargs_weight, **kwargs_original
        )
        module_original.weight = nn.Parameter(w_original)
        if decoding.use_bias:
            module_original.bias = torch.cat([layer.bias for layer in decoding.layers])
        else:
            module_original.bias = None

        return module_original


class ProjectedConv2d(ProjectedModule):
    """An projected conv2d that contains encoding and decoding."""

    @property
    def _ungrouped_module_type(self):
        return nn.Conv2d

    @property
    def _grouped_module_type(self):
        return GroupedConv2d

    @property
    def _feature_name(self):
        return "channels"

    @property
    def _weight_dim(self):
        return 4

    def _get_init_kwargs(self, weights, arrangements):
        return {
            "arrangements": arrangements,
            "j_rank": weights[0].shape[0],
        }

    def _get_ungrouped_init_kwargs(self, weight):
        return {
            "in_channels": weight.shape[1],
            "out_channels": weight.shape[0],
        }


class ProjectedLinear(ProjectedModule):
    """A projected nn.Linear that contains encoding and decoding."""

    @property
    def _ungrouped_module_type(self):
        return nn.Linear

    @property
    def _grouped_module_type(self):
        return GroupedLinear

    @property
    def _feature_name(self):
        return "features"

    @property
    def _weight_dim(self):
        return 2

    def _get_init_kwargs(self, weights, arrangements):
        return {
            "arrangements": arrangements,
            "j_rank": weights[0].shape[0],
        }

    def _get_ungrouped_init_kwargs(self, weight):
        return {
            "in_features": weight.shape[1],
            "out_features": weight.shape[0],
        }


@unique
class FoldScheme(Enum):
    """An enumeration of possible folding schemes."""

    KERNEL_ENCODE = 0
    KERNEL_SPLIT1 = 1
    KERNEL_SPLIT2 = 2
    KERNEL_DECODE = 3

    @property
    def _scheme(self):
        """Return fold scheme corresponding to current name."""
        schemes_lookup = [_EncoderFold, _Split1Fold, _Split2Fold, _DecoderFold]
        return schemes_lookup[int(self.value)]

    @classmethod
    def get_default(cls):
        """Return default folding scheme."""
        return cls.KERNEL_ENCODE

    @final
    def get_kernel(self, tnsr):
        """Get the kernel size from the provided tensor."""
        if tnsr.dim() == 2:
            return tuple()
        elif tnsr.dim() == 4:
            return tnsr.shape[2:]
        else:
            raise NotImplementedError(
                "Only 2 or 4-dimensional tensors supported!"
            )

    @final
    def fold(self, tnsr):
        """Fold the tensor into a 2d-matrix."""
        if tnsr.dim() == 2:
            return self._scheme.fold(tnsr[:, :, None, None])
        elif tnsr.dim() == 4:
            return self._scheme.fold(tnsr)
        else:
            raise NotImplementedError(
                "Only 2 or 4-dimensional tensors supported!"
            )

    @final
    def unfold(self, tnsr, kernel_size):
        """Unfold the tensor into a 4d or 2d-tensor depending on kernel."""
        if len(kernel_size) == 0:
            return self._scheme.unfold(tnsr, (1, 1))[:, :, 0, 0]
        elif len(kernel_size) == 2:
            return self._scheme.unfold(tnsr, kernel_size)
        else:
            raise NotImplementedError(
                "Only 2 or 4-dimensional tensors supported!"
            )

    @final
    def unfold_decomposition(self, w_dec, w_enc, kernel_size):
        """Unfold the tensors into a 4d or 2d-tensor depending on kernel.

        This will take care of unfolding both the encoder and decoder part with
        the appropriate kernel size.
        """
        if len(kernel_size) == 0:
            is_2d = True
            kernel_size = (1, 1)
        elif len(kernel_size) == 2:
            is_2d = False
        else:
            raise NotImplementedError(
                "Only 2 or 4-dimensional tensors supported!"
            )
        k_dec, k_enc = self._scheme.decompose_kernel(kernel_size)

        w_dec_unf = self._scheme.unfold(w_dec, k_dec)
        w_enc_unf = self._scheme.unfold(w_enc, k_enc)

        if is_2d:
            w_dec_unf = w_dec_unf[:, :, 0, 0]
            w_enc_unf = w_enc_unf[:, :, 0, 0]

        return w_dec_unf, w_enc_unf

    @property
    def _decomposable_kwargs(self):
        """Get dict of decomposable kwargs for nn.Module initialization.

        Value indicates default value that should be used.
        """
        return {
            "kernel_size": 1,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "padding_mode": None,
        }

    @final
    def decompose_kwargs(self, module):
        """Get decomposed initialization kwargs for this module."""
        kwargs_dec = {}
        kwargs_enc = {}

        for attr, default in self._decomposable_kwargs.items():
            try:
                val_attr = getattr(module, attr)
            except AttributeError:
                continue
            if default is None:
                val_dec, val_enc = val_attr, val_attr
            else:
                val_dec, val_enc = self._scheme.decompose_kernel(
                    val_attr, size_default=default
                )
            kwargs_dec[attr] = val_dec
            kwargs_enc[attr] = val_enc

        return kwargs_dec, kwargs_enc

    @final
    def compose_kwargs(self, mod_dec, mod_enc):
        """Compose initialization kwargs from decoder/encoder modules."""
        kwargs = {}

        for attr, default in self._decomposable_kwargs.items():
            try:
                val_dec = getattr(mod_dec, attr)
                val_enc = getattr(mod_enc, attr)
            except AttributeError:
                continue
            if default is None:
                val_attr = val_dec
            else:
                val_attr = self._scheme.compose_kernel(val_dec, val_enc)
            kwargs[attr] = val_attr

        return kwargs

    @final
    def get_decomposed_kernel_sizes(self, kernel_size):
        """Compute the decomposed kernel from the actual kernel."""
        if len(kernel_size) == 0:
            kernel_size = (1, 1)
        k_dec, k_enc = self._scheme.decompose_kernel(kernel_size)
        return k_dec[0] * k_dec[1], k_enc[0] * k_enc[1]


class _BaseFold(ABC):
    """An abstract class representing the different folding schemes.

    Depending on the folding scheme we can represent the 4d-tensor of a Conv2d
    as various types of matrices.
    """

    @staticmethod
    @abstractmethod
    def fold(tnsr):
        """Fold according to the actual folding operation."""

    @staticmethod
    @abstractmethod
    def unfold(tnsr, kernel_size):
        """Unfold according to the actual folding operation and kernel size."""

    @staticmethod
    @abstractmethod
    def decompose_kernel(kernel_size, size_default=1):
        """Get decomposed kernel sizes, i.e., for decoding and encoding."""

    @staticmethod
    @abstractmethod
    def compose_kernel(kernel_dec, kernel_enc):
        """Get composed kernel from decoding and encoding."""


class _EncoderFold(_BaseFold):
    """Folding scheme with kernel on the encoder side.

    f  ... # filters
    c  ... # channels
    k1 ... # kernel dimension 1 (height)
    k2 ... # kernel dimension 2 (width)

    Folding:
    f x c x k1 x k2 --> f x (c * k1 * k2)

    Low-rank decomposition with rank j:
    f x c x k1 x k2 --> f x j x 1 x 1, j x c x k1 x k2
    """

    @staticmethod
    def fold(tnsr):
        """Fold according to the actual folding operation."""
        return tnsr.reshape(tnsr.shape[0], -1)

    @staticmethod
    def unfold(tnsr, kernel_size):
        """Unfold according to the actual folding operation and kernel size."""
        return tnsr.reshape(tnsr.shape[0], -1, *kernel_size)

    @staticmethod
    def decompose_kernel(kernel_size, size_default=1):
        """Get decomposed kernel sizes, i.e., for decoding and encoding."""
        k_dec = (size_default, size_default)
        k_enc = (kernel_size[0], kernel_size[1])
        return k_dec, k_enc

    @staticmethod
    def compose_kernel(kernel_dec, kernel_enc):
        """Get composed kernel from decoding and encoding."""
        return (kernel_enc[0], kernel_enc[1])


class _Split1Fold(_BaseFold):
    """Folding scheme with kernel split on both sides.

    f  ... # filters
    c  ... # channels
    k1 ... # kernel dimension 1 (height)
    k2 ... # kernel dimension 2 (width)

    Folding:
    f x c x k1 x k2 --> (f * k1) x (k2 * c)

    Low-rank decomposition with rank j:
    f x c x k1 x k2 --> f x j x k1 x 1, j x c x 1 x k2
    """

    @staticmethod
    def fold(tnsr):
        """Fold according to the actual folding operation."""
        # f x k1 x k2 x c
        tnsr = tnsr.movedim(1, 3)
        #  (f * k1) x (k2 * c)
        tnsr = tnsr.reshape(tnsr.shape[0] * tnsr.shape[1], -1)
        return tnsr

    @staticmethod
    def unfold(tnsr, kernel_size):
        """Unfold according to the actual folding operation and kernel size."""
        num_f = tnsr.shape[0] // kernel_size[0]
        # f x k1 x k2 x c
        tnsr = tnsr.reshape(num_f, *kernel_size, -1)
        # f x c x k1 x k2
        tnsr = tnsr.movedim(3, 1)
        return tnsr

    @staticmethod
    def decompose_kernel(kernel_size, size_default=1):
        """Get decomposed kernel sizes, i.e., for decoding and encoding."""
        k_dec = (kernel_size[0], size_default)
        k_enc = (size_default, kernel_size[1])
        return k_dec, k_enc

    @staticmethod
    def compose_kernel(kernel_dec, kernel_enc):
        """Get composed kernel from decoding and encoding."""
        return (kernel_dec[0], kernel_enc[1])


class _Split2Fold(_BaseFold):
    """Folding scheme with kernel split on both sides.

    f  ... # filters
    c  ... # channels
    k1 ... # kernel dimension 1 (height)
    k2 ... # kernel dimension 2 (width)

    Folding:
    f x c x k1 x k2 --> (f * k2) x (c * k1)

    Low-rank decomposition with rank j:
    f x c x k1 x k2 --> f x j x 1 x k2, j x c x k1 x 1
    """

    @staticmethod
    def fold(tnsr):
        """Fold according to the actual folding operation."""
        # f x k2 x k1 x c
        tnsr = tnsr.transpose(1, 3)
        #  (f * k2) x (k1 * c)
        tnsr = tnsr.reshape(tnsr.shape[0] * tnsr.shape[1], -1)
        return tnsr

    @staticmethod
    def unfold(tnsr, kernel_size):
        """Unfold according to the actual folding operation and kernel size."""
        num_f = tnsr.shape[0] // kernel_size[1]
        # f x k2 x k1 x c
        tnsr = tnsr.reshape(num_f, kernel_size[1], kernel_size[0], -1)
        # f x c x k1 x k2
        tnsr = tnsr.transpose(1, 3)
        return tnsr

    @staticmethod
    def decompose_kernel(kernel_size, size_default=1):
        """Get decomposed kernel sizes, i.e., for decoding and encoding."""
        k_dec = (size_default, kernel_size[1])
        k_enc = (kernel_size[0], size_default)
        return k_dec, k_enc

    @staticmethod
    def compose_kernel(kernel_dec, kernel_enc):
        """Get composed kernel from decoding and encoding."""
        return (kernel_enc[0], kernel_dec[1])


class _DecoderFold(_BaseFold):
    """Folding scheme with kernel on the decoder side.

    f  ... # filters
    c  ... # channels
    k1 ... # kernel dimension 1 (height)
    k2 ... # kernel dimension 2 (width)

    Folding:
    f x c x k1 x k2 --> (f * k1 * k2) x c

    Low-rank decomposition with rank j:
    f x c x k1 x k2 -->  f x j x k1 x k2, j x c x 1 x 1
    """

    @staticmethod
    def fold(tnsr):
        """Fold according to the actual folding operation."""
        # f x k1 x k2 x c
        tnsr = tnsr.movedim(1, 3)
        # (f * k1 * k2) x c
        tnsr = tnsr.reshape(-1, tnsr.shape[3])
        return tnsr

    @staticmethod
    def unfold(tnsr, kernel_size):
        """Unfold according to the actual folding operation and kernel size."""
        # f x k1 x k2 x c
        tnsr = tnsr.reshape(-1, *kernel_size, tnsr.shape[1])
        # f x c x k1 x k2
        tnsr = tnsr.movedim(3, 1)
        return tnsr

    @staticmethod
    def decompose_kernel(kernel_size, size_default=1):
        """Get decomposed kernel sizes, i.e., for decoding and encoding."""
        k_dec = (kernel_size[0], kernel_size[1])
        k_enc = (size_default, size_default)
        return k_dec, k_enc

    @staticmethod
    def compose_kernel(kernel_dec, kernel_enc):
        """Get composed kernel from decoding and encoding."""
        return (kernel_dec[0], kernel_dec[1])
