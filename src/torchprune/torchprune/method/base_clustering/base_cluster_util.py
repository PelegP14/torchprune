"""Some utility functions for decomposition pytorch modules."""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np

from ..base_decompose.base_decompose_util import FoldScheme
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

    def __init__(self, arrangements, j_ranks, bias=True):
        """Initialize a list of linear layers each corresponding to a cluster."""
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(arrangement.shape[0], j_rank, bias) for arrangement, j_rank in zip(arrangements, j_ranks)]
        )
        self.arrangements = arrangements
        self.j_ranks = j_ranks
        self.use_bias = bias

    def forward(self, x):
        out = [self.layers[i](x[:,self.arrangements[i]]) for i in range(len(self.arrangements))]
        out = torch.cat(out,dim=1)
        return out

    def set_weights(self,weights,bias):
        if self.use_bias and bias is not None:
            start = 0
            for i in range(len(self.layers)):
                self.layers[i].bias = nn.Parameter(bias[start:start + self.j_ranks[i]])
                start += self.j_ranks[i]

        for i in range(len(self.layers)):
            self.layers[i].weight = nn.Parameter(weights[i])


class GroupedConv2d(nn.Module):
    """A Conv2d layer that supports channel clustering"""

    def __init__(self, arrangements, j_ranks, kernel_size, stride, padding, dilation, padding_mode, bias=True):
        """Initialize a list of conv layers each corresponding to a cluster."""
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Conv2d(arrangement.shape[0], j_rank, kernel_size, stride=stride, padding=padding
                       , dilation=dilation, bias=bias, padding_mode=padding_mode) for arrangement, j_rank in zip(arrangements,j_ranks)]
        )
        self.arrangements = arrangements
        self.j_ranks = j_ranks
        self.use_bias = bias
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode

    def forward(self, x):
        out = [self.layers[i](x[:,self.arrangements[i]]) for i in range(len(self.arrangements))]
        out = torch.cat(out,dim=1)
        return out

    def set_weights(self,weights,bias):
        if self.use_bias and bias is not None:
            start = 0
            for i in range(len(self.layers)):
                self.layers[i].bias = nn.Parameter(bias[start:start+self.j_ranks[i]])
                start += self.j_ranks[i]

        for i in range(len(self.layers)):
            self.layers[i].weight = nn.Parameter(weights[i])


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
            w_dec.shape[0] == out_f for (w_dec, _), _ in weights_hat
        ), "Output feature dimension doesn't agree"

        # check dimensions of each tensor
        w_dim = self._weight_dim
        assert all(
            w_enc.dim() == w_dim and w_dec.dim() == w_dim
            for (w_dec, w_enc), _ in weights_hat
        ), f"Enc/dec w's need to be {w_dim}-dimensional."

        # check that embedding dimensions agree
        assert all(
            w_dec.shape[1] == w_enc.shape[0] for (w_dec, w_enc), _ in weights_hat
        ), "Embedding dimension must agree!"

        # retrieve decomposed init kwargs
        kwargs_dec, kwargs_enc = scheme.decompose_kwargs(module_original)

        # set new decoding
        w_dec = torch.cat([w_dec for (w_dec, _), _ in weights_hat], dim=1)
        decoding = self._grouped_module_type(
            **self._get_init_kwargs((w_dec,), (torch.arange(w_dec.shape[1]),)),
            **kwargs_dec,
            bias=module_original.bias is not None
        )
        bias = module_original.bias
        decoding.set_weights((w_dec,), bias)

        # set new encoding
        w_enc = [w_enc for (_, w_enc), _ in weights_hat]
        arrangements = [arrangement for _, arrangement in weights_hat]
        encoding = self._grouped_module_type(
            **self._get_init_kwargs(w_enc, arrangements),
            **kwargs_enc,
            bias=False,
        )
        encoding.set_weights(w_enc, None)

        self.encoding = encoding
        self.decoding = decoding

        # in the case where some cluster gets j=0
        self.in_f = getattr(module_original, f"in_{self._feature_name}")

        # register arrangements as buffers
        self.register_buffer("k_splits", torch.tensor(k_splits))
        for i in range(k_splits):
            self.register_buffer(f"arrangement{i}", weights_hat[i][1])
        self.register_buffer("j_ranks", torch.tensor([w.shape[0] for w in w_enc]))
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
        # add back missing channels (that got j=0)
        full_range = np.arange(self.in_f)
        indices = np.where(np.isin(full_range, total_arrangement.numpy(), assume_unique=True, invert=True))[0]
        if indices.size != 0:
            indices = torch.tensor(indices)
            total_arrangement = torch.cat((total_arrangement, indices))
            pad = torch.zeros((weight_enc.shape[0],indices.shape[0]))
            weight_enc = torch.cat((weight_enc,pad),dim=1)
        orig_arrangement = torch.sort(total_arrangement)[1]
        weight_enc = weight_enc[:,orig_arrangement]

        weight_dec = torch.block_diag(
            *[
                scheme.fold(layer.weight)
                for layer in decoding.layers
            ]
        )

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
            module_original.bias = nn.Parameter(torch.cat([layer.bias for layer in decoding.layers]))
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
            "j_ranks": [weight.shape[0] for weight in weights],
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
            "j_ranks": [weight.shape[0] for weight in weights],
        }

    def _get_ungrouped_init_kwargs(self, weight):
        return {
            "in_features": weight.shape[1],
            "out_features": weight.shape[0],
        }
