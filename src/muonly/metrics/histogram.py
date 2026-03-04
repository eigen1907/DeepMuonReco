from functools import cached_property
from typing import Any
import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import aim
from hist import Hist
from hist.axis import Regular
from hist.storage import Double


__all__ = [
    "Histogram",
]


class Histogram(Metric):
    """
    TODO:
        - plot
        - density
    """

    histogram: Tensor

    def __init__(
        self,
        bins: int | None = None,
        range: tuple[float, float] | None = None,
        bin_edges: Any | None = None,
        clamp: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            bins: the number of bins
            range: range
            clamp: a boolean indicating whether entries are clamped or not before filling
        """
        super().__init__(**kwargs)
        if bin_edges is not None:
            bin_edges = torch.tensor(bin_edges)
            bins = len(bin_edges) - 1
            range = (float(bin_edges[0]), float(bin_edges[-1]))
            variable_bins = True
        elif bins is not None and range is not None:
            variable_bins = False
        else:
            raise ValueError(
                "Either `bin_edges` or both `bins` and `range` have to be provided."
            )

        if bins > 512:
            import warnings

            warnings.warn(
                f"You are trying to create a histogram with {bins} bins. "
                "This may lead to high memory consumption.",
            )

        self.bins = bins
        self.range = tuple(range)
        self.clamp = clamp
        self.bin_edges = bin_edges
        self.variable_bins = variable_bins

        self.add_state(
            name="histogram",
            default=torch.zeros((bins,)),
            dist_reduce_fx="sum",
        )

    @cached_property
    def histogram_kwargs(self) -> dict[str, Any]:
        kwargs = {}
        if self.variable_bins:
            kwargs["bin_edges"] = self.bin_edges
        else:
            kwargs["bins"] = self.bins
            kwargs["range"] = self.range
        kwargs["density"] = False
        return kwargs

    @cached_property
    def clamp_range(self) -> tuple[float, float]:
        min, max = self.range
        bin_half_width = (max - min) / self.bins
        return (min + bin_half_width, max - bin_half_width)

    def update(
        self,
        input: Tensor,
        weight: Tensor | None = None,
    ) -> None:
        """ """
        # NOTE:
        input = input.cpu()
        if weight is not None:
            weight = weight.cpu()

        # NOTE:
        input = input.float()
        if self.clamp:
            input = input.clamp(*self.clamp_range)

        histogram = torch.histogram(
            input=input,
            weight=weight,
            **self.histogram_kwargs,
        )

        self.histogram += histogram.hist.to(self.histogram.device)

    def compute(self) -> Tensor | aim.Distribution:
        return self.histogram

    @cached_property
    def edges(self) -> np.ndarray:
        return np.linspace(*self.range, num=(self.bins + 1))

    @property
    def lower_edges(self):
        return self.edges[:-1]

    @property
    def upper_edges(self):
        return self.edges[1:]

    @cached_property
    def bin_width(self) -> np.ndarray:
        return self.upper_edges - self.lower_edges

    @cached_property
    def bin_centers(self) -> np.ndarray:
        return (self.lower_edges + self.upper_edges) / 2

    def plot(
        self,
        ax: Axes | None = None,
        **kwargs: Any,
    ):
        data = self.histogram.cpu().numpy()

        axis = Regular(self.bins, self.range[0], self.range[1])

        h = Hist(axis, storage=Double(), data=data)

        ax = ax or plt.gca()
        h.plot(ax=ax, **kwargs)
