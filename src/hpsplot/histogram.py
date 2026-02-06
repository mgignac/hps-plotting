"""Histogram filling with error propagation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class HistogramData:
    """Container for histogram data with bin edges, contents, and errors."""
    bin_edges: np.ndarray
    bin_contents: np.ndarray
    bin_errors: np.ndarray

    @property
    def bin_centers(self):
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    @property
    def bin_widths(self):
        return self.bin_edges[1:] - self.bin_edges[:-1]

    @property
    def integral(self):
        return np.sum(self.bin_contents)

    def normalized(self):
        """Return a copy normalized to unit area."""
        total = self.integral
        if total == 0:
            return HistogramData(
                bin_edges=self.bin_edges.copy(),
                bin_contents=self.bin_contents.copy(),
                bin_errors=self.bin_errors.copy(),
            )
        return HistogramData(
            bin_edges=self.bin_edges.copy(),
            bin_contents=self.bin_contents / total,
            bin_errors=self.bin_errors / total,
        )


def fill_histogram(values, weights, bins, x_min, x_max):
    """Fill a histogram with values and weights, propagating errors.

    Parameters
    ----------
    values : array-like
        Values to histogram.
    weights : array-like
        Per-event weights.
    bins : int
        Number of bins.
    x_min : float
        Lower edge of histogram range.
    x_max : float
        Upper edge of histogram range.

    Returns
    -------
    HistogramData
        Filled histogram with sqrt(sum(w^2)) errors.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    bin_contents, bin_edges = np.histogram(
        values, bins=bins, range=(x_min, x_max), weights=weights
    )

    # Error propagation: sqrt(sum(w^2)) per bin
    w2_sum, _ = np.histogram(
        values, bins=bins, range=(x_min, x_max), weights=weights ** 2
    )
    bin_errors = np.sqrt(w2_sum)

    return HistogramData(
        bin_edges=bin_edges,
        bin_contents=bin_contents,
        bin_errors=bin_errors,
    )
