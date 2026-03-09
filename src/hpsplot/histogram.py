"""Histogram filling with error propagation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Histogram2DData:
    """Container for 2D histogram data.

    x is the "slicing" axis (e.g. tanLambda); y is the "fitted" axis (e.g. p).
    contents and errors have shape (n_x_bins, n_y_bins).
    """
    x_edges: np.ndarray
    y_edges: np.ndarray
    contents: np.ndarray
    errors: np.ndarray

    @property
    def x_centers(self):
        return 0.5 * (self.x_edges[:-1] + self.x_edges[1:])

    @property
    def y_centers(self):
        return 0.5 * (self.y_edges[:-1] + self.y_edges[1:])

    def __add__(self, other):
        """Merge two 2D histograms: sum contents, combine errors in quadrature."""
        return Histogram2DData(
            x_edges=self.x_edges.copy(),
            y_edges=self.y_edges.copy(),
            contents=self.contents + other.contents,
            errors=np.sqrt(self.errors ** 2 + other.errors ** 2),
        )


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

    def __add__(self, other):
        """Merge two histograms: sum contents, combine errors in quadrature."""
        return HistogramData(
            bin_edges=self.bin_edges.copy(),
            bin_contents=self.bin_contents + other.bin_contents,
            bin_errors=np.sqrt(self.bin_errors ** 2 + other.bin_errors ** 2),
        )

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


def fill_histogram_2d(x_values, y_values, weights, x_bins, x_min, x_max, y_bins, y_min, y_max):
    """Fill a 2D histogram with weights and sqrt(sum(w^2)) errors.

    Parameters
    ----------
    x_values : array-like
        Values for the X (slicing) axis.
    y_values : array-like
        Values for the Y (fitted) axis.
    weights : array-like
        Per-event weights.

    Returns
    -------
    Histogram2DData
        contents[i, j] = sum of weights in x-bin i, y-bin j.
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    x_range = (x_min, x_max)
    y_range = (y_min, y_max)

    contents, x_edges, y_edges = np.histogram2d(
        x_values, y_values,
        bins=[x_bins, y_bins],
        range=[x_range, y_range],
        weights=weights,
    )
    w2_sum, _, _ = np.histogram2d(
        x_values, y_values,
        bins=[x_bins, y_bins],
        range=[x_range, y_range],
        weights=weights ** 2,
    )
    errors = np.sqrt(w2_sum)

    return Histogram2DData(
        x_edges=x_edges,
        y_edges=y_edges,
        contents=contents,
        errors=errors,
    )
