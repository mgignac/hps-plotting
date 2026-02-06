"""Selection mask evaluation."""

import logging

import numpy as np

from .config import RegionConfig
from .utils import safe_evaluate

logger = logging.getLogger(__name__)


class Region:
    """Wraps a RegionConfig and provides selection mask computation."""

    def __init__(self, config: RegionConfig):
        self.config = config
        self.name = config.name
        self.selection = config.selection
        self.label = config.label

    def apply(self, data):
        """Evaluate the selection expression and return a boolean mask.

        Parameters
        ----------
        data : dict
            Mapping of branch names to numpy arrays.

        Returns
        -------
        numpy.ndarray
            Boolean mask where True means the event passes selection.
        """
        logger.debug("Applying region '%s': %s", self.name, self.selection)
        mask = safe_evaluate(self.selection, data)
        mask = np.asarray(mask, dtype=bool)
        n_pass = np.sum(mask)
        n_total = len(mask)
        logger.debug("  %d / %d events pass (%.1f%%)",
                      n_pass, n_total, 100 * n_pass / n_total if n_total > 0 else 0)
        return mask
