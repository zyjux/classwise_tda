"""Visualization routines for poset landscapes"""

from typing import Optional

import matplotlib.axes as mpl_axes
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colormaps


def plot_landscape(
    grid: np.ndarray,
    landscapes: np.ndarray,
    ax: Optional[mpl_axes.Axes] = None,
    legend: bool = False,
):
    """Routine to plot a single set of persistence landscapes"""

    cmap = colormaps["hsv"]
    colors = cmap(np.linspace(0, 1.0, landscapes.shape[0], endpoint=False))
    for i in range(landscapes.shape[0]):
        ax.plot(grid, landscapes[i, ...].T, color=colors[i], label=f"H{i}")
    if legend:
        ax.legend()
    return ax


def plot_all_landscapes(discretized_poset_landscapes: xr.DataArray):
    """Plot landscapes for each union on separate axes"""
