"""Visualization routines for poset landscapes"""

from typing import Optional, Union

import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_fig
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colormaps


def plot_landscape(
    grid: np.ndarray,
    landscapes: np.ndarray,
    ax: Optional[mpl_axes.Axes] = None,
    legend: bool = False,
) -> Union[tuple[mpl_fig.Figure, mpl_axes.Axes], mpl_axes.Axes]:
    """Routine to plot a single set of persistence landscapes"""

    if ax is None:
        created_fig = True
        F, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        created_fig = False
    cmap = colormaps["hsv"]
    colors = cmap(np.linspace(0, 1.0, landscapes.shape[0], endpoint=False))
    legend_lines = []
    for i in range(landscapes.shape[0]):
        lines = ax.plot(grid, landscapes[i, ...].T, color=colors[i], label=f"H{i}")
        legend_lines.append(lines[0])
    if legend:
        ax.legend(handles=legend_lines)
    ax.set_yticks([])
    ax.set_aspect("equal")
    if created_fig:
        return F, ax
    else:
        return ax


def plot_all_landscapes(
    discretized_poset_landscapes: xr.DataArray,
    grid_layout: Optional[tuple[int, int]] = None,
    figsize: tuple[float, float] = (12.0, 12.0),
) -> tuple[mpl_fig.Figure, mpl_axes.Axes]:
    """Plot landscapes for each union on separate axes"""
    num_plots = discretized_poset_landscapes.sizes["union"]
    if grid_layout is None:
        grid_size = int(np.ceil(np.sqrt(num_plots)))
        F, axes = plt.subplots(grid_size, grid_size, figsize=figsize, sharey=True)
    else:
        F, axes = plt.subplots(
            grid_layout[0], grid_layout[1], figsize=figsize, sharey=True
        )
    for i, union in enumerate(discretized_poset_landscapes["union"]):
        this_ax = np.ravel(axes)[i]
        _ = plot_landscape(
            discretized_poset_landscapes["filt_vals"].values,
            discretized_poset_landscapes.sel({"union": union}).values,
            ax=this_ax,
            legend=True,
        )
        this_ax.set_title(union.item())
    return F, axes
