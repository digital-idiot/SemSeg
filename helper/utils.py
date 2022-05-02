import warnings
import numpy as np
import pandas as pd
from seaborn import heatmap
from matplotlib import cm as mpl_cm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable


# noinspection PyUnresolvedReferences,SpellCheckingInspection
def plot_confusion_matrix(
        cm: np.ndarray,
        normed=True,
        annot=True,
        fig_size=10,
        dpi=300,
        font_size=22,
        fmt='.2f',
        cmap=mpl_cm.plasma,
        cbar=True,
        key=None
) -> Figure:
    """
    Makes pretty plot of confusion matrix
    Args:
        cm: confusion matrix (2D square matrix)
        normed: if confusion matrix is normalized
        annot: annotation flag
        fig_size: FIgure Size
        dpi: DPI of the figure
        font_size: font size
        fmt: float format signature
        cmap: color map
        cbar: color bar flag
        key: addition string for title
    Returns:
        Instance of matplotlib.figure.Figure containing the plot
    """
    assert (
            len(cm.shape) == 2
    ) and (
            len(set(cm.shape)) == 1
    ), (
        f"The confusion matrix has invalid shape: {cm.shape}" +
        "Expected a 2D square matrix"
    )
    cm = cm.astype(float)
    val_min = None
    val_max = None
    c = None
    if normed:
        val_min = 0
        val_max = 0
        c = 0.5
    n = cm.shape[0]
    labels = [f'$C_{str(i).zfill(len(str(n - 1)))}$' for i in range(n)]
    df = pd.DataFrame(data=cm, columns=labels, index=labels)
    # noinspection PyUnresolvedReferences
    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure(figsize=(fig_size, (fig_size * 1.055)), dpi=dpi)
    if key:
        fig.suptitle(f"Confusion Matrix: {key}")
    else:
        fig.suptitle(f"Confusion Matrix")

    ax = fig.gca()
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    ax.xaxis.tick_top()
    heatmap(
        data=df,
        annot=annot,
        vmin=val_min,
        vmax=val_max,
        center=c,
        fmt=fmt,
        cmap=cmap,
        cbar=cbar,
        ax=ax,
        cbar_ax=cax
    )
    ax.tick_params(axis="y", rotation=0)
    return fig


def format_float(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}"


def format_dict(
        dictionary: dict,
        key_prefix: str = '',
        key_suffix: str = '',
        delimiter: str = '_',
        case: str = None
) -> dict:
    formatted_dict = dict()
    key_prefix = f"{key_prefix}{delimiter}" if key_prefix else ''
    key_suffix = f"{delimiter}{key_suffix}" if key_suffix else ''
    for key, value in dictionary.items():
        k = f"{key_prefix}{key}{key_suffix}"
        if case == 'lower':
            k = k.lower()
        if case == 'upper':
            k = k.upper()
        else:
            if case:
                warnings.warn(f'Unknown case: {case}! Ignoring..')
        formatted_dict[f"{key_prefix}{key}{key_suffix}"] = value
    return formatted_dict
