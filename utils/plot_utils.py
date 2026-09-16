import numpy as np
from typing import *
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
import seaborn as sns

def ARC_cmap() -> Dict:
    '''
    Creates a color palette for the plots based on the int -> color mappings.
    '''
    color_list = [color_recode(num) for num in range(8)]
    cmap = ListedColormap(color_list)
    norm = Normalize(vmin=0, vmax=7)
    return {'cmap': cmap, 'norm': norm}

def color_recode(int_encoding) -> str:
    int_to_color = {
        0: "black",
        1: "firebrick",
        2: "darkorange",
        3: "gold",
        4: "teal",
        5: "dodgerblue",
        6: "rebeccapurple",
        7: "hotpink",
    }
    return int_to_color[int_encoding]

def plot_matrix(
        matrix: np.array, 
        title: str, 
        ax=None, 
        size=14, 
        grid_color='grey',
        cmap_cfg=None
        ):
    fig, ax = plt.subplots() if ax is None else (None, ax)
    cmap_cfg = ARC_cmap() if cmap_cfg is None else cmap_cfg

    # Use the created Axes object to draw the image
    cax = ax.imshow(matrix, **cmap_cfg)
    ax.set_title(title, size=size)
    
    # Customize the axis
    midpoints = [x - 0.5 for x in range(1 + len(matrix))]
    ax.set_xticks(midpoints)
    ax.set_yticks(midpoints)
    ax.grid(True, which='both', color=grid_color, linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)

    if ax is not None:
        return ax  
    else:
        plt.show() 
 
def plot_item(
    arrays: List[np.array],
    title: str = None,
    labels: List[str] = ['A', 'B', 'C', 'D Concept', 'D Matrix'],
    return_fig: bool = False,
    empty_idx: List[int] = None,
    n_row: int = None,
    **kwargs
):  
    #assert len(arrays) in [4, 5, 6], f"Invalid number of arrays: {len(arrays)}"

    # Initialize the subplots
    if n_row is None:
        n_row = 2 if len(arrays) == 4 else 3
    fig, ax = plt.subplots(n_row, 3, figsize=(4, n_row+2), dpi=300)
    axs = ax.flatten()

    if empty_idx is None:
        empty_idx = {
            4: [1, 4],
            5: [1, 4, 5, 7],
            6: [1, 4, 5]
        }.get(len(arrays))

    # Plot the matrices
    arr_idx = 0
    for i, ax in enumerate(axs):
        if i not in empty_idx:
            matrix = arrays[arr_idx]
            plot_matrix(matrix=matrix, title=labels[arr_idx], ax=ax, **kwargs)
            arr_idx += 1
        else:
            ax.axis('off')

    # Add arrows
    arrow_properties = dict(arrowstyle="->", color="black", lw=1)
    axs[0].annotate('', xy=(2, 0.5), xytext=(1.5, 0.5), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrow_properties)
    axs[3].annotate('', xy=(2, 0.5), xytext=(1.5, 0.5), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrow_properties)

    if title:
        fig.suptitle(title, size=14)

    if return_fig:
        plt.close(fig) # Close the figure to avoid displaying
        return fig, ax  
    else:
        plt.show()

def get_percentage_ticks(df: pd.DataFrame, num=6) -> Tuple[np.array, List[str]]:
    max_val = df.to_numpy().max()
    max_val = (max_val // 0.1 + 1) * 0.1 # round up to the nearest 10%
    linespace = np.linspace(0, max_val, num=num)
    return linespace, [f'{x:.0%}' for x in linespace]

def plot_correlation_matrix(corr_mat, vmin=-1, vmax=1, annot=True, square=True, lower=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    mask = np.tril(corr_mat) == 0 if lower else None

    heatmap = sns.heatmap(
        corr_mat,
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        square=square,
        mask=mask,
        linewidths=0,  # Disable bounding boxes for all cells
        annot_kws={"size": 12},  # Increase annotation font size
        # italic
        cbar_kws={'label': 'r'},  # Add colorbar label
        ax=ax,  # Use provided Axes
    )

    # Increase the font size of the colorbar ticks
    heatmap.figure.axes[-1].tick_params(labelsize=14)
    heatmap.figure.axes[-1].yaxis.label.set_size(16)
    # vertical 
    heatmap.figure.axes[-1].yaxis.label.set_rotation(0)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14)

    # Add bounding boxes only for non-masked cells
    if lower:
        for i in range(corr_mat.shape[0]):
            for j in range(corr_mat.shape[1]):
                if not mask[i, j]:  # Only if the cell is not masked
                    rect = plt.Rectangle(
                        (j, i), 1, 1, fill=False, edgecolor='black', linewidth=0.5
                    )
                    ax.add_patch(rect)
    if ax is None:
        plt.show()


# from matplotlib import cm
# import torch

# token_attr = torch.tensor([-0.9765625,  0.1796875, -0.125    , -0.3671875, -0.0703125,
#        -0.3125   , -0.46875  , -0.0546875, -0.3125   , -0.703125 ,
#        -0.046875 ,  0.0546875, -0.78125  , -0.3359375, -0.5234375,
#        -1.046875 , -0.5546875, -0.65625  , -0.8984375, -0.3671875,
#        -0.1484375, -0.484375 , -0.140625 , -0.4453125, -0.4296875,
#        -0.03125  , -0.0703125])

# max_abs_attr_val = token_attr.abs().max().item()

# # Plot the heatmap
# data = token_attr.numpy()
# data = data.reshape((3,3,3))


# cmap_cfg = dict(cmap=cm.RdYlGn, norm=Normalize(vmin=-max_abs_attr_val, vmax=max_abs_attr_val))
# plot_item(data, cmap_cfg=cmap_cfg, empty_idx=[1, 4, 5], n_row=2)