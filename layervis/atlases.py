"""Code to plot an image atlas.

    Copyright (C) 2023 Mitchell "Mythcell" Cavanagh

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib.colors import Colormap

from .utils import euclidean_dist

def plot_image_atlas(
        images: np.ndarray, embedding: np.ndarray, nx: int, ny: int,
        max_image_dist: float, grid_pad: int = 0, labels: np.ndarray | list[int] = None,
        normalize_embedding: bool = True, label_colormap: Colormap = None,
        label_display: str = 'border', border_thickness: float = 0.5,
        overlay_alpha: float = 0.1, figscale: float = 1, dpi: float = 100) -> Figure:
    """
    Plots an atlas of an imageset subject to some embedding.

    Args:
        images: Dataset of images to plot. Note there must
            at least be nx * ny images.
        embedding: 2D array of embeddings to visualise, i.e. coordinates
            of each image in a reduced, 2D plane, such as returned by t-SNE, UMAP, etc.
        nx: Number of images to display in the x-direction
        ny: Number of images to display in the y-direction
        max_image_dist: An image is only plotted for a given point (x,y)
            if the closest point in the embedding is less than this distance.
            Note the distance metric is euclidean.
        grid_pad: Amount to pad/offset the minimum and maximum x and y values
            of each plot in the grid. Useful for adding whitespace. Default is 0.
        labels: List of integer image labels. Note labels are only displayed
            if this parameter is supplied. Default is None.
        normalize_embedding: Whether to normalize the provided embedding to
            the range [0,1]. Default is True.
        label_colormap: A matplotlib colormap object to colour the labels.
            Note that this is not a string! It must be an object, e.g. plt.cm.viridis.
            plt.cm.viridis will be used if no colormap was provided.
        label_display: One of 'border' or 'overlay'. If set to 'border', labels
            will be shown by adding a coloured, rectangular border around each image
            in the grid. If set to 'overlay', the image will instead be overlaid with
            an opaque, coloured rectangle. Default is 'border'.
        border_thickness: Thickness of the borders. Has no effect if
            label_display is not set to 'border'. Default is 0.5.
        overlay_alpha: Transparency of the overlaid rectangle. Has no effect
            if label_display is not set to 'overlay'. Default is 0.2
        figscale: Figure scale multiplier. The figure size is nx*figscale
            by ny*figscale. Default is 1.
        dpi: Base resolution, passed to plt.figure. Default is 100.
    """

    # first normalize the embedding
    if normalize_embedding:
        embedding[:,0] = (
            (embedding[:,0] - np.min(embedding[:,0])) / np.ptp(embedding[:,0])
        )
        embedding[:,1] = (
            (embedding[:,1] - np.min(embedding[:,1])) / np.ptp(embedding[:,1])
        )
    # normalize image labels (this is done regardless of the above normalize_embedding)
    show_labels = labels is not None
    if show_labels:
        labels = (labels - np.min(labels)) / np.ptp(labels)
        if label_colormap is None:
            label_colormap = plt.cm.viridis

    # determine grid values
    xvals = np.linspace(
        np.min(embedding[:,0]) - grid_pad, np.max(embedding[:,0]) + grid_pad, nx
    )
    yvals = np.linspace(
        np.max(embedding[:,1]) + grid_pad, np.min(embedding[:,1]) - grid_pad, ny
    )

    # get size of the image
    image_size = images[0].shape[0]

    fig = plt.figure(figsize=(figscale*nx, figscale*ny), dpi=dpi)
    fig.subplots_adjust(wspace=0, hspace=0)

    nn = 1 # subplot index
    for i in yvals:
        for j in xvals:
            ax = fig.add_subplot(ny, nx, nn)
            closest = min(
                enumerate(embedding), key=lambda x: euclidean_dist(x[1], (j,i))
            )
            distance = euclidean_dist(closest[1], (j,i))
            if distance <= max_image_dist:
                ax.imshow(images[closest[0],...])
                if show_labels:
                    if label_display == 'border':
                        rect = patches.Rectangle(
                            (1,1), width=image_size-2, height=image_size-2,
                            linewidth=border_thickness, facecolor='none',
                            edgecolor=label_colormap(labels[closest[0]])
                        )
                        ax.add_patch(rect)
                    elif label_display == 'overlay':
                        rect = patches.Rectangle(
                            (0,0), width=image_size, height=image_size,
                            linewidth=0, edgecolor='none', alpha=overlay_alpha,
                            facecolor=label_colormap(labels[closest[0]])
                        )
                        ax.add_patch(rect)
            plt.axis('off')
            nn += 1
    return fig