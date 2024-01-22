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
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import umap
import math

from tensorflow import keras
from keras import layers
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib.colors import Colormap

import layervis.utils as lvutils


class ImageAtlas:
    """
    Class for visualising image atlases.

    Attributes:
        model: The Keras model to visualise
    """

    def __init__(self, model: keras.Model):
        """
        Initialises an ImageAtlas object for use with the given model.

        Args:
            model: The Keras model to visualise
        """
        self.model = model

    def get_unet_embedding(
        self,
        images: tf.Tensor | np.ndarray,
        layer: int | str | layers.Layer,
        mapper: umap.UMAP = None,
        n_neighbors: int = 15,
        n_components: int = 2,
        metric: str = "euclidean",
        n_epochs: int = 500,
        min_dist: float = 0.1,
        spread: float = 1,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Uses UMAP to create a lower-dimensional embedding of the outputs of the
        specified model layer when inputted with the given array of images. You can
        provide a UMAP object to perform the dimensionality reduction. Otherwise, this
        function acts as a partial wrapper for UMAP and will create a new UMAP object
        with a subset of common parameters.

        Args:
            images: Array of images. Must be 4-dimensional.
            layer: The layer whose outputs are to be reduced. Can be a layer index,
                layer name or layer object. Recommended to use dense layers.
            mapper: UMAP object, used to fit and transform the model outputs. If None,
                creates a new UMAP object with a subset of common parameters.
            n_neighbors: See umap.UMAP(). Number of neighbouring points for
                manifold approximation.
            n_components: See umap.UMAP(). Dimension of the embedding.
            metric: See umap.UMAP(). Distance metric.
            n_epochs: See umap.UMAP(). Number of training epochs to fit the mapper.
            min_dist: See umap.UMAP(). Minimum distance between points in the embedding.
            spread: See umap.UMAP(). Effective scale of the embedding.
            verbose: See umap.UMAP(). Verbosity of UMAP logging.
        """
        layer = lvutils.get_layer_object(model=self.model, layer_spec=layer)
        if mapper is None:
            mapper = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                metric=metric,
                n_epochs=n_epochs,
                min_dist=min_dist,
                spread=spread,
                verbose=verbose,
            )
        keras.backend.clear_session()
        output_model = keras.Model(inputs=self.model.inputs, outputs=layer.output)
        model_outputs = output_model.predict(images, verbose=0)
        model_outputs = model_outputs.reshape(len(model_outputs), -1)
        return mapper.fit_transform(model_outputs)

    def plot_image_atlas(
        self,
        images: np.ndarray,
        embedding: np.ndarray,
        nx: int,
        ny: int,
        max_image_dist: float,
        grid_pad: int = 0,
        labels: np.ndarray | list[int] = None,
        normalize_embedding: bool = True,
        label_cmap: Colormap = plt.cm.viridis,
        label_display: str = "border",
        border_thickness: float = 0.5,
        overlay_alpha: float = 0.1,
        figscale: float = 1,
        dpi: float = 100,
        cmap: str = "binary_r",
    ) -> Figure:
        """
        Plots an atlas of an imageset subject to some embedding.

        Args:
            images: Dataset of images to plot. Note there must
                at least be nx * ny images.
            embedding: 2D array of embeddings to visualise. These are coordinates
                of each image in a reduced, 2D plane,
                such as that returned by t-SNE, UMAP, etc.
            nx: Number of images to display in the x-direction
            ny: Number of images to display in the y-direction
            max_image_dist: An image is only plotted at a given point (x,y) if it is
                at most this far away from the closest point in the embedding.
                Note the distance metric is euclidean.
            grid_pad: Amount to pad/offset the minimum and maximum x and y values
                of each plot in the grid. Useful for adding whitespace. Default is 0.
            labels: List of integer image labels. Note labels are only displayed
                if this parameter is supplied. Default is None.
            normalize_embedding: Whether to normalize the provided embedding to
                the range [0,1]. Default is True.
            label_cmap: A matplotlib colormap object to colour the labels.
                Note that this is not a string! It must be a Colormap object,
                    e.g. plt.cm.viridis.
                If set to None (default), plt.cm.viridis will be used.
            label_display: One of 'border' or 'overlay'. If set to 'border', labels
                will be shown by adding a coloured border around each image in the grid.
                If set to 'overlay', the image will instead be overlaid with
                an opaque, coloured rectangle. Default is 'border'.
            border_thickness: Thickness of the borders. Has no effect if
                label_display is not set to 'border'. Default is 0.5.
            overlay_alpha: Transparency of the overlaid rectangle. Has no effect
                if label_display is not set to 'overlay'. Default is 0.2
            figscale: Figure scale multiplier. The figure size is nx*figscale
                by ny*figscale. Default is 1.
            dpi: Base resolution, passed to plt.figure. Default is 100.
            cmap: Colormap to use for the images. Default is 'binary_r'.

        Returns:
            The figure displaying the image atlas,
            where each image is an individual subplot.
        """
        # first normalize the embedding
        if normalize_embedding:
            embedding[:, 0] = (embedding[:, 0] - np.min(embedding[:, 0])) / np.ptp(
                embedding[:, 0]
            )
            embedding[:, 1] = (embedding[:, 1] - np.min(embedding[:, 1])) / np.ptp(
                embedding[:, 1]
            )
        # label normalisation is performed regardless of normalize_embedding
        show_labels = labels is not None
        if show_labels:
            labels = (labels - np.min(labels)) / np.ptp(labels)

        xvals = np.linspace(
            np.min(embedding[:, 0]) - grid_pad, np.max(embedding[:, 0]) + grid_pad, nx
        )
        yvals = np.linspace(
            np.max(embedding[:, 1]) + grid_pad, np.min(embedding[:, 1]) - grid_pad, ny
        )

        image_size = images[0].shape[0]
        fig = plt.figure(figsize=(figscale * nx, figscale * ny), dpi=dpi)
        fig.subplots_adjust(wspace=0, hspace=0)
        nn = 1  # subplot index
        for i in yvals:
            for j in xvals:
                ax = fig.add_subplot(ny, nx, nn)
                closest = min(
                    enumerate(embedding),
                    key=lambda x: math.dist(x[1], (j, i)),
                )
                distance = math.dist(closest[1], (j, i))
                if distance <= max_image_dist:
                    ax.imshow(images[closest[0], ...], cmap=cmap)
                    if show_labels:
                        if label_display == "border":
                            rect = patches.Rectangle(
                                (1, 1),
                                width=image_size - 2,
                                height=image_size - 2,
                                linewidth=border_thickness,
                                facecolor="none",
                                edgecolor=label_cmap(labels[closest[0]]),
                            )
                        elif label_display == "overlay":
                            rect = patches.Rectangle(
                                (0, 0),
                                width=image_size,
                                height=image_size,
                                linewidth=0,
                                edgecolor="none",
                                alpha=overlay_alpha,
                                facecolor=label_cmap(labels[closest[0]]),
                            )
                        ax.add_patch(rect)
                plt.axis("off")
                nn += 1
        return fig
