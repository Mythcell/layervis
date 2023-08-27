"""Code for visualising the feature maps of convolutional neural networks.

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
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os

from .utils import get_layer_object, obtain_reasonable_figsize, save_image


class FeatureMaps:
    """
    Class for visualising the feature maps of convolutional neural networks.
    """

    def __init__(
            self, model: keras.Model,
            valid_layers: tuple[layers.Layer] = None):
        """
        Initialises a FeatureMaps object with the provided model and layers to extract.

        Args:
            model: The keras.Model object from which to extract the feature maps
            valid_layers: Tuple of Keras layers to extract features from. All other
                layers are ignored. If left as None (by default), it will be set to
                include Conv2D, MaxPool2D, AvgPool2D, Conv2DTranpose, SeparableConv2D,
                DepthwiseConv2D, ReLU and LeakyReLU. NB: The plotting functions will
                automatically ignore layers with invalid output shapes.
        """
        self.model = model
        if valid_layers is None:
            self.valid_layers = (
                layers.Conv2D, layers.SeparableConv2D, layers.Conv2DTranspose,
                layers.DepthwiseConv2D, layers.MaxPool2D, layers.AvgPool2D,
                layers.ReLU, layers.LeakyReLU
            )
        else:
            self.valid_layers = valid_layers


    def get_feature_maps(
            self, input_image: tf.Tensor | np.ndarray,
            layer: int | str) -> tf.Tensor | None:
        """
        Extracts feature maps for the given input with respect to the specified layer.

        Args:
            input_image: Single input with which to generate feature maps.
                Its shape must be the same as the model's input shape.
            layer: Name OR index of the layer to plot the feature maps of.

        Returns:
            Tensor of feature maps, or None if the feature maps are invalid.

        Raises:
            ValueError if the provided layer is not a valid layer.
        """
        layer = get_layer_object(self.model, layer)
        if not isinstance(layer, self.valid_layers):
            raise ValueError(f'Invalid layer, must be one of {self.valid_layers}')
        
        keras.backend.clear_session()
        feature_model = keras.Model(inputs=self.model.input, outputs=layer.output)
        feature_maps = feature_model.predict(input_image, verbose=0)
        if len(feature_maps.shape) != 4:
            print(
                f'Skipping layer {layer.name} '
                f'due to invalid output dimensions'
            )
            return None
        return feature_maps


    def plot_feature_map(
            self, input_image: tf.Tensor | np.ndarray, layer: str | int, 
            figscale: float = 1, dpi: float = 100, colormap: str = 'cividis',
            fig_aspect: str = 'uniform', fig_orient: str = 'h',
            include_title: bool = False,
            include_corner_axis: bool = True) -> Figure | None:
        """
        Plots and returns the feature maps for a given layer, with the given input.

        Args:
            input_image: Single input with which to generate feature maps.
                Its shape must be the same as the model's input shape.
            layer: Name OR index of the layer to plot the feature maps of.
            figscale: Figure size multiplier, passed to plt.figure. Default is 1.
            dpi: Base dpi, passed to plt.figure. Default is 100.
            colormap: Base colormap, passed to plt.figure. Default is 'cividis'
            fig_aspect: One of 'uniform' or 'wide', controls the aspect ratio
                of the figure. Use 'uniform' for squarish plots
                and 'wide' for rectangular. Default is 'uniform'.
            fig_orient One of 'h' or 'v'. If set to 'h', the number of columns
                will be >= the number of rows (vice versa if set to 'v'). Default is 'h'.
            include_title: Whether to add a title to the figure.
            include_corner_axis: Whether to display an axis on the
                bottom-left subplot.

        Returns:
            A matplotlib.figure.Figure object of the desired plot.

        Raises:
            ValueError if the desired layer is not one of the valid layers, if the
                value provided for layer is not of the correct type, and/or if the
                layer str/index does not exist.
        """
        feature_maps = self.get_feature_maps(
            input_image=input_image, layer=layer
        )
        if feature_maps is None:
            return None
        # determine figure dimensions
        nmaps = feature_maps.shape[-1]
        nrows, ncols = obtain_reasonable_figsize(
            nmaps, aspect_mode=fig_aspect, orient=fig_orient
        )
        fig = plt.figure(figsize=(figscale*ncols, figscale*nrows), dpi=dpi)
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        if include_title:
            fig.suptitle(f'{self.layer_names[i]}')
        for i in range(nmaps):
            fig.add_subplot(nrows, ncols, i+1)
            plt.imshow(feature_maps[0, ..., i], cmap=colormap)
            if include_corner_axis:
                # remove axes from all but the bottom-left subplot
                if i != (nrows - 1)*ncols:
                    plt.axis('off')
            else:
                plt.axis('off')
        return fig


    def plot_feature_maps(
            self, feature_input: tf.Tensor | np.ndarray, layers: list[int | str] = [],
            plot_input: bool = True, figscale: float = 1, dpi: float = 100,
            colormap: str = 'cividis', facecolor: str = 'white',
            fig_aspect: str = 'uniform', fig_orient: str = 'h',
            save_dir: str = 'feature_maps', save_format: str = 'png', prefix: str = '',
            suffix: str = '', include_titles: bool = False,
            include_corner_axis: bool = False, verbose: bool = True) -> None:
        """
        Plots and saves all feature maps for all convolutional and pooling
        layers with respect to the provided (single) test image. Ensure that
        the input has the same shape as the model input (including the batch size),
        e.g. (1, 28, 28, 1) for a single MNIST image, or (1, 784)
        for a flattened MNIST input.

        Args:
            feature_input: Single input with which to generate feature maps.
                Its shape must be the same as model's input shape for a batch size of 1.
            layers: List of layer names and/or indices of the layers to extract
                features from. If empty, the code will process all valid layers.
            plot_input: Whether to also plot the input image. Defaults to True.
            figscale: Base figure size multiplier, passed to plt.figure. Default is 1.
            Note the figure will have dimensions figscale*ncols, figscale*nrows,
                where ncols and nrows are automatically determined based on the number
                of feature channels to plot.
            dpi: Base resolution, passed to plt.figure. Default is 100.
            colormap: Base colormap, passed to plt.imshow. Default is 'cividis'.
            facecolor: Figure background color, passed to fig.savefig
            fig_aspect: One of 'uniform' or 'wide', controls the aspect ratio
                of the figure. Use 'uniform' for squarish plots
                and 'wide' for rectangular. Default is 'uniform'.
            fig_orient: One of 'h' or 'v'. If set to 'h', the number of columns
                will be >= the number of rows (vice versa if set to 'v'). Default is 'h'.
            save_dir: Directory in which to save images.
            save_format: Format to save images with
                e.g. 'png', 'pdf', 'jpg'
            prefix: Prefix to prepend to the file name of each plot,
                e.g. example1_, ex2-, test
            suffix: Suffix to append to the file name of each plot
                e.g. _example1, -ex2, test
            include_titles: Whether to title each figure with the name
                of the respective layer. Defaults to True.
            include_corner_axis: Whether to display axes on the
                bottom-left subplot of each figure. Defaults to False.
            verbose: Whether to print each filename as it is saved. Defaults to True.
        """
        if len(layers) == 0:
            layers = [
                i.name for i in self.model.layers if isinstance(i, self.valid_layers)
            ]
        else:
            layers = [get_layer_object(self.model, l).name for l in layers]

        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass
        
        if plot_input:
            save_image(
                image=feature_input[0, ...], figsize=figscale*6, dpi=dpi,
                colormap=colormap, facecolor=facecolor, save_dir=save_dir,
                filename=f'input{suffix}', save_format=save_format,
                figure_title=('input' if include_titles else None),
                include_axis=include_corner_axis, verbose=verbose
            )
        for layer in layers:
            fig = self.plot_feature_map(
                feature_input=feature_input, layer=layer, figscale=figscale, dpi=dpi,
                colormap=colormap, fig_aspect=fig_aspect, fig_orient=fig_orient,
                include_title=include_titles, include_corner_axis=include_corner_axis
            )
            if fig is None:
                continue
            file_name = os.path.join(
                f'{save_dir}',
                f'{prefix}{layer}{suffix}.{save_format}'
            )
            fig.savefig(
                file_name, format=save_format, facecolor=facecolor, bbox_inches='tight'
            )
            if verbose:
                print(f'Saved {file_name}')
            fig.clear()
            plt.close(fig)