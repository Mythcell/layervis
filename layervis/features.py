"""Code for visualising the feature maps of convolutional neural networks.

    Copyright (C) 2022 Mitchell "Mythcell" Cavanagh

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

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from .utils import save_image, obtain_reasonable_figsize

class FeatureMaps:
    """
    Class for visualising the feature maps of convolutional neural networks.
    """

    def __init__(self, model, valid_layers=None):
        """
        Initialises a FeatureMaps object with the provided model and layers to extract.

        Args:
            model: The keras.Model object from which to extract the feature maps
            valid_layers: Tuple of Keras layers to extract features from. All other
                layers are ignored. By default, includes Conv2D, MaxPool2D,
                AvgPool2D, Conv2DTranpose, SeparableConv2D, DepthwiseConv2D, ReLU
                and LeakyReLU. Note the plotting functions will automatically ignore
                layers with invalid output shapes.
        """

        self.model = model
        if valid_layers is None:
            self.valid_layers = (
                layers.Conv2D, layers.SeparableConv2D, layers.Conv2DTranspose,
                layers.DepthwiseConv2D, layers.MaxPool2D, layers.AvgPool2D,
                layers.ReLU, layers.LeakyReLU)
        else:
            self.valid_layers = valid_layers


    def plot_feature_map(self, layer, feature_input, figscale=1, dpi=100, 
        colormap='cividis', fig_aspect='uniform', fig_orient='h',
        include_title=False, include_corner_axis=True):
        """
        Plots and returns the feature maps for a given layer, with the given input.

        Args:
            layer (str|int): Name OR index of the layer to plot the feature maps of.
            feature_input (np.array): Single input with which to generate feature maps.
                Its shape must be the same as the model's input shape.
            figscale (float): Figure size multiplier, passed to plt.figure. Default is 1.
            dpi (int): Base dpi, passed to plt.figure. Default is 100.
            colormap (str): Base colormap, passed to plt.figure. Default is 'cividis'
            fig_aspect (str): One of 'uniform' or 'wide', controls the aspect ratio
            of the figure. Use 'uniform' for squarish plots and 'wide' for rectangular.
            fig_orient (str): One of 'h' or 'v'. If set to 'h', the number of columns
                will be >= the number of rows (vice versa if set to 'v').
            include_title (bool): Whether to add a title to the figure.
            include_corner_axis (bool): Whether to display an axis on the
                bottom-left subplot.

        Returns:
            plt.Figure object

        Raises:
            ValueError if the desired layer is not one of the valid layers, if the
                value provided for layer is not of the correct type, and/or if the
                layer str/index does not exist.
        """
        # extract the desired layer either by name or by index
        if isinstance(layer,str):
            feature_layer = self.model.get_layer(layer)
        elif isinstance(layer,int):
            feature_layer = self.model.layers[layer]
        else:
            raise ValueError('Value for layer must be either a name or an index.')
        if not isinstance(feature_layer,self.valid_layers):
            raise ValueError(f'Invalid layer, must be one of {self.valid_layers}')

        # construct model and extract features
        feature_model = keras.Model(inputs=self.model.input,
            outputs=feature_layer.output,name='feature_model')
        feature_maps = feature_model.predict(feature_input)

        # skip layers with incorrect output dimensions. This is helpful when
        # visualising standalone activation layers by ignoring layers after the Flatten()
        if len(feature_maps.shape) != 4:
            print(f'Skipping layer {feature_layer.name}'
                f'due to invalid output dimensions')
            return None

        # determine figure dimensions
        nmaps = feature_maps.shape[-1]
        nrows, ncols = obtain_reasonable_figsize(
            nmaps, aspect_mode=fig_aspect, orient=fig_orient)
        fig = plt.figure(figsize=(figscale*ncols, figscale*nrows),dpi=dpi)
        fig.subplots_adjust(wspace=0.05,hspace=0.05)
        if include_title:
            fig.suptitle(f'{self.layer_names[i]}')

        # plot the feature maps
        for i in range(nmaps):
            fig.add_subplot(nrows, ncols, i+1)
            plt.imshow(feature_maps[0,...,i],cmap=colormap)
            if include_corner_axis:
                # remove axes from all but the bottom-left subplot
                if i != (nrows - 1)*ncols:
                    plt.axis('off')
            else:
                plt.axis('off')

        return fig


    def plot_feature_maps(self, feature_input, layers=None, plot_input=True, figscale=1,
        dpi=100, colormap='cividis', facecolor='white', fig_aspect='uniform',
        fig_orient='h', save_dir='feature_maps', save_format='png', prefix='', suffix='',
        include_titles=False, include_corner_axis=False, verbose=True):
        """
        Plots and saves all feature maps for all convolutional and pooling
        layers with respect to the provided (single) test image. Ensure that
        the input has the same shape as the model input (including the batch size),
        e.g. (1, 28, 28, 1) for a single MNIST image, or (1, 784)
        for a flattened MNIST input.

        Args:
            feature_input (np.array): Single input with which to generate feature maps.
                Its shape must be the same as model's input shape for a batch size of 1.
            layers (list): List of layer names and/or indices of the layers to extract
                features from. If set to None, the code will process all valid layers.
                Lists can contain both names and indices. Invalid layers are ignored.
            plot_input (bool): Whether to also plot the input image. Defaults to True.
            figscale (float): Base figure size multiplier, passed to plt.figure. Default
                is 1. Note the figure will have dimensions figscale*ncols, figscale*nrows,
                where ncols and nrows are automatically determined based on the number
                of feature channels to plot.
            dpi (int): Base resolution, passed to plt.figure
            colormap (str): Base colormap, passed to plt.imshow
            facecolor (str): Figure background color, passed to fig.savefig
            fig_aspect (str): One of 'uniform' or 'wide', controls the aspect ratio
            of the figure. Use 'uniform' for squarish plots and 'wide' for rectangular.
            fig_orient (str): One of 'h' or 'v'. If set to 'h', the number of columns
                will be >= the number of rows (vice versa if set to 'v').
            save_dir (str): Directory in which to save images.
            save_format (str): Format to save images with
                e.g. 'png', 'pdf', 'jpg'
            prefix (str): Prefix to prepend to the file name of each plot,
                e.g. example1_, ex2-, test
            suffix (str): Suffix to append to the file name of each plot
                e.g. _example1, -ex2, test
            include_titles (bool): Whether to title each figure with the name
                of the respective layer. Defaults to True.
            include_corner_axis (bool): Whether to display axes on the
                bottom-left subplot of each figure. Defaults to False.
            verbose (bool): Whether to print each filename as it is saved.
                Defaults to True.
        """
        # get the names of all valid layers in the model
        if layers is None:
            layers = [
                i.name for i in self.model.layers if isinstance(i, self.valid_layers)
            ]
        else:
            # convert indices to layer names
            layers = [
                (i if isinstance(i,str) else self.model.layers[i].name) for i in layers]
            # filter invalid layers
            layers = [i for i in layers if (
                isinstance(self.model.get_layer(i), self.valid_layers))]

        # create save directory if it does not yet exist
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass
        
        # save initial image if required to do so
        if plot_input:
            if len(feature_input.shape) != 4:
                print('W: Input is not an image. Skipping plot_input.')
            else:
                save_image(image=feature_input[0,...], figscale=figscale, dpi=dpi,
                    colormap=colormap, facecolor=facecolor, save_dir=save_dir,
                    filename=f'input{suffix}', save_format=save_format,
                    figure_title=('input' if include_titles else None),
                    include_axis=include_corner_axis, verbose=verbose)

        for layer in layers:
            assert isinstance(layer,str) # for debug
            fig = self.plot_feature_map(
                layer=layer, feature_input=feature_input, figscale=figscale, dpi=dpi,
                colormap=colormap, fig_aspect=fig_aspect, fig_orient=fig_orient,
                include_title=include_titles, include_corner_axis=include_corner_axis
            )
            # in case no figure was returned due to invalid layer output
            if fig is None:
                continue
            file_name = os.path.join(f'{save_dir}',
                f'{prefix}{layer}{suffix}.{save_format}')
            fig.savefig(file_name, format=save_format,
                facecolor=facecolor, bbox_inches='tight')
            if verbose:
                print(f'Saved {file_name}')
            
            # clear the figure
            fig.clear()
            plt.close(fig)
