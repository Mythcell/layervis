"""Code for visualising filter weights and filter patterns of CNNs

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
from skimage.filters import gaussian
import os

from .utils import get_blank_image, obtain_reasonable_figsize

class FilterWeights:
    """
    Class for visualising the filter weights of a convolutional neural network.

    Attributes:
        model: A keras.Model object from which to visualise filters
        valid_layers: Tuple of keras.Layer objects. By default, the filter
            visualisation is enabled for Conv2D and Conv2DTranpose layers.
    """
    
    def __init__(self, model: keras.Model):
        self.model = model
        self.valid_layers = (layers.Conv2D, layers.Conv2DTranspose)
        

    def plot_filter_weights(
            self, layer: int | str, input_map_index: int = 0,
            normalize_weights: bool = True, figscale: float = 1, dpi: float = 100,
            colormap: str = 'cividis', fig_aspect: str = 'uniform',
            fig_orient: str = 'h', include_title: bool = False) -> Figure:
        """
        Plots the filter weights for a given layer.

        Args:
            layer: Name of the layer to extract filters from.
                The layer itself must be a Conv2D layer.
            input_map_index:
                The input feature map to extract filters from
                e.g. First layer typically has 1 (mono) or 3 (RGB) inputs
            normalize_weights: Whether to normalize the filter weights before
                visualisation. Default is True.
            figscale: Base figure scale multiplier, passed to plt.figure.
            dpi: Base dpi, passed to plt.figure
            colormap: Base colormap, passed to plt.figure.
            fig_aspect: One of 'uniform' or 'wide'. Use 'uniform' for
                squarish plots and 'wide' for more rectangular plots.
            fig_orient: One of 'h' or 'v'. If set to 'h', the number of columns
                will be >= the number of rows (vice versa if set to 'v').
            include_title: Whether to title the figure with the
                name of the layer.
        
        Returns:
            A single matplotlib.figure.Figure object.

        Raises:
            ValueError if attempting to call with a non-convolutional layer
        """
        if isinstance(layer, int):
            layer = self.model.layers[layer]
        elif isinstance(layer, str):
            layer = self.model.get_layer(layer)
        else:
            raise ValueError(f'Invalid layer {layer}')
        if not isinstance(layer, self.valid_layers):
            raise ValueError(f'Invalid layer, must be one of {self.valid_layers}')

        filters = layer.get_weights()[0]

        # extract the filters for the given input_map_index
        # only if the filter input isn't an RGB image
        is_rgb = filters.shape[-2] == 3
        if not is_rgb:
            filters = filters[..., input_map_index, :]
        # normalise to [0,1]
        if normalize_weights:
            filters = (
                (filters - np.min(filters))
                / (np.ptp(filters) + keras.backend.epsilon())
            )
        # convert to [0,255]
        filters = np.uint8(filters*255)

        num_filters = filters.shape[-1]
        nrows, ncols = obtain_reasonable_figsize(
            num_filters, aspect_mode=fig_aspect, orient=fig_orient
        )
        fig = plt.figure(figsize=(figscale*ncols, figscale*nrows), dpi=dpi)
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        if include_title:
            fig.suptitle(layer.name)

        for i in range(num_filters):
            fig.add_subplot(nrows, ncols, i+1)
            if is_rgb:
                plt.imshow(filters[..., i])
            else: # colormap should only apply to non-rgb filters
                plt.imshow(filters[..., i], cmap=colormap)
            plt.axis('off')
        
        return fig


    def plot_all_weights(
            self, input_map_index: int = 0, normalize_weights: bool = True,
            save_dir: str = 'filter_weights', save_format: str = 'png',
            figscale: float = 1, dpi: float = 100, colormap: str = 'cividis',
            fig_aspect: str = 'uniform', fig_orient: str = 'h',
            facecolor: str = 'white', include_titles: bool = False, prefix: str = '',
            suffix: str = '', verbose: bool = True) -> None:
        """
        Plots the filter weights for all valid layers.

        Args:
            input_map_index: The input feature map to extract filters from
                e.g. First layer typically has 1 (mono) or 3 (RGB) inputs
            normalize_weights: Whether to normalize the filter weights before
                visualisation. Default is True.
            save_dir: Directory in which to save the images
            save_format: Format with which to save the images
            figscale: Base figure size, passed to plt.figure
                Note the dimensions of the figure are automatically determined,
                figscale is merely a multiplier with which to scale.
            dpi: Base resolution, passed to plt.figure
            colormap: Base colormap, passed to plt.figure
            fig_aspect: One of 'uniform' or 'wide'. Use 'uniform' for
                squarish plots and 'wide' for rectangular.
            fig_orient: One of 'h' or 'v'. If set to 'h', the number of columns
                will be >= the number of rows (vice versa if set to 'v').
            facecolor: Background color, passed to fig.savefig.
            include_titles: Whether to title each figure with the name
                of each layer. Defaults to False.
            prefix: Optional prefix to add to the filename of each image.
            suffix: Optional suffix to add to the filename of each image.
            verbose: Whether to print the name of each file as it is
                saved. Defaults to True.
        """
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass

        conv_layers = [
            i.name for i in self.model.layers if isinstance(i,self.valid_layers)
        ]
        for layer in conv_layers:
            fig = self.plot_filter_weights(
                layer, input_map_index=input_map_index,
                normalize_weights=normalize_weights, figscale=figscale, dpi=dpi,
                colormap=colormap, fig_aspect=fig_aspect, fig_orient=fig_orient,
                include_title=include_titles
            )
            filename = os.path.join(
                save_dir, f'{prefix}{layer}{suffix}.{save_format}'
            )
            if verbose:
                print(f'Saving to {filename}')
            fig.savefig(
                filename, format=save_format,
                facecolor=facecolor, bbox_inches='tight'
            )
            
            fig.clear()
            plt.close(fig)


class FilterVis:
    """
    Class for visualising convolutional filter patterns via gradient ascent,
    starting from either a visually neutral image or a specific input image.
    There are several class attributes for controlling the gradient ascent
    and specifying the scaling for the initial, neutral images.
    """
    
    def __init__(
            self, model: keras.Model,
            valid_layers: tuple[layers.Layer] = (layers.Conv2D, layers.Conv2DTranspose),
            scale_factor: float = 0.2, brightness_factor: float = 0.4,
            image_decay: float = 0.8, enable_blur: bool = True,
            blur_freq: int = 5, blur_size: int = 3):
        """
        Initialises a FilterVis class for the given model and parameters.

        Args:
            model: A keras.Model object from which to visualise filters
            valid_layers: Tuple of keras.Layer objects. By default, the filter
                visualisation is enabled for Conv2D and Conv2DTranpose layers.
            scale_factor: Amount to multiply pixel values by.
                See utils.get_blank_image(). Default is 0.2.
            brightness_factor: Amount to add to the pixel values by before
                scaling. See utils.get_blank_image(). Default is 0.4.
            image_decay: A factor to multiply the image pixels by (before
                incorporating the gradients). Default is 0.8.
            enable_blur: Whether to enable Gaussian blur for regularisation
                during the gradient ascent process. Default is True.
            blur_freq: How often to apply Gaussian blur to the image. Default is
                5 (i.e. every fifth iteration).
            blur_size: Strength of the Gaussian blur. Default is 3.
        """

        self.model = model
        self.valid_layers = valid_layers
        self.scale_factor = scale_factor
        self.brightness_factor = brightness_factor
        self.image_decay = image_decay
        self.enable_blur = enable_blur
        self.blur_freq = blur_freq
        self.blur_size = blur_size


    def compute_image_loss(
            self, image: tf.Tensor, filter_index: int,
            extractor: keras.Model, activation_crop: int) -> tf.Tensor:
        """
        Computes the loss, which in this context is
        the mean activation of a given index filter.

        Args:
            image: Input image
            filter_index: The filter to compute loss with respect to
            extractor: keras.Model object that returns the output of a layer
            activation_crop: Amount by which to crop the filter activation.
        
        Returns:
            tf.Tensor of the loss
        """
        activation = extractor(image,training=False)
        # trim border pixels
        if activation_crop > 0:
            filter_activation = activation[
                :, activation_crop:-activation_crop,
                activation_crop:-activation_crop, filter_index
            ]
        else:
            filter_activation = activation[..., filter_index]
        return tf.reduce_mean(filter_activation)


    def gradient_ascent_step(
            self, image: tf.Tensor, filter_index: int, extractor: keras.Model,
            learning_rate: float, activation_crop: int, current_step: int) -> tf.Tensor:
        """
        Performs a single gradient ascent step.

        Args:
            image: Input image
            filter_index: Index of the filter to compute loss with
            extractor: keras.Model object
            learning_rate: The degree to which the gradients are multiplied.
            activation_crop: Amount by which to crop the filter activation.
            current_step: The current iteration, used to determine when to
                apply Gaussian blur.

        Returns:
            A tf.Tensor of the image after the gradient ascent step
        """

        with tf.GradientTape() as tape:
            tape.watch(image)
            loss = self.compute_image_loss(
                image, filter_index=filter_index,
                extractor=extractor, activation_crop=activation_crop
            )
        gradients = tape.gradient(loss, image)

        # apply image regularisations (where applicable)
        gradients = tf.math.l2_normalize(gradients)
        image *= self.image_decay
        if current_step % self.blur_freq == 0 and self.enable_blur:
            if image.shape[-1] == 3:
                image = tf.convert_to_tensor(
                    gaussian(
                        image.numpy()[0, ...],
                        sigma=self.blur_size
                    )[np.newaxis, ...]
                )
            else:
                image = tf.convert_to_tensor(
                    gaussian(
                        image.numpy()[0, ..., 0],
                        sigma=self.blur_size
                    )[np.newaxis, ..., np.newaxis]
                )

        image += learning_rate*gradients
        return image

    def process_filter(
            self, filter_index: int, initial_image: tf.Tensor | None,
            extractor: keras.Model, num_iterations: int, learning_rate: float,
            activation_crop: int) -> tf.Tensor:
        """
        Runs a gradient ascent loop to process the filter pattern
        for the given filter index.

        Args:
            filter_index: The filter to process
            initial_image: Optional initial image to start off with.
                If None, a random, visually neutral image is used instead.
            extractor: keras.Model object that returns the outputs
                of the convolutional layer to visualise
            num_iterations: Number of gradient ascent steps
            learning_rate: Learning rate for the gradient ascent step
            activation_crop: Amount by which to crop the filter activation.

        Returns:
            A np.array representing the image of the given filter
        """
        image = (
            get_blank_image(
                shape=(1, *extractor.input_shape[1:]),
                scale_factor=self.scale_factor,
                brightness_factor=self.brightness_factor
            ) if initial_image is None
            else initial_image
        )
        for n in range(num_iterations):
            image = self.gradient_ascent_step(
                image, filter_index=filter_index, extractor=extractor,
                learning_rate=learning_rate, activation_crop=activation_crop,
                current_step=n
            )
        return image


    def postprocess_image(
            self, image: np.ndarray | tf.Tensor, image_crop: int) -> np.ndarray:
        """
        Processes the filter pattern image into an array of uint8 values
        for use with plt.imshow or equivalent. Pixels are first normalised
        into the range [0,1] before being scaled to [0,255]

        Args:
            image: The image to postprocess
            image_crop: The amount by which to center crop the image.

        Returns:
            np.ndarray of type uint8 representing the post-processed image
        """
        # remove batch dimension
        if len(image.shape) == 4:
            image = image[0, ...]

        if image_crop > 0:
            image = image[image_crop:-image_crop, image_crop:-image_crop, :]

        # normalise and rescale
        image = (image - np.min(image)) / (np.ptp(image) + keras.backend.epsilon())
        image *= 255
        return np.clip(image, 0, 255).astype('uint8')


    def plot_filter_pattern(
            self, layer: str | int, initial_image: np.ndarray | tf.Tensor = None,
            num_iterations: int = 30, learning_rate: float = 10,
            activation_crop: int = 1, image_crop: int = 0, figscale: float = 1,
            dpi: float = 100, colormap: str = 'binary_r', fig_aspect: str = 'uniform',
            fig_orient: str = 'h', include_title: bool = False,
            include_corner_axis: bool = False) -> Figure:
        """
        Visualises and plots the patterns of each filter in the given layer.

        Args:
            layer: Name or index of the keras.layers.Conv2D layer to visualise
            initial_image: An optional image to run the gradient ascent on.
                If set to None, a visually neutral "blank" image is used instead.
                Defaults to None. Image shape must be (1, size, size, depth).
            num_iterations: The number of gradient ascent steps
            learning_rate: Learning rate for each gradient ascent step
            activation_crop: Amount by which to crop the filter activation.
                This is used to avoid border artifacts, and should therefore be kept
                small. Defaults to 1.
            image_crop: Amount by which to crop the final image during
                postprocessing. Defaults to 0.
            figscale: Base figure size multiplier, passed to plt.figure.
            dpi: Base resolution, passed to plt.figure
            colormap: Base colormap, passed to plt.figure.
                Note this only applies to monochromatic image inputs.
            fig_aspect: One of 'uniform' or 'wide'. Use 'uniform' for
                squarish plots and 'wide' for rectangular.
            fig_orient: One of 'h' or 'v'. If set to 'h', the number of columns
                will be >= the number of rows (vice versa if set to 'v').
            include_title: Whether to add a title to the plot.
                Defaults to False
            include_corner_axis: Whether to add an axis
                to the bottom-left subplot. Defaults to False

        Returns:
            A matplotlib.pyplot.Figure object

        Raises:
            ValueError if trying to call on an invalid layer
        """

        if isinstance(layer, int):
            layer = self.model.layers[layer]
        elif isinstance(layer, str):
            layer = self.model.get_layer(name=layer)
        else:
            raise ValueError(f'Invalid layer type')
        if not isinstance(layer, self.valid_layers):
            raise ValueError(f'Invalid layer, must be one of {self.valid_layers}')

        activation_crop = max(0, activation_crop)
        image_crop = max(0, image_crop)
        if initial_image is not None:
            initial_image = tf.convert_to_tensor(initial_image)

        keras.backend.clear_session()
        extractor = keras.Model(inputs=self.model.input, outputs=layer.output)

        num_filters = extractor.output_shape[-1]
        nrows, ncols = obtain_reasonable_figsize(
            num_filters, aspect_mode=fig_aspect, orient=fig_orient
        )

        fig = plt.figure(figsize=(figscale*ncols, figscale*nrows), dpi=dpi)
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        if include_title:
            fig.add_suptitle(layer.name)

        for i in range(num_filters):
            fig.add_subplot(nrows, ncols, i+1)
            # obtain the image
            image = self.process_filter(
                i, initial_image, extractor,
                num_iterations, learning_rate, activation_crop
            )
            # normalise it for plotting
            image = self.postprocess_image(image, image_crop)
            # now plot it
            if image.shape[-1] == 1:
                plt.imshow(image[..., 0], cmap=colormap)
            else:
                plt.imshow(image)
            if include_corner_axis:
                # remove axes from all but the bottom-left subplot
                if i != (nrows - 1)*ncols:
                    plt.axis('off')
            else:
                plt.axis('off')

        return fig


    def plot_all_patterns(
            self, initial_image: np.ndarray | tf.Tensor = None, num_iterations: int = 30,
            learning_rate: float = 10, activation_crop: int = 1, image_crop: int = 1,
            figscale: float = 1, dpi: float = 100, colormap: str = 'binary_r',
            fig_aspect: str = 'uniform', fig_orient: str = 'h', facecolor: str = 'white',
            save_dir: str = 'filter_patterns', save_format: str = 'png',
            prefix: str = '', suffix: str = '', include_titles: bool = False,
            include_corner_axis: bool = False, verbose: bool = True) -> None:
        """
        Visualises and plots the filter patterns of all valid layers.
        Each plot is saved to save_dir.
        NB: This can potentially take a (very) long time to run for larger models.

        Args:
            initial_image: An optional image to run the gradient ascent on.
                If set to None, a visually neutral "blank" image is used instead.
                Defaults to None. Image shape must be (1, size, size, depth)
            num_iterations: The number of gradient ascent steps
            learning_rate: Learning rate for each gradient ascent step
            activation_crop: Amount by which to crop the filter activation.
                This is used to avoid border artifacts, and should therefore be kept
                small. Defaults to 1.
            image_crop: Amount by which to crop the final image during
                postprocessing. Defaults to 0.
            figscale: Base figure size multiplier, passed to plt.figure.
            dpi: Base resolution, passed to plt.figure
            colormap: Base colormap, passed to plt.figure.
            fig_aspect: One of 'uniform' or 'wide'. Use 'uniform' for
                squarish plots and 'wide' for rectangular.
            fig_orient: One of 'h' or 'v'. If set to 'h', the number of columns
                will be >= the number of rows (vice versa if set to 'v').
            facecolor: Background color, passed to fig.savefig.
            save_dir: Directory to save the images.
                Each image is named according to the layername.
            save_format: Format with which to save each image
                e.g 'png', 'pdf', 'jpg'
            prefix: Prefix to prepend to each filename.
            suffix: Suffix to append to each filename.
            include_titles: Whether to include titles in each figure.
                Defaults to False.
            include_corner_axis: Whether to include axes
                on the bottom left subplot. Defaults to False.
            verbose: Whether to include print statements.
        """

        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass

        layer_names = [
            layer.name for layer in self.model.layers
            if isinstance(layer, self.valid_layers)
        ]

        for layer_name in layer_names:
            fig = self.plot_filter_pattern(
                layer_name, initial_image=initial_image,
                num_iterations=num_iterations, learning_rate=learning_rate,
                activation_crop=activation_crop, image_crop=image_crop,
                figscale=figscale, dpi=dpi, colormap=colormap, fig_aspect=fig_aspect,
                fig_orient=fig_orient, include_title=include_titles,
                include_corner_axis=include_corner_axis
            )
            file_name = os.path.join(
                save_dir, f'{prefix}{layer_name}{suffix}.{save_format}'
            )
            if verbose:
                print(f'Saving to {file_name}')
            fig.savefig(
                file_name,format=save_format,facecolor=facecolor,bbox_inches='tight'
            )

            fig.clear()
            plt.close(fig)