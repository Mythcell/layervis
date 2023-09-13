"""
Code for gradient-based visualisation techniques including saliency maps (simple +
guided backpropagation), class models and class activation maps (GradCAM).

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
from keras.preprocessing.image import array_to_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from skimage.filters import gaussian
import os

from layervis.utils import (
    get_blank_image, get_layer_object,
    highest_mean_filter, obtain_reasonable_figsize
)

class GradCAM():
    """
    Class for visualising class activation maps with GradCAM

    Attributes:
        model: The keras Model to visualise.
    """

    def __init__(self, model: keras.Model):
        """
        Initialises a GradCAM object to be used with the given model.

        Args:
            model: The keras Model to visualise.
        """
        self.model = model


    def generate_heatmap(
            self, input_image: tf.Tensor | np.ndarray, class_index: int,
            layer: int | str | layers.Layer = None) -> np.ndarray:
        """
        Generates a class activation heatmap using Grad-CAM for the given input image
        and prediction class index. Can optionally specify a layer (uses the last
        convolutional layer by default).

        Args:
            input_image: The desired input image.
            class_index: Index corresponding to the predicted class to visualise.
            layer: The layer from which to visualise the gradients. Can be
                specified either by a layer index (int) or layer name (str).
                Default is None, which specifies the last Conv2D layer.

        Raises:
            ValueError if the specified layer has an invalid shape.
        """
        # default value is to take gradients w.r.t the final convolutional layer
        if layer is None:
            for l in self.model.layers[::-1]:
                if isinstance(l, layers.Conv2D):
                    layer = l
                    break
        else:
            layer = get_layer_object(self.model, layer)
        if len(layer.output_shape) != 4:
            raise ValueError(
                f'Layer {layer.name} has an invalid shape. '
                f'It is recommended to use the last Conv2D layer.'
            )

        gradmodel = keras.Model(
            inputs=self.model.input,
            outputs=[layer.output, self.model.output]
        )
        with tf.GradientTape() as tape:
            layer_output, preds = gradmodel(input_image)
            pred_class_output = preds[:, class_index]
        grads = tape.gradient(pred_class_output, layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.squeeze(layer_output[0] @ pooled_grads[..., tf.newaxis]).numpy()
        heatmap = (
            (heatmap - np.min(heatmap))
            / (np.ptp(heatmap) + keras.backend.epsilon())
        )
        return np.uint8(heatmap*255)


    def merge_heatmap(
            self, input_image: tf.Tensor | np.ndarray, heatmap: np.ndarray,
            colormap: str = 'jet', heatmap_alpha: float = 0.42) -> np.ndarray:
        """
        Merge the provided heatmap with the given input image. Returns a new image.

        Args:
            input_image: The desired input image.
            class_index: Index corresponding to the predicted class to visualise.
            layer: The layer from which to visualise the gradients. Can be
                specified either by a layer index (int) or layer name (str).
                Default is None, which specifies the last Conv2D layer.
            colormap: Colormap to use for the activation heatmap. Must be a valid
                matplotlib colormap string. Default is 'jet'.
            heatmap_alpha: Opacity of the overlaid heatmap. Default is 0.42.

        Returns:
            An image of the heatmap merged with the input image.
        """
        cmap_colors = plt.get_cmap(colormap)(np.arange(256))[:, :3]
        heatmap = array_to_img(cmap_colors[heatmap])
        heatmap = heatmap.resize((input_image.shape[1], input_image.shape[2]))
        heatmap = img_to_array(heatmap)/255.
        heatmap = input_image[0, ...] + heatmap * heatmap_alpha
        return np.array(array_to_img(heatmap))


    def plot_heatmap(
            self, input_image: tf.Tensor | np.ndarray, class_index: int,
            layer: int | str | layers.Layer = None, colormap: str = 'jet',
            heatmap_alpha: float = 0.42, figsize: float = 6, dpi: float = 100) -> Figure:
        """
        Generates and plots a class activation heatmap using Grad-CAM for the
        given input image, prediction index and layer.

        Args:
            input_image: The desired input image.
            class_index: Index corresponding to the predicted class to visualise.
            layer: The layer from which to visualise the gradients. Can be
                specified either by a layer index (int) or layer name (str).
                Default is None, which specifies the last Conv2D layer.
            colormap: Colormap to use for the activation heatmap. Must be a valid
                matplotlib colormap string. Default is 'jet'.
            heatmap_alpha: Opacity of the overlaid heatmap. Default is 0.42.
            figsize: Figure size, passed to plt.figure. Default is 10.
            dpi: Base resolution, passed to plt.figure. Default is 100.
        
        Returns:
            The figure with the desired heatmap.
        """
        heatmap = self.generate_heatmap(
            input_image=input_image, class_index=class_index, layer=layer,
        )
        heatmap = self.merge_heatmap(
            input_image=input_image, heatmap=heatmap,
            colormap=colormap, heatmap_alpha=heatmap_alpha
        )
        fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
        plt.imshow(heatmap)
        plt.axis('off')
        return fig
    

    def plot_heatmaps(
            self, input_image: np.ndarray, class_indices: list[int] = [],
            layer: int | str | layers.Layer = None, colormap: str = 'jet',
            heatmap_alpha: float = 0.42, include_class_titles: bool = True,
            figscale: float = 2, dpi: float = 100, textcolor: str = 'white',
            facecolor: str = 'black', save_dir: str = 'heatmaps', save_str: str = '',
            save_format: str = 'png', fig_aspect: str = 'uniform',
            fig_orient: str = 'h') -> None:
        """
        Plots and saves class activation heatmaps using Grad-CAM for the given input
        image with respect to the specified class indices, or all class indices.

        Args:
            input_image: The desired input image.
            class_indices: List of indices to the predicted class to visualise.
            layer: The layer from which to visualise the gradients. Can be
                specified either by a layer index (int) or layer name (str).
                Default is None, which specifies the last Conv2D layer.
            colormap: Colormap to use for the activation heatmap. Default is 'jet'.
            heatmap_alpha: Opacity of the overlaid heatmap. Default is 0.4.
            include_class_titles: Whether to title each subplot with the class index.
                Default is True.
            figscale: Base figure scale multiplier, passed to plt.figure. Default is 2.
            dpi: Base resolution, passed to plt.figure. Default is 100.
            textcolor: Text color to use for the subplot titles. Default is 'white'.
            facecolor: Figure facecolor. Passed to fig.savefig. Default is 'black'.
            save_dir: Directory to save the plot to. Default is 'heatmaps'
            save_str: Optional string to name the output file. The output format is
                gradcam_{save_str}{model.name}.{save_format}. Default is ''.
            save_format: Format to save images with, e.g. 'png' (default), 'pdf', 'jpg'.
            fig_aspect: One of 'uniform' or 'wide', controls the aspect ratio
                of the figure. Use 'uniform' for squarish plots.
                and 'wide' for rectangular. Default is 'uniform'.
            fig_orient One of 'h' or 'v'. If set to 'h', the number of columns
                will be >= the number of rows (vice versa if set to 'v'). Default is 'h'.
            
        Returns:
            An all-in-one figure with each heatmap as a subplot.
        """
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass

        if len(class_indices) == 0:
            class_indices = list(range(self.model.output_shape[-1]))

        nrows, ncols = obtain_reasonable_figsize(
            num_subplots=len(class_indices), aspect_mode=fig_aspect, orient=fig_orient
        )
        fig = plt.figure(figsize=(figscale*ncols, figscale*nrows), dpi=dpi)
        if include_class_titles:
            fig.subplots_adjust(wspace=0.05, hspace=0.2)
        else:
            fig.subplots_adjust(wspace=0.05, hspace=0.05)

        for i, ci in enumerate(class_indices):
            fig.add_subplot(nrows, ncols, i+1)
            heatmap = self.generate_heatmap(
                input_image=input_image, class_index=ci, layer=layer
            )
            heatmap = self.merge_heatmap(
                input_image=input_image, heatmap=heatmap,
                colormap=colormap, heatmap_alpha=heatmap_alpha
            )
            plt.imshow(heatmap)
            plt.axis('off')
            if include_class_titles:
                plt.title(f'{ci}', c=textcolor)
        fig.savefig(
            os.path.join(
                save_dir, f'gradcam_{save_str}{self.model.name}.{save_format}'
            ),
            format=save_format, facecolor=facecolor, bbox_inches='tight'
        )
        fig.clear()
        plt.close(fig)


class GradientSaliency():
    """
    Class for visualising saliency maps via vanilla gradient backpropagation.

    Attributes:
        model: The Keras model to visualise.
        invalid_layers: Tuple of layers.Layer objects to ignore when plotting saliency
            maps with plot_layer_saliency_maps.
    """
    
    def __init__(self, model: keras.Model):
        """
        Initialises a GradientSalency object with the given model.

        Args:
            model: A keras Model to visualise.
        """
        self.model = model
        self.invalid_layers: tuple[layers.Layer] = (layers.InputLayer,)


    def get_saliency_map(
            self, input_image: tf.Tensor | np.ndarray, class_index: int = None,
            layer: int | str | layers.Layer = -1) -> np.ndarray:
        """
        Returns a gradient-based saliency map for the given input image. You can
        optionally specify a class index to visualise and/or layer to backpropagate from.
        By default, it will backpropagate from the final layer with respect to
        the channel with the highest mean activation.

        Args:
            input_image: The desired input image.
            class_index: Class / output channel to visualise the saliency map for.
                If set to None, will visualise the channel with the highest mean
                activation. Default is None.
            layer: The layer to backpropagate from. Can be a layer index, layer name
                or layers.Layer object. Default is -1, i.e. the output layer.

        Returns:
            A tf.Tensor with the desired gradient-based saliency map.
        """
        if not isinstance(input_image, tf.Tensor):
            input_image = tf.convert_to_tensor(input_image)
        layer = get_layer_object(self.model, layer)
        
        keras.backend.clear_session()
        smodel = keras.Model(
            inputs=self.model.inputs, outputs=layer.output, name='saliency_model'
        )
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            pred = smodel(input_image)
            if class_index is None:
                class_index = highest_mean_filter(pred)
            loss = pred[..., class_index]
        grads = tf.squeeze(tape.gradient(loss, input_image)).numpy()
        grads = (grads - np.min(grads)) / (np.ptp(grads) + keras.backend.epsilon())
        return grads


    def plot_saliency_maps(
            self, input_image: tf.Tensor | np.ndarray, class_indices: list[int] = [],
            layer: int | str | layers.Layer = -1, image_mode: str = 'overlay',
            overlay_alpha: float = 0.5, figscale: float = 2, dpi: float = 100,
            image_cmap: str = 'binary_r', overlay_cmap: str = 'jet',
            include_class_titles: bool = True, textcolor: str = 'white',
            facecolor: str = 'black', save_dir: str = 'saliency_maps',
            save_str: str = '', save_format: str = 'png',
            fig_aspect: str = 'uniform', fig_orient: str = 'h') -> None:
        """
        Plot gradient-based saliency map for the given input images. You can optionally
        specify class indices to visualise and/or a layer to backpropagate from.
        By default, it will backpropagate from the final layer with respect to
        the channel with the highest mean activation.

        Args:
            input_image: The desired input image.
            class_indices: Classes / output channels to visualise the saliency map for.
                If set to an empty list, will automatically plot saliency
                maps for all output channels of the provided layer. Specifying None
                as a class will instead visualise the channel with the highest mean
                activation.
            layer: The layer to backpropagate from. Can be a layer index, layer name
                or layers.Layer object. Default is -1, i.e. the output layer.
            image_mode: One of 'overlay', 'saliency' or 'image'. If 'overlay',
                plots the image overlaid with the saliency heatmap. The other two options
                plot just the saliency map or image by itself respectively.
                Default is 'overlay'.
            overlay_alpha: Alpha for the overlaid saliency heatmap, passed to
                plt.imshow. Default is 0.5. Has no effect for image_mode != 'overlay'.
            include_class_titles: Whether to title each subplot with the class index.
                Default is True.
            figscale: Base figure size multiplier; passed to plt.figure. Default is 2.
            dpi: Base figure resolution, passed to plt.figure.
            image_cmap: The colormap to use for the image (ignored for RGB images).
                Default is 'binary_r'.
            overlay_cmap: The colormap to use for the saliency heatmap.
                Default is 'jet'.
            textcolor: Text color to use for the subplot titles. Default is 'white'.
            facecolor: Figure background color. Default is 'black'.
            save_dir: Directory to save the plots to. Default is 'saliency_maps'.
            save_str: Base output filename for each plot. Plots are saved with the
                filename {save_str}class{class_index}.{save_format}.
                Default is '' (i.e. class0, class1, ...).
            save_format: Format to save images with, e.g. 'png' (default), 'pdf', 'jpg'.
            fig_aspect: One of 'uniform' or 'wide', controls the aspect ratio
                of the figure. Use 'uniform' for squarish plots.
                and 'wide' for rectangular. Default is 'uniform'.
            fig_orient One of 'h' or 'v'. If set to 'h', the number of columns
                will be >= the number of rows (vice versa if set to 'v'). Default is 'h'.
        """
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass

        if not isinstance(input_image, tf.Tensor):
            input_image = tf.convert_to_tensor(input_image)
        layer = get_layer_object(self.model, layer)
        if len(class_indices) == 0:
            class_indices = list(range(layer.output_shape[-1]))

        keras.backend.clear_session()
        smodel = keras.Model(
            inputs=self.model.inputs, outputs=layer.output, name='saliency_model'
        )
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(input_image)
            pred = smodel(input_image)

        nrows, ncols = obtain_reasonable_figsize(
            num_subplots=len(class_indices), aspect_mode=fig_aspect, orient=fig_orient
        )
        fig = plt.figure(figsize=(figscale*ncols, figscale*nrows), dpi=dpi)
        if include_class_titles:
            fig.subplots_adjust(wspace=0.05, hspace=0.2)
        else:
            fig.subplots_adjust(wspace=0.05, hspace=0.05)

        for i, ci in enumerate(class_indices):
            fig.add_subplot(nrows, ncols, i+1)
            # use existing tape instead of repeating forward passes
            with tape:
                loss = (
                    pred[..., highest_mean_filter(pred)] if ci is None
                    else pred[..., ci]
                )
            grads = tf.squeeze(tape.gradient(loss, input_image)).numpy()
            grads = (grads - np.min(grads)) / (np.ptp(grads) + keras.backend.epsilon())

            if image_mode != 'saliency':
                plt.imshow(input_image[0], cmap=image_cmap)
                if image_mode == 'overlay':
                    plt.imshow(grads, alpha=overlay_alpha, cmap=overlay_cmap)
            else:
                plt.imshow(grads, cmap=overlay_cmap)
            plt.axis('off')
            if include_class_titles:
                plt.title(f'{ci}', c=textcolor)

        del tape  # important!
        fig.savefig(
            os.path.join(
                save_dir, f'{save_str}{layer.name}.{save_format}'
            ),
            format=save_format, facecolor=facecolor, bbox_inches='tight'
        )
        fig.clear()
        plt.close(fig)
    

    def plot_saliency_maps_layers(
            self, input_image: tf.Tensor | np.ndarray,
            layers_list: list[int | str | layers.Layer] = [],
            class_indices: list[int] = [], max_classes: int = 1024,
            image_mode: str = 'overlay', overlay_alpha: float = 0.5,
            figscale: float = 2, dpi: float = 100, image_cmap: str = 'binary_r',
            overlay_cmap: str = 'jet', include_class_titles: bool = False,
            textcolor: str = 'white', facecolor: str = 'black',
            save_dir: str = 'saliency_maps', save_str: str = '',
            save_format: str = 'png', fig_aspect: str = 'uniform',
            fig_orient: str = 'h') -> None:
        """
        Plots and saves gradient-based saliency maps for the given input image
        in each layer in the specified list of layers.

        Args:
            input_image: The desired input image.
            layers_list: List of layers to plot saliency maps for. Can be a layer index,
                layer name or layer object. If empty, will automatically
                create saliency maps for all layers in the model.
            class_indices: Classes / output channels to visualise the saliency map for.
                If set to an empty list, will automatically plot saliency
                maps for all output channels of the provided layer. Specifying None
                as a class will instead visualise the channel with the highest mean
                activation.
            max_classes: Skips layers containing more than this number of output
                classes / channels. Useful for avoiding huge figures for large Dense
                layers. Default is 1024.
            image_mode: One of 'overlay', 'saliency' or 'image'. If 'overlay',
                plots the image overlaid with the saliency heatmap. The other two options
                plot just the saliency map or image by itself respectively.
                Default is 'overlay'.
            overlay_alpha: Alpha for the overlaid saliency heatmap, passed to
                plt.imshow. Default is 0.5. Has no effect for image_mode != 'overlay'.
            figscale: Base figure size multiplier; passed to plt.figure. Default is 2.
            dpi: Base figure resolution, passed to plt.figure.
            image_cmap: The colormap to use for the image (ignored for RGB images).
                Default is 'binary_r'.
            overlay_cmap: The colormap to use for the saliency heatmap.
                Default is 'jet'.
            include_class_titles: Whether to title each subplot with the class index.
                Default is True.
            textcolor: Text color to use for the subplot titles. Default is 'white'.
            facecolor: Figure background colour, passed to fig.savefig.
                Default is 'black'.
            save_dir: Directory to save the plots to. Default is 'saliency_maps'.
            save_str: Base output filename for each plot. Plots are saved with the
                filename {save_str}{layer.name}.{save_format}. Default is ''
            save_format: Format to save images with, e.g. 'png' (default), 'pdf', 'jpg'.
            fig_aspect: One of 'uniform' or 'wide', controls the aspect ratio
                of the figure. Use 'uniform' for squarish plots.
                and 'wide' for rectangular. Default is 'uniform'.
            fig_orient One of 'h' or 'v'. If set to 'h', the number of columns
                will be >= the number of rows (vice versa if set to 'v'). Default is 'h'.
        """
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass

        if len(layers_list) == 0:
            layers_list = [
                l for l in self.model.layers if not isinstance(l, self.invalid_layers)
            ]
        else:
            layers_list = [get_layer_object(self.model, l) for l in layers_list]
        for l in layers_list:
            if l.output_shape[-1] > max_classes:
                print(
                    f'Skipping layer {l.name} '
                    f'as it has more than {max_classes} output channels'
                )
                continue
            self.plot_saliency_maps(
                input_image=input_image, class_indices=class_indices, layer=l,
                image_mode=image_mode, overlay_alpha=overlay_alpha, figscale=figscale,
                dpi=dpi, image_cmap=image_cmap, overlay_cmap=overlay_cmap,
                include_class_titles=include_class_titles,
                textcolor=textcolor, facecolor=facecolor, save_dir=save_dir,
                save_str=save_str, save_format=save_format, fig_aspect=fig_aspect,
                fig_orient=fig_orient
            )


class ClassModel():
    """
    Class for generating class model images.

    Attributes:
        model: The Keras model to visualise.
        scale_factor: All pixels are multiplied by this amount. Used when generating
            the random initial images.Controls the difference
            between the maximal and minimal pixel values.
        brightness_factor: This amount is added to all pixels thus equal.
            Used when generating the random initial images. Controls the
            minimum pixel value.
        score_model: Intermediate Keras model which outputs class scores. Is included
            as an attribute so as not to be recreated when looping over class indices.
        score_layer: The current model layer being used as the output for the score
            model. If this changes then the score model is recreated.
    """

    def __init__(
            self, model: keras.Model, scale_factor: float = 0.2,
            brightness_factor: float = 0.4):
        """
        Instantiates a ClassModel object to visualise the given model.
        The scale and brightness factors control the starting image
        for the gradient ascent loop.

        Args:
            model: A Keras model to visualise
            scale_factor: All pixels are multiplied by this amount. Used when generating
                the random initial images.Controls the difference
                between the maximal and minimal pixel values.
            brightness_factor: This amount is added to all pixels thus equal.
                Used when generating the random initial images. Controls the
                minimum pixel value.
        """
        self.model = model
        self.scale_factor = scale_factor
        self.brightness_factor = brightness_factor
        self.score_model: keras.Model = None
        self.score_layer: layers.Layer = None


    def gradient_ascent_step(
            self, image: tf.Tensor, class_index: int, current_step: int,
            learning_rate: float, image_decay: float, enable_blur: bool,
            blur_freq: int, blur_size: int) -> tf.Tensor:
        """
        Performs a single gradient ascent step.

        Args:
            image: Input image tensor
            class_index: Index of the class to compute the relevant loss from
            current_step: Current step in the overall loop
                (used to determine whether to apply Gaussian blur regularisation)
            learning_rate: Gradient multiplier.
            image_decay: A factor to multiply the image pixels by (before
                incorporating the gradients).
            enable_blur: Whether to enable Gaussian blur for regularisation
                during the gradient ascent process.
            blur_freq: How often to apply Gaussian blur to the image.
            blur_size: Strength of the Gaussian blur.

        Returns:
            A tf.Tensor of gradients to be added to the image tensor.
        """
        
        with tf.GradientTape() as tape:
            tape.watch(image)
            loss = self.score_model(image)[:, class_index]
        grads = tape.gradient(loss, image)
        grads = tf.math.l2_normalize(grads)
        image *= image_decay
        if current_step % blur_freq == 0 and enable_blur:
            if image.shape[-1] == 3:
                image = tf.convert_to_tensor(
                    gaussian(
                        image.numpy()[0, ...],
                        sigma=blur_size
                    )[np.newaxis, ...]
                )
            else:
                image = tf.convert_to_tensor(
                    gaussian(
                        image.numpy()[0, ..., 0],
                        sigma=blur_size
                    )[np.newaxis, ..., np.newaxis]
                )
        return learning_rate * grads
    

    def generate_class_model(
            self, class_index: int, score_layer: int | str | layers.Layer = -2,
            num_iterations: int = 30, learning_rate: int = 10, image_decay: float = 0.8,
            enable_blur: bool = True, blur_freq: int = 5,
            blur_size: int = 3) -> np.ndarray:
        """
        Generates a class model for the specified class index and score layer.
        It is recommended to use the Dense layer prior to Softmax activation.

        Args:
            class_index: The class index to visualise.
            score_layer: Index or string specifying the score layer
                to run the gradient ascent from. This is usually the penultimate layer
                of a CNN classifier, assuming the final layer is a Softmax activation.
                Defaults to -2.
            num_iterations: Number of iterations to run the gradient ascent for.
                Default is 30.
            learning_rate: Learning rate for the gradient ascent step. Default is 10.
            image_decay: A factor to multiply the image pixels by (before
                incorporating the gradients). Default is 0.8.
            enable_blur: Whether to enable Gaussian blur for regularisation
                during the gradient ascent process. Default is True.
            blur_freq: How often to apply Gaussian blur to the image. Default is
                5 (i.e. every fifth iteration).
            blur_size: Strength of the Gaussian blur. Default is 3.

        Returns:
            np.ndarray of the class model image.
        """
        layer = get_layer_object(self.model, score_layer)
        # only create the score model if it does not already exist
        # OR if the specified layer has changed
        # (mild performance boost when looping multiple classes)
        if self.score_model is None or layer != self.score_layer:
            keras.backend.clear_session()
            self.score_model = keras.Model(
                inputs=self.model.input, outputs=layer.output, name='score_model'
            )
            self.score_layer = layer

        image = get_blank_image(
            shape=(1, *self.model.input_shape[1:]), scale_factor=self.scale_factor,
            brightness_factor=self.brightness_factor
        )
        for i in range(num_iterations):
            image += self.gradient_ascent_step(
                image, class_index, i, learning_rate, image_decay,
                enable_blur, blur_freq, blur_size
            )
        image = tf.squeeze(image).numpy()
        image = (
            (image - np.min(image))
            / (np.ptp(image) + keras.backend.epsilon())
        )
        return np.uint8(image*255)
    

    def plot_class_model(
            self, class_index: int, score_layer: int | str | layers.Layer = -2,
            num_iterations: int = 30, learning_rate: int = 10, image_decay: float = 0.8,
            enable_blur: bool = True, blur_freq: int = 5, blur_size: int = 3,
            figsize: float = 6, dpi: float = 100, colormap: str = 'viridis') -> Figure:
        """
        Generates and plots a class model image for the desired class index. Uses
        gradient ascent to generate an image that maximises the activation of a given
        class score. It is recommended that the gradient ascent is computed from the
        unnormalised class scores / logits prior to any softmax activation
        (e.g. the Dense layer before the final Softmax output layer).

        Args:
            class_index: The class index to visualise.
            score_layer: Index or string specifying the score layer
                to run the gradient ascent from. This is usually the penultimate layer
                of a CNN classifier, assuming the final layer is a Softmax activation.
                Defaults to -2.
            num_iterations: Number of iterations to run the gradient ascent for.
                Default is 30.
            learning_rate: Learning rate for the gradient ascent step. Default is 10.
            image_decay: A factor to multiply the image pixels by (before
                incorporating the gradients). Default is 0.8.
            enable_blur: Whether to enable Gaussian blur for regularisation
                during the gradient ascent process. Default is True.
            blur_freq: How often to apply Gaussian blur to the image. Default is
                5 (i.e. every fifth iteration).
            blur_size: Strength of the Gaussian blur. Default is 3.
            figsize: Base figure size. Default is 6.
            dpi: Base figure resolution. Default is 100.
            colormap: Image colormap; ignored for RGB images. Default is 'viridis'.

        Returns:
            A matplotlib.pyplot Figure object. 
        """
        image = self.generate_class_model(
            class_index=class_index, score_layer=score_layer,
            num_iterations=num_iterations, learning_rate=learning_rate,
            image_decay=image_decay, enable_blur=enable_blur,
            blur_freq=blur_freq, blur_size=blur_size
        )
        fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
        plt.imshow(image, cmap=colormap)
        plt.axis('off')
        return fig


    def plot_class_models(
            self, class_indices: list[int] = [],
            score_layer: int | str | layers.Layer = -2, num_iterations: int = 30,
            learning_rate: int = 10, image_decay: float = 0.8,
            enable_blur: bool = True, blur_freq: int = 5, blur_size: int = 3,
            figscale: float = 2, dpi: float = 100, colormap: str = 'viridis',
            include_class_titles: bool = True, textcolor: str = 'white',
            facecolor: str = 'black', save_dir: str = 'class_models',
            save_str: str = '', save_format: str = 'png',
            fig_aspect: str = 'uniform', fig_orient: str = 'h') -> None:
        """
        Generates, plots and saves class model images for the desired class indices.
        Uses gradient ascent to generate an image that maximises the activation
        of a given class score. It is recommended that the gradient ascent is computed
        from the unnormalised class scores prior to any softmax activation
        (e.g. the Dense layer before the final Softmax output layer).

        Args:
            class_indices: List of class indices to generate class models for.
                If an empty list is provided, will automatically plot class models
                for all classes.
            score_layer: Index or string specifying the score layer
                to run the gradient ascent from. This is usually the penultimate layer
                of a CNN classifier, assuming the final layer is a Softmax activation.
                Defaults to -2.
            num_iterations: Number of iterations to run the gradient ascent for.
                Default is 30.
            learning_rate: Learning rate for the gradient ascent step. Default is 10.
            image_decay: A factor to multiply the image pixels by (before
                incorporating the gradients). Default is 0.8.
            enable_blur: Whether to enable Gaussian blur for regularisation
                during the gradient ascent process. Default is True.
            blur_freq: How often to apply Gaussian blur to the image. Default is
                5 (i.e. every fifth iteration).
            blur_size: Strength of the Gaussian blur. Default is 3.
            figscale: Base figure size multiplier; passed to plt.figure. Default is 2.
            dpi: Base figure resolution. Default is 100.
            colormap: Image colormap; ignored for RGB images. Default is 'viridis'.
            include_class_titles: Whether to title each subplot with the class index.
                Default is True.
            textcolor: Text color to use for the subplot titles. Default is 'white'.
            facecolor: Figure background colour, passed to fig.savefig.
                Default is 'black'.
            save_dir: Directory to save the plots to. Default is 'saliency_maps'.
            save_str: Base output filename. The plot are saved with the
                filename [save_str]classmodels.[save_format] where X is the class index.
                Default is ''.
            save_format: Format to save images with, e.g. 'png' (default), 'pdf', 'jpg'.
            fig_aspect: One of 'uniform' or 'wide', controls the aspect ratio
                of the figure. Use 'uniform' for squarish plots.
                and 'wide' for rectangular. Default is 'uniform'.
            fig_orient One of 'h' or 'v'. If set to 'h', the number of columns
                will be >= the number of rows (vice versa if set to 'v'). Default is 'h'.
        """
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass

        if len(class_indices) == 0:
            class_indices = list(range(self.model.output_shape[-1]))
        nrows, ncols = obtain_reasonable_figsize(
            num_subplots=len(class_indices), aspect_mode=fig_aspect, orient=fig_orient
        )
        fig = plt.figure(figsize=(figscale*ncols, figscale*nrows), dpi=dpi)
        if include_class_titles:
            fig.subplots_adjust(wspace=0.05, hspace=0.2)
        else:
            fig.subplots_adjust(wspace=0.05, hspace=0.05)

        for i, ci in enumerate(class_indices):
            fig.add_subplot(nrows, ncols, i+1)
            image = self.generate_class_model(
                class_index=ci, score_layer=score_layer,
                num_iterations=num_iterations, learning_rate=learning_rate,
                image_decay=image_decay, enable_blur=enable_blur,
                blur_freq=blur_freq, blur_size=blur_size
            )
            plt.imshow(image, cmap=colormap)
            plt.axis('off')
            if include_class_titles:
                plt.title(f'{ci}', c=textcolor)
        fig.savefig(
            os.path.join(save_dir, f'{save_str}classmodels.{save_format}'),
            format=save_format, facecolor=facecolor, bbox_inches='tight'
        )
        fig.clear()
        plt.close(fig)


class GuidedBackpropagation():
    """
    Class for visualising saliency maps with guided backpropagation.

    Attributes:
        model: The Keras model to visualise.
        invalid_layers: Tuple of layers.Layer objects to ignore when plotting saliency
            maps with plot_layer_saliency_maps.
    """

    def __init__(self, model: keras.Model):
        """Instantiates a GuidedBackpropagation object to be used with the given model.

        Args:
            model: A Keras model to visualise.
        """
        self.model = model
        self.invalid_layers: tuple[layers.Layer] = (layers.InputLayer,)

    @tf.custom_gradient
    def guided_relu(x):
        """
        Custom gradient to apply for guided backpropagation.
        """
        def grad(dy):
            return tf.cast(dy > 0, 'float32') * tf.cast(x > 0, 'float32') * dy
        return tf.nn.relu(x), grad


    def get_saliency_map(
            self, input_image: tf.Tensor | np.ndarray, class_index: int = None,
            layer: int | str | layers.Layer = None,
            saliency_mode: str = 'norm') -> np.ndarray:
        """
        Returns saliency map with guided backpropagation.

        Args:
            input_image: Input image tensor
            class_index: Class / output channel to visualise the saliency map for.
                If set to None, will visualise the channel with the highest mean
                activation. Default is None.
            layer: The layer from which to visualise the gradients. Can be
                specified either by a layer index (int) or layer name (str).
                Default is None, which specifies the last Conv2D layer.
            saliency_mode: One of 'norm', 'abs', 'pos' or ''. Specifies whether
                to return the normalised saliency ('norm'), the absolute values of the
                saliency ('abs'), only all nonnegative values ('pos'), or the saliency
                as-is with no further normalisation (''). Default is 'norm'.

        Returns:
            np.ndarray of the normalised saliency map.
        """
        if not isinstance(input_image, tf.Tensor):
            input_image = tf.convert_to_tensor(input_image)
        if layer is None:
            for l in self.model.layers[::-1]:
                if isinstance(l, layers.Conv2D):
                    layer = l
                    break
        else:
            layer = get_layer_object(self.model, layer)
        # create new model and replace all relus with guided_relus.
        keras.backend.clear_session()
        gbmodel = keras.Model(
            inputs=self.model.inputs, outputs=layer.output, name='gbmodel'
        )
        for l in gbmodel.layers:
            if (
                (hasattr(l, 'activation') and l.activation == keras.activations.relu)
                or isinstance(l, layers.ReLU)
            ):
                l.activation = self.guided_relu
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            pred = gbmodel(input_image)
            if class_index is None:
                class_index = highest_mean_filter(pred)
            loss = pred[..., class_index]
        saliency = tf.squeeze(tape.gradient(loss, input_image)).numpy()
        if saliency_mode == 'abs':
            saliency = np.abs(saliency)
        elif saliency_mode == 'pos':
            saliency = np.maximum(saliency, 0)
        elif saliency_mode == 'norm':
            saliency = (
                (saliency - np.min(saliency))
                / (np.ptp(saliency) + keras.backend.epsilon())
            )
        return saliency

    
    def plot_saliency_maps(
            self, input_image: tf.Tensor | np.ndarray, class_indices: list[int] = [],
            layer: int | str | tf.Tensor = None, saliency_mode: str = 'norm',
            figscale: float = 2, dpi: float = 100, colormap: str = 'binary_r',
            include_class_titles: bool = True, textcolor: str = 'white',
            facecolor: str = 'black', save_dir: str = 'guided_backprop',
            save_str: str = '', save_format: str = 'png', fig_aspect: str = 'uniform',
            fig_orient: str = 'h') -> None:
        """
        Plot and save saliency maps using guided backpropagation for the given class
        indices with respect to the specified layer.

        Args:
            input_image: An input image of shape (1,width,height,depth).
            class_indices: Classes / output channels to visualise the saliency map for.
                If set to an empty list, will automatically plot saliency
                maps for all output channels of the provided layer. Specifying None
                as a class will instead visualise the channel with the highest mean
                activation.
            layer: The layer from which to visualise the gradients. Can be
                specified either by a layer index (int) or layer name (str).
                Default is None, which specifies the last Conv2D layer.
            saliency_mode: One of 'all', 'abs' or 'pos'. Specifies whether to visualise
                the gradients as-is (all), the absolute values of the gradients (abs),
                or only all nonnegative values (pos). Default is 'all'.
            figscale: Base figure size multiplier; passed to plt.figure. Default is 2.
            dpi: Base figure resolution. Default is 100.
            colormap: Colormap to use for the saliency map. Default is 'binary_r'.
            include_class_titles: Whether to title each subplot with the class index.
                Default is True.
            textcolor: Text color to use for the subplot titles. Default is 'white'.
            facecolor: Figure background colour, passed to fig.savefig.
                Default is 'black'.
            save_dir: Directory to save the saliency maps. Default is 'guided_backprop'.
            save_str: Base output filename for each plot. Plots are saved with the
                filename {save_str}{layer.name}.{save_format}. Default is ''.
            save_format: Format to save the image with (e.g. 'jpg','png',etc.).
                Default is 'png'.
            fig_aspect: One of 'uniform' or 'wide', controls the aspect ratio
                of the figure. Use 'uniform' for squarish plots.
                and 'wide' for rectangular. Default is 'uniform'.
            fig_orient One of 'h' or 'v'. If set to 'h', the number of columns
                will be >= the number of rows (vice versa if set to 'v'). Default is 'h'.
        """
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass

        if not isinstance(input_image, tf.Tensor):
            input_image = tf.convert_to_tensor(input_image)
        if layer is None:
            for l in self.model.layers[::-1]:
                if isinstance(l, layers.Conv2D):
                    layer = l
                    break
        else:
            layer = get_layer_object(self.model, layer)
        if len(class_indices) == 0:
            class_indices = list(range(layer.output_shape[-1]))

        keras.backend.clear_session()
        gbmodel = keras.Model(
            inputs=self.model.inputs, outputs=layer.output, name='gbmodel'
        )
        for l in gbmodel.layers:
            if (
                (hasattr(l, 'activation') and l.activation == keras.activations.relu)
                or isinstance(l, layers.ReLU)
            ):
                l.activation = self.guided_relu
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(input_image)
            pred = gbmodel(input_image)

        nrows, ncols = obtain_reasonable_figsize(
            num_subplots=len(class_indices), aspect_mode=fig_aspect, orient=fig_orient
        )
        fig = plt.figure(figsize=(figscale*ncols, figscale*nrows), dpi=dpi)
        if include_class_titles:
            fig.subplots_adjust(wspace=0.05, hspace=0.2)
        else:
            fig.subplots_adjust(wspace=0.05, hspace=0.05)

        for i, ci in enumerate(class_indices):
            fig.add_subplot(nrows, ncols, i+1)
            with tape:
                loss = (
                    pred[..., highest_mean_filter(pred)] if ci is None
                    else pred[..., ci]
                )
            saliency = tf.squeeze(tape.gradient(loss, input_image)).numpy()
            if saliency_mode == 'abs':
                saliency = np.abs(saliency)
            elif saliency_mode == 'pos':
                saliency = np.maximum(saliency, 0)
            elif saliency_mode == 'norm':
                saliency = (
                    (saliency - np.min(saliency))
                    / (np.ptp(saliency) + keras.backend.epsilon())
                )
            plt.imshow(saliency, cmap=colormap)
            plt.axis('off')
            if include_class_titles:
                plt.title(f'{ci}', c=textcolor)

        del tape # important!
        fig.savefig(
            os.path.join(
                save_dir, f'{save_str}{layer.name}.{save_format}'
            ),
            format=save_format, facecolor=facecolor, bbox_inches='tight'
        )
        fig.clear()
        plt.close(fig)


    def plot_saliency_maps_layers(
            self, input_image: tf.Tensor | np.ndarray,
            layers_list: list[int | str | layers.Layer] = [],
            class_indices: list[int] = [], max_classes: int = 1024,
            saliency_mode: str = 'norm', figscale: float = 2, dpi: float = 100,
            colormap: str = 'binary_r', include_class_titles: bool = True,
            textcolor: str = 'white', facecolor: str = 'black',
            save_dir: str = 'guided_backprop', save_str: str = '',
            save_format: str = 'png', fig_aspect: str = 'uniform',
            fig_orient: str = 'h') -> None:
        """
        Plot and save saliency maps using guided backpropagation for the given class
        indices with respect to each layer in the specified list of layers.

        Args:
            input_image: An input image of shape (1,width,height,depth).
            layers_list: List of layers to plot saliency maps for. Can be a layer index,
                layer name or layer object. If empty, will automatically
                create saliency maps for all layers in the model.
            class_indices: Classes / output channels to visualise the saliency map for.
                If set to an empty list, will automatically plot saliency
                maps for all output channels of the provided layer. Specifying None
                as a class will instead visualise the channel with the highest mean
                activation.
            max_classes: Skips layers containing more than this number of output
                classes / channels. Useful for avoiding huge figures for large Dense
                layers. Default is 1024.
            saliency_mode: One of 'all', 'abs' or 'pos'. Specifies whether to visualise
                the gradients as-is (all), the absolute values of the gradients (abs),
                or only all nonnegative values (pos). Default is 'all'.
            figscale: Base figure size multiplier; passed to plt.figure. Default is 2.
            dpi: Base figure resolution. Default is 100.
            colormap: Colormap to use for the saliency map. Default is 'binary_r'.
            include_class_titles: Whether to title each subplot with the class index.
                Default is True.
            textcolor: Text color to use for the subplot titles. Default is 'white'.
            facecolor: Figure background colour, passed to fig.savefig.
                Default is 'black'.
            save_dir: Directory to save the saliency maps. Default is 'guided_backprop'.
            save_str: Base output filename for each plot. Plots are saved with the
                filename {save_str}{layer.name}.{save_format}. Default is ''.
            save_format: Format to save the image with (e.g. 'jpg','png',etc.).
                Default is 'png'.
            fig_aspect: One of 'uniform' or 'wide', controls the aspect ratio
                of the figure. Use 'uniform' for squarish plots.
                and 'wide' for rectangular. Default is 'uniform'.
            fig_orient One of 'h' or 'v'. If set to 'h', the number of columns
                will be >= the number of rows (vice versa if set to 'v'). Default is 'h'.
        """
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass
        
        if len(layers_list) == 0:
            layers_list = [
                l for l in self.model.layers if not isinstance(l, self.invalid_layers)
            ]
        else:
            layers_list = [get_layer_object(self.model, l) for l in layers_list]
        for l in layers_list:
            if l.output_shape[-1] > max_classes:
                print(
                    f'Skipping layer {l.name} '
                    f'as it has more than {max_classes} output channels'
                )
                continue
            self.plot_saliency_maps(
                input_image=input_image, class_indices=class_indices, layer=l,
                saliency_mode=saliency_mode, figscale=figscale, dpi=dpi,
                colormap=colormap, include_class_titles=include_class_titles,
                textcolor=textcolor, facecolor=facecolor, save_dir=save_dir,
                save_str=save_str, save_format=save_format,
                fig_aspect=fig_aspect, fig_orient=fig_orient
            )