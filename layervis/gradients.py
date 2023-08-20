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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from skimage.filters import gaussian
import os

from .utils import get_blank_image, obtain_reasonable_figsize


class GradCAM():
    """Class for visualising class activation maps with GradCAM"""

    def __init__(self, model: keras.Model):
        """Initialises a GradCAM object to be used with the given model.

        Args:
            model: keras.Model object.
        """
        self.model = model


    def plot_heatmap(
            self, image: np.ndarray, pred_index: int, layer: int | str,
            colormap: str = 'jet', heatmap_alpha: float = 0.42,
            figsize: float = 10, dpi: float = 100) -> Figure:
        """
        Plots a class activation heatmap using Grad-CAM for the given image, prediction
        index and layer.

        Args:
            image: An example image to plot the heatmap over. Must be a
                numpy array of shape (1, width, height, depth).
            pred_index: Index corresponding to the predicted class to visualise.
            layer: The layer from which to visualise the gradients. Can be
                specified either by a layer index (int) or layer name (str). The layer
                must have an output shape of size 4.
            colormap: Colormap to use for the activation heatmap. Default is 'jet'.
            heatmap_alpha: Opacity of the overlaid heatmap. Default is 0.4.
            figsize: Figure size, passed to plt.figure. Default is 10.
            dpi: Base resolution, passed to plt.figure. Default is 100.
        
        Returns:
            A matplotlib.figure.Figure object corresponding to the
            heatmap superimposed over the original image.
        """

        # extract the desired layer and check that it has the correct output shape
        if isinstance(layer,int):
            layer = self.model.layers[layer]
        else:
            layer = self.model.get_layer(layer)
        if len(layer.output_shape) != 4:
            raise ValueError(
                f'Layer {layer.name} has an invalid output shape. '
                f'The output shape must have 4 dimensions.'
            )
        
        gradmodel = keras.Model(
            inputs=self.model.input,
            outputs=[layer.output, self.model.output]
        )

        with tf.GradientTape() as tape:
            layer_output, preds = gradmodel(image)
            pred_class_output = preds[:, pred_index]
        
        # create the heatmap based on the pooled gradients 
        grads = tape.gradient(pred_class_output, layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.squeeze(layer_output[0] @ pooled_grads[..., tf.newaxis])

        # normalise heatmap and colour it according to the desired colormap
        heatmap = (
            (heatmap - np.min(heatmap))
            / (np.ptp(heatmap) + keras.backend.epsilon())
        )
        heatmap = np.uint8(heatmap*255)
        cmap_colors = plt.get_cmap(colormap)(np.arange(256))[:, :3]

        # resize image
        heatmap = keras.preprocessing.image.array_to_img(cmap_colors[heatmap])
        heatmap = heatmap.resize((image.shape[1],image.shape[2]))
        heatmap = keras.preprocessing.image.img_to_array(heatmap)/255.

        # finally combine the heatmap with the original image
        final_heatmap = image[0, ...] + heatmap * heatmap_alpha
        final_heatmap = keras.preprocessing.image.array_to_img(final_heatmap)

        # create and return a matplotlib figure
        fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
        plt.imshow(final_heatmap)
        plt.axis('off')

        return fig
    

class GradientSaliency():
    """
    Class for visualising saliency maps via vanilla gradient backpropagation.
    """
    
    def __init__(self, model: keras.Model):
        """
        Initialises a GradientSalency object with the given model.

        Args:
            model (keras.Model): A Keras model.
        """
        self.model = model


    def plot_saliency_map(
            self, input_image: tf.Tensor | np.ndarray, class_index: int = 0,
            image_mode: str = 'overlay', overlay_alpha: float = 0.5, figsize: float = 6,
            dpi: float = 100, image_cmap: str = 'binary_r', overlay_cmap: str = 'jet',
            annotate_index: bool = False) -> Figure:
        """
        Plot a gradient-based saliency map for the given input image and class index.

        Args:
            input_image: The desired input Tensor or np.ndarray.
            class_index: Output layer index to backpropagate from. Default is 0.
            image_mode: One of 'overlay', 'saliency' or 'image'. If 'overlay',
                plots the image overlaid with the saliency heatmap. The other two options
                plot just the saliency map or image by itself respectively.
                Default is 'overlay'.
            overlay_alpha: Alpha for the overlaid saliency heatmap, passed to
                plt.imshow. Default is 0.5. Has no effect for image_mode != 'overlay'.
            figsize: Base figure size; passed to plt.figure. Default is 6.
            dpi: Base figure resolution, passed to plt.figure.
            image_cmap: The colormap to use for the image (ignored for RGB images).
                Default is 'binary_r'.
            overlay_cmap: The colormap to use for the saliency heatmap.
                Default is 'jet'.
            annotate_index: Whether to annotate the plot with its corresponding
                class index. Default is False.

        Returns:
            A matplotlib.figure.Figure object with the desired plot.
        """

        if not isinstance(input_image, tf.Tensor):
            input_image = tf.convert_to_tensor(input_image)

        with tf.GradientTape() as tape:
            tape.watch(input_image)
            loss = self.model(input_image)[:, class_index]
        grads = tape.gradient(loss, input_image)[0].numpy()
        grads = (grads - np.min(grads)) / (np.ptp(grads) + keras.backend.epsilon())

        fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
        # plot the image with the desired image mode
        if image_mode != 'saliency':
            plt.imshow(input_image[0], cmap=image_cmap)
            if image_mode == 'overlay':
                plt.imshow(grads, alpha=overlay_alpha, cmap=overlay_cmap)
        else:
            plt.imshow(grads, cmap=overlay_cmap)
        # optional annotation to indicate the class index at the top-left of the plot
        if annotate_index:
            plt.annotate(
                f'{class_index}', xy=(0.02, 0.92), xycoords='axes fraction',
                ha='left', fontsize=24, c='white'
            )
        plt.axis('off')
        return fig
    

    def plot_saliency_maps(
            self, input_image: tf.Tensor | np.ndarray, class_indices: list[int] = [],
            image_mode: str = 'overlay', overlay_alpha: float = 0.5, figsize: float = 6,
            dpi: float = 100, image_cmap: str = 'binary_r', overlay_cmap: str = 'jet',
            annotate_index: bool = False, facecolor: str = 'white',
            save_dir: str = 'saliency_maps', save_str: str = '',
            save_format: str = 'png') -> None:
        """
        Plot and save gradient-based saliency maps for the given image
        and class indices.

        Args:
            input_image: The desired input Tensor or np.ndarray.
            class_indices: Output layer indices to backpropagate from. 
                An empty list will automatically plot saliency maps for all output
                class indices, as inferred from the model's output shape (and under the
                assumption that the outputs correspond to classes.)
            image_mode: One of 'overlay', 'saliency' or 'image'. If 'overlay',
                plots the image overlaid with the saliency heatmap. The other two options
                plot just the saliency map or image by itself respectively.
                Default is 'overlay'.
            overlay_alpha: Alpha for the overlaid saliency heatmap, passed to
                plt.imshow. Default is 0.5. Has no effect for image_mode != 'overlay'.
            figsize: Base figure size; passed to plt.figure. Default is 6.
            dpi: Base figure resolution, passed to plt.figure.
            image_cmap: The colormap to use for the image (ignored for RGB images).
                Default is 'binary_r'.
            overlay_cmap: The colormap to use for the saliency heatmap.
                Default is 'jet'.
            annotate_index: Whether to annotate the plots with their corresponding
                class indices. Default is False.
            facecolor: Figure background colour, passed to fig.savefig.
                Default is 'white'.
            save_dir: Directory to save the plots to. Default is 'saliency_maps'.
            save_str: Base output filename for each plot. Plots are saved with the
                filename [save_str]classX.[save_format] where X is the class index.
                Default is '' (i.e. class0, class1, ...)
            save_format: Format to save the image with (e.g. 'jpg','png',etc.).
                Default is 'png'.
        """

        if len(class_indices) == 0:
            class_indices = list(range(self.model.output_shape[-1]))

        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass
        
        for ci in class_indices:
            fig = self.plot_saliency_map(
                input_image, class_index=ci, image_mode=image_mode,
                overlay_alpha=overlay_alpha, figsize=figsize, dpi=dpi,
                image_cmap=image_cmap, overlay_cmap=overlay_cmap,
                annotate_index=annotate_index
            )
            fig.savefig(
                os.path.join(f'{save_dir}', f'{save_str}class{ci:02d}.{save_format}'),
                format=save_format, facecolor=facecolor, bbox_inches='tight'
            )
            fig.clear()
            plt.close(fig)


class ClassModel():
    """
    Class for generating class model images.
    """

    def __init__(
            self, model: keras.Model, scale_factor: float = 0.2,
            brightness_factor: float = 0.4):
        """
        Instantiates a ClassModel object to visualise the given model.
        The scale and brightness factors control the starting image
        for the gradient ascent loop.

        Args:
            model: The Keras model
            scale_factor: All pixels are multiplied by this amount.
                Is thus equal to the difference between the
                maximal and minimal pixel values.
            brightness_factor: This amount is added to all pixels.
                Is thus equal to the minimum pixel value.
        """
        self.model = model
        self.scale_factor = scale_factor
        self.brightness_factor = brightness_factor
        self.score_model = None


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


    def plot_class_model(
            self, class_index: int, score_layer: int | str = -2,
            num_iterations: int = 30, learning_rate: int = 10, image_decay: float = 0.8,
            enable_blur: bool = True, blur_freq: int = 5, blur_size: int = 3,
            figsize: float = 6, dpi: float = 100, colormap: str = 'viridis',
            annotate_index: bool = False) -> Figure:
        """
        Generates and plots a class model image for the desired class index. Uses
        gradient ascent to generate an image that maximises the activation of a given
        class score. It is recommended that the gradient ascent is computed from the
        unnormalised class scores prior to any softmax activation (e.g. the Dense layer
        before the final Softmax output layer).

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
            annotate_index: Whether to annotate the plots with their corresponding
                class indices. Default is False.

        Returns:
            A matplotlib.pyplot Figure object. 
        """

        if isinstance(score_layer, int):
            layer = self.model.layers[score_layer]
        elif isinstance(score_layer, str):
            layer = self.model.get_layer(score_layer)
        else:
            raise ValueError('Invalid score_layer')

        keras.backend.clear_session()
        self.score_model = keras.Model(
            inputs=self.model.input, outputs=layer.output, name='score_model'
        )
        image = get_blank_image(
            shape=(1, *self.model.input_shape[1:]), scale_factor=self.scale_factor,
            brightness_factor=self.brightness_factor
        )

        # main gradient ascent loop
        for i in range(num_iterations):
            image += self.gradient_ascent_step(
                image, class_index, i, learning_rate, image_decay, enable_blur,
                blur_freq, blur_size
            )

        # post-processing
        image = tf.squeeze(image).numpy()
        image = (
            (image - np.min(image))
            / (np.ptp(image) + keras.backend.epsilon())
        )
        image = np.uint8(image*255)
        
        # plot result
        fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
        plt.imshow(image, cmap=colormap)
        plt.axis('off')
        if annotate_index:
            plt.annotate(
                f'{class_index}', xy=(0.02, 0.92), xycoords='axes fraction',
                ha='left', fontsize=24, c='white'
            )
        return fig


    def plot_class_models(
            self, class_indices: list[int] = [], score_layer: int | str = -2,
            num_iterations: int = 30, learning_rate: int = 10, image_decay: float = 0.8,
            enable_blur: bool = True, blur_freq: int = 5, blur_size: int = 3,
            figsize: float = 6, dpi: float = 100, colormap: str = 'viridis',
            annotate_index: bool = False, facecolor: str = 'white',
            save_dir: str = 'class_models', save_str: str = '',
            save_format: str = 'png') -> None:
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
            figsize: Base figure size. Default is 6.
            dpi: Base figure resolution. Default is 100.
            colormap: Image colormap; ignored for RGB images. Default is 'viridis'.
            annotate_index: Whether to annotate the plots with their corresponding
                class indices. Default is False.
            facecolor: Figure background colour, passed to fig.savefig.
                Default is 'white'.
            save_dir: Directory to save the plots to. Default is 'saliency_maps'.
            save_str: Base output filename for each plot. Plots are saved with the
                filename [save_str]classX.[save_format] where X is the class index.
                Default is '' (i.e. class0, class1, ...)
            save_format: Format to save the image with (e.g. 'jpg','png',etc.).
                Default is 'png'. 
        """

        if len(class_indices) == 0:
            class_indices = list(range(self.model.output_shape[-1]))

        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass

        for ci in class_indices:
            fig = self.plot_class_model(
                class_index=ci, score_layer=score_layer, num_iterations=num_iterations,
                learning_rate=learning_rate, image_decay=image_decay,
                enable_blur=enable_blur, blur_freq=blur_freq, blur_size=blur_size,
                figsize=figsize, dpi=dpi, colormap=colormap,
                annotate_index=annotate_index
            )
            fig.savefig(
                os.path.join(f'{save_dir}', f'{save_str}class{ci:02d}.{save_format}'),
                format=save_format, facecolor=facecolor, bbox_inches='tight'
            )
            fig.clear()
            plt.close(fig)


class GuidedBackpropagation():
    """
    Class for visualising saliency maps with guided backpropagation

    TODO
    """

    def __init__(self, model: keras.Model):
        """Instantiates a GuidedBackpropagation object to be used with the given model.

        Args:
            model: A Keras model.
        """
        self.model = model

    
    def plot_saliency_map(
            self, image: tf.Tensor | np.ndarray, layer: int | str) -> Figure:
        """
        Plot a saliency map by applying guided backpropagation from the given layer.
        """

    
    def plot_saliency_maps(
            self, image: tf.Tensor | np.ndarray, layers: list[int | str] = []) -> None:
        """
        Plot and save saliency maps using guided backpropagation from either the list
        of provided layers, or all suitable layers.

        Args:
            image (np.array): An input image of shape (1,width,height,depth).
            layers (list): List of layers to visualise. Elements can be either integers
                specifiying the layer index, or strings specifying the layer by name.
        """