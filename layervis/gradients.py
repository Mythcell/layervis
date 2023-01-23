"""
Code for gradient-based visualisations, e.g. class activation maps with Grad-CAM.

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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from .utils import obtain_reasonable_figsize


class GradCAM():
    """Class for visualising class activation maps with Grad-CAM"""

    def __init__(self, model):
        """Initialises a GradCAM object to be used with the given model.

        Args:
            model (keras.Model): keras.Model object.
        """
        self.model = model


    def plot_heatmap(self, image, pred_index, layer, colormap='jet', heatmap_alpha=0.42,
        figsize=10, dpi=100):
        """
        Plots a class activation heatmap using Grad-CAM for the given image, prediction
        index and layer.

        Args:
            image (np.array): An example image to plot the heatmap over. Must be a
                numpy array of shape (1,width,height,depth).
            pred_index (int): Index corresponding to the predicted class to visualise.
            layer (int|str): The layer from which to visualise the gradients. Can be
                specified either by a layer index (int) or layer name (str). The layer
                must have an output shape of size 4.
            colormap (str): Colormap to use for the activation heatmap. Default is 'jet'.
            heatmap_alpha (float): Opacity of the overlaid heatmap. Default is 0.4.
            figsize (int): Figure size, passed to plt.figure. Default is 10.
            dpi (int): Base resolution, passed to plt.figure. Default is 100.
        
        Returns:
            A plt.figure object corresponding to the heatmap superimposed over the
            original image.
        """

        # extract the desired layer and check that it has the correct output shape
        if isinstance(layer,int):
            layer = self.model.layers[layer]
        else:
            layer = self.model.get_layer(layer)
        if len(layer.output_shape) != 4:
            raise ValueError(f'Layer {layer.name} has an invalid output shape. '
                f'The output shape must have 4 dimensions.')
        
        gradmodel = keras.Model(inputs=self.model.input,
            outputs=[layer.output,self.model.output])

        with tf.GradientTape() as tape:
            layer_output, preds = gradmodel(image)
            pred_class_output = preds[:, pred_index]
        
        # create the heatmap based on the pooled gradients 
        grads = tape.gradient(pred_class_output, layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.squeeze(layer_output[0] @ pooled_grads[...,tf.newaxis])

        # normalise heatmap and colour it according to the desired colormap
        heatmap = (heatmap - np.min(heatmap)) / np.ptp(heatmap)
        heatmap = np.uint8(heatmap*255)
        cmap_colors = plt.get_cmap(colormap)(np.arange(256))[:,:3]

        # resize image
        heatmap = keras.preprocessing.image.array_to_img(cmap_colors[heatmap])
        heatmap = heatmap.resize((image.shape[1],image.shape[2]))
        heatmap = keras.preprocessing.image.img_to_array(heatmap)/255.

        # finally combine the heatmap with the original image
        final_heatmap = image[0,...] + heatmap * heatmap_alpha
        final_heatmap = keras.preprocessing.image.array_to_img(final_heatmap)

        # create and return a matplotlib figure
        fig = plt.figure(figsize=(figsize,figsize),dpi=dpi)
        plt.imshow(final_heatmap)
        plt.axis('off')

        return fig