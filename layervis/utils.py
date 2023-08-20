"""Code for various utilities and helper functions.

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
import os

def get_blank_image(
        shape: tuple[int], scale_factor: float = 0.2,
        brightness_factor: float = 0.4) -> tf.Tensor:
    """
    Creates a blank, visually neutral image (predominantly gray).
    Pixel values are uniformly sampled from between 0 and 1, and subsequently
    altered with a multiplicative scale factor and additive brightness factor.
    e.g. To go from [0,1] to [0.4,0.6] (default), use a scale factor of 0.2
    and brightness factor of 0.4.
    Other examples:
        [0,1] to [0.45,0.55]: scale = 0.1, brightness = 0.45
        [0,1] to [-0.1,0.1]: scale = 0.2, brightness = -0.1
        [0,1] to [-1,1]: scale = 2.0, brightness = -1.0

    Args:
        shape: A tuple specifying the desired dimensions
            of the image tensor.
        scale_factor: All pixels are multiplied by this amount.
            Is thus equal to the difference between the
            maximal and minimal pixel values.
        brightness_factor: This amount is added to all pixels.
            Is thus equal to the minimum pixel value.

    Returns:
        A tf.Tensor of a greyish, visually-neutral image
    """
    return tf.random.uniform(shape=shape) * scale_factor + brightness_factor


def save_image(
        image: np.ndarray, figscale: float, dpi: float, colormap: str, facecolor: str,
        save_dir: str, filename: str, save_format: str, figure_title: str,
        include_axis: bool, verbose: bool) -> None:
    """
    Plots and saves the given image according to the provided
    figure and filename parameters. Note the image must be a
    3-dimensional array, e.g. (28, 28, 1) for an MNIST image.

    Args:
        image: Image to save, must be a 3-dimensional array
        figscale: Base figure scale, passed to plt.figure.
            Note the dimensions of the figure are automatically determined,
            figscale is merely a multiplier with which to scale.
        dpi: Base dpi, passed to plt.figure
        colormap: Base colormap, passed to plt.figure.
            Note this is ignored for colour (RGB) images
        facecolor: Figure background colour, passed to fig.savefig
        save_dir: Directory to save the image in
        filename: Name of the file to save
        save_format: Format to save the image with
            (e.g. 'png','jpg','pdf)
        figure_title: Title to add to the figure.
            Note that no title is added if this is set to None or ''
        include_axis: Whether to include an axis
        verbose: Whether to include print statements
        
    """

    fig = plt.figure(figsize=(figscale, figscale), dpi=dpi)
    if figure_title is not (None or ''):
        fig.suptitle(figure_title)

    if image.shape[-1] == 1: # mono
        plt.imshow(image[..., 0], cmap=colormap)
    else:
        plt.imshow(image[...])

    if not include_axis:
        plt.axis('off')
    
    file_name = os.path.join(save_dir, f'{filename}.{save_format}')

    if verbose:
        print(f'Saving to {file_name}')

    fig.savefig(
        file_name, format=save_format,
        facecolor=facecolor, bbox_inches='tight'
    )


def euclidean_dist(a, b):
    """
    Returns the Euclidean distance between a and b.
    Curiously, this is around 10% faster than np.linalg.norm()
    """
    return ((b[0] - a[0])**2 + (b[1] - a[1])**2)**0.5


def obtain_reasonable_figsize(
        num_subplots: int, aspect_mode: str = 'uniform',
        orient: str = 'h') -> tuple[int]:
    """
    Automatically determines somewhat aesthetically pleasing figure dimensions
    given the required number of subplots to plot. Dimensions will be a tuple
    (height,width) of integer factors of the provided number of subplots.
 
    Args:
        num_subplots: The number of individual subplots to be plotted
            in the figure. Odd numbers are incremented by one.
        aspect_mode: One of 'uniform' or 'wide'. If set to uniform,
            it will try to make the dimensions as close to square as possible,
                e.g. 64 --> (8, 8), 72 --> (9, 8)
            If aspect_mode is instead set to 'wide', it will favour skinnier
            dimensions,
                e.g 64 --> (4, 16), 72 --> (6, 12)
        orient: One of 'h' or 'v'.
            If set to 'h', will return (height,width).
            If set to 'v', the order is reversed, i.e. (width, height).
            This is to make it easier to create tall figures upstream.

    Returns:
        Tuple of ints (height,width) of reasonable figure dimensions
            (i.e. rows, columns)
    """
    if num_subplots % 2 != 0:
        num_subplots += 1

    i = 2
    j = num_subplots // i
    ibest, jbest = i, j

    if aspect_mode == 'wide':
        while j > i*2:
            i += 1
            j = num_subplots // i
            if i*j != num_subplots:
                continue
            ibest, jbest = i, j
    else:
        while j > i:
            i += 1
            j = num_subplots // i
            if i*j != num_subplots:
                continue
            ibest, jbest = i, j

    if orient == 'h':
        return ibest, jbest
    return jbest, ibest