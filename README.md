# layervis
A simple suite of utilities for visualising convolutional neural networks.
***

`layervis` contains several functions intended to be used to visualise convolutional neural networks (TensorFlow+Keras).

The code currently supports the following visualisation techniques:

* **Feature maps**
* **Filter weights** (actual kernel weights)
* **Filter patterns** (filter responses visualised with gradient ascent, based on [this keras.io example](https://keras.io/examples/vision/visualizing_what_convnets_learn/))
* **Class activation maps**
* **Saliency maps**
* **Image atlases**

with plans for several more.

### Examples
An example gallery Jupyter notebook will be included soon. <br> Here are some example snippets for typical usage:

#### Feature maps

```python
from layervis.features import FeatureMaps

model = keras.models.load_model('mnist_cnn.h5')

# plot feature maps for the first convolutional layer
fmaps = FeatureMaps(model=model)
fig = fmaps.plot_feature_map(layer='conv2d',feature_input=test_image)

# plot all feature maps (includes conv + pooling + relu layers by default)
fmaps.plot_feature_maps(feature_input=test_image)

# only plot feature maps for Conv2D layers)
fmaps = FeatureMaps(model=model, valid_layers=(layers.Conv2D))
```

#### Filters

```python
from layervis.filters import FilterWeights, FilterVis

# Initialise a filter visualisation object, applying a Gaussian blur of size 5 every 5 epochs, with an image decay of 0.75 
fvis = FilterVis(model=model, blur_size=5, blur_freq=5, image_decay=0.75)

# Visualise all filter patterns
fvis.plot_all_patterns(num_iterations=30, learning_rate=1)
```

#### Class activation maps
```python
from layervis.gradients import GradCAM

gcam = GradCAM(model=model)
fig = gcam.plot_heatmap(image=test_image, pred_index=0, layer='conv2d_3')
```

#### Saliency maps & class models
```python
from layervis.gradients import GradientSaliency, ClassModel
gs = GradientSaliency(model=model)
gs.plot_saliency_maps(test_image)

cm = ClassModel(model=model)
cm.plot_class_models()
```

#### Image atlases
```python
from layervis.atlases import plot_image_atlas

fig = plot_image_atlas(images=X_train, embedding=embedding, nx=12, ny=12, max_image_dist=0.05)
```
