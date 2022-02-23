---
layout: post
title: Segmentation and Detection
excerpt: "A comparison between edge-based segmentation, Felzenszwalb's method and morphological segmentation."
first_p: |-
  Following the pattern of my previous posts,
  this one is based on an assignment submitted to the class of 2021/1 of course
  Introduction to Image Processing (MO443) at Universidade Estadual de Campinas.
  Its goal is to apply segmentation algorithms over images and extract characteristics of the objects using
  Python programming language and assess its results.
  We compare strategies such as edge-based segmentation, Felzenszwalb's method and morphological segmentation.
toc: true
date: 2021-07-16 22:37:00
lead_image: /assets/images/posts/cv/segmentation/cover.webp
tags:
  - Computer Vision
  - Segmentation
---

<span>Following the pattern</span> of my previous posts,
this one is based on an assignment submitted to the class of 2021/1 of course
Introduction to Image Processing (MO443) at Universidade Estadual de Campinas.
Its goal is to apply segmentation algorithms over images and extract characteristics of the objects using
Python programming language and assess its results.

{% include posts/collapse-btn.html id="csetup" text="show setup code" %}
```python
from math import ceil
from functools import partial as ftpartial

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import os
import skimage.io
import skimage.filters
import skimage.feature
import skimage.segmentation

from skimage.color import label2rgb

import seaborn as sns

sns.set()

C = sns.color_palette("Set3")

def partial(func, *args, **kwargs):
    partial_func = ftpartial(func, *args, **kwargs)
    partial_func.__name__ = (f"{func.__name__} {' '.join(map(str, args))}"
                             f" {' '.join((f'{a}={b}' for a, b in kwargs.items()))}")
    return partial_func

def visualize(
    image,
    title=None,
    rows=2,
    cols=None,
    cmap=None,
    figsize=(12, 6)
):
    if image is not None:
        if isinstance(image, list) or len(image.shape) > 3:  # many images
            fig = plt.figure(figsize=figsize)
            cols = cols or ceil(len(image) / rows)
            for ix in range(len(image)):
                plt.subplot(rows, cols, ix+1)
                visualize(image[ix],
                          cmap=cmap,
                          title=title[ix] if title is not None and len(title) > ix else None)
            plt.tight_layout()
            return fig

        if image.shape[-1] == 1: image = image[..., 0]
        plt.imshow(image, cmap=cmap)
    
    if title is not None: plt.title(title)
    plt.axis('off')

def plot_segments(g, s, p, alpha=0.8, ax=None, linewidth=1):
    ax = ax or plt.gca()
    ax.imshow(label2rgb(s, image=g, bg_label=0, alpha=alpha, colors=C, bg_color=(1,1,1)))
    ax.axis('off')

    for i, region in p.iterrows():
        minr, minc, maxr, maxc = region['bbox-0'], region['bbox-1'], region['bbox-2'],region['bbox-3']
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False,
                             edgecolor=C[i % len(C)], linewidth=linewidth)
        ax.add_patch(rect)

        plt.text(region['centroid-1'], region['centroid-0'], int(region.label))

def plot_detection_boxes(p, ax=None, linewidth=1):
    if ax is None:
        ax = plt.gca()

    H, W, C = p['image'].shape
    bboxes = p['objects']['bbox'].numpy()
    labels = p['objects']['label'].numpy()
    
    for b, l in zip(bboxes, labels):
        minr, minc, maxr, maxc = b
        minr, minc, maxr, maxc = minr*H, minc*W, maxr*H, maxc*W
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False,
                             edgecolor=C[l % len(C)], linewidth=linewidth)
        ax.add_patch(rect)
        ax.text(minc,minr, int2str(l))
```
{: class="collapse" id="csetup"}

## Segmentation and Detection of Simple Geometric Shapes

Using scikit-image, multiple segmentation strategies are available.
For instance, one can extract borders {% cite torre4767769 %} and label the connected regions;
or find central regions and apply label expansion methods such as Watershed {% cite strahler1957quantitative %}.

In the following sections, we exemplify two strategies (edge-based segmentation and the Felzenszwalb's method)
over simple geometric shapes, illustrated in the figure below.

```python
image_files = sorted(os.listdir(Config.data.path))
images = [skimage.io.imread(os.path.join(Config.data.path, f)) for f in image_files]
grays = [skimage.color.rgb2gray(i) for i in images]

visualize([*images, *grays], image_files, cmap='gray');
```
{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_7_0.webp.webp"
   alt="Simple colored geometric shapes."
   caption="Simple colored geometric shapes." %}


### Edge-Based Segmentation

Scikit-image implements many of the border extraction methods available in the literature, such as Sobel, Prewitt and Scharr {% cite torre4767769 %}.
The can be trivially used as described below.

```python
%%time

N = 3

methods = [
  skimage.filters.sobel,
  skimage.filters.roberts,
  skimage.filters.prewitt,
  skimage.filters.farid,
  skimage.filters.scharr,
  partial(skimage.feature.canny, sigma=0.),
  partial(skimage.feature.canny, sigma=0.1),
  partial(skimage.feature.canny, sigma=3.),
]

segments = [[m(g) for g in grays] for m in methods]
```
```shell
CPU times: user 381 ms, sys: 4.75 ms, total: 386 ms
Wall time: 389 ms
```

{% include posts/collapse-btn.html id="cv1" %}
```python
method_names = [n.__name__ for n in methods]

visualize(grays, image_files, rows=1, cmap='gray');

for n, s in zip(method_names, segments):
  visualize(s, [n], rows=1, cmap='gray_r');
```
{: class="collapse" id="cv1"}

<div id="carouselBorders"
     class="carousel slide carousel-dark"
     data-bs-ride="carousel"
     alt="Results from multiple border extraction methods over images containing simple geometric shapes.">
  <div class="carousel-inner">
    <div class="carousel-item active"><img src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_10_1.webp.webp" class="d-block w-100"></div>
    <div class="carousel-item"><img src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_10_2.webp.webp" class="d-block w-100"></div>
    <div class="carousel-item"><img src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_10_3.webp.webp" class="d-block w-100"></div>
    <div class="carousel-item"><img src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_10_4.webp.webp" class="d-block w-100"></div>
    <div class="carousel-item"><img src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_10_5.webp.webp" class="d-block w-100"></div>
    <div class="carousel-item"><img src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_10_6.webp.webp" class="d-block w-100"></div>
    <div class="carousel-item"><img src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_10_7.webp.webp" class="d-block w-100"></div>
    <div class="carousel-item"><img src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_10_8.webp.webp" class="d-block w-100"></div>
  </div>
  <button class="carousel-control-prev" type="button" data-bs-target="#carouselBorders"  data-bs-slide="prev">
    <span class="carousel-control-prev-icon" aria-hidden="true"></span>
    <span class="visually-hidden">Previous</span>
  </button>
  <button class="carousel-control-next" type="button" data-bs-target="#carouselBorders"  data-bs-slide="next">
    <span class="carousel-control-next-icon" aria-hidden="true"></span>
    <span class="visually-hidden">Next</span>
  </button>
</div>

The figure above contains a comparison between the available border extraction methods. Roberts seems to produce the thinnest borders amongst all methods, while Farid produces thicker ones. Sobel, Prewitt and Scharr have similar results. Finally, Canny with a small $\sigma$ parameter (used to set the standard deviation of the Gaussian filter) results in accurate borders for simple shapes. However, the original gray intensity of the object's border is lost, with all borders presenting the same gray intensity. Canny with a very large sigma parameter ($\sigma=3$) still manages to extract borders from simple objects, though corners become round.

With closed borders in hands, extracted from one of the methods above,
we can simply fill the wholes to separate what's background and foreground:
```python
from scipy import ndimage

segments = [skimage.filters.roberts(g) for g in grays]
filled = [ndimage.binary_fill_holes(s) for s in segments]

visualize(filled, rows=1, cmap='gray_r');
```
{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_12_0.webp.webp"
   alt="The figure displays three binary images, in which the geometric shapes are filled with the color black, and the background is shown in white."
   caption="Binary filling of the borders previously extracted." %}

Notice we used the color map `gray_r`, indicating objects were filled with the value `1` while the background is represented with the value `0`.
As each object is now represented by a distinct connected component, we can extract properties using functions in the `skimage.measure` module:

```python
%%time

from skimage.measure import label
from skimage.measure import regionprops_table

properties = ('label', 'area', 'convex_area', 'eccentricity',
              'solidity', 'perimeter', 'centroid', 'bbox')

segments = [label(f) for f in filled]
props = [pd.DataFrame(regionprops_table(s, properties=properties))
         for s in segments]
```
```shell
CPU times: user 129 ms, sys: 4.74 ms, total: 133 ms
Wall time: 137 ms
```

{% include posts/collapse-btn.html id="cv2" %}
```python
def plot_all_segments(grays, segments, props):
  plt.figure(figsize=(12, 3))
  for ix, (i, s, p) in enumerate(zip(grays, segments, props)):
    plt.subplot(1, 3, ix+1)
    plot_segments(i, s, p)
  plt.tight_layout()

  plt.figure(figsize=(12, 3))
  for ix, (i, s, p) in enumerate(zip(grays, segments, props)):
    plt.subplot(1, 3, ix+1)
    plt.hist(p.area)
  plt.tight_layout()

plot_all_segments(grays, segments, props)
```
{: class="collapse" id="cv2"}

    
{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_14_0.webp.webp"
   caption="Segmentation of the objects in the three original images using edge detection and binary fill of wholes. Segmentation maps are overlayed with objects, while the bounding boxes used to delimit each object was extracted as a property in <code>regionprops_table</code>." %}

{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_14_1.webp.webp"
   caption="Histograms of areas detected from objects in each input image." %}

A simple way to count objects based on their overall area is to use histograms.
Fig. 5 illustrate this application over each image. In the first, forms are separated into two similar groups:
small(close to 500), and large (approx. 2,500).
A more diverse set of areas are present in the second image, producing a more disperse histogram.
In the last image, three objects have very large area (1, 3 and 8).
Object 6 has an intermediate size, while the remaining objects are small.


### Segmentation using Felzenszwalb's Method

This segmentation method was described in "Efficient graph-based image segmentation", by Felzenszwalb et. al. {% cite felzenszwalb2004 %}
It consists of representing the pixel-color intensity in an image as a grid and find $N$ partitions representing similarity.
As the algorithm progresses, the closest partitions (with respect to a connection predicate) are iteratively merged.
The final number of partitions is minimum (optimal) while still respecting the connection predicate.

The following code exemplifies the application of Felzenszwalb's segmentation method implemented in scikit-image over images:
```python
segments = [skimage.segmentation.felzenszwalb(g, scale=1e6, sigma=0.1, min_size=10) for g in grays]
props = [pd.DataFrame(regionprops_table(s, properties=Config.properties)) for s in segments]

props[0].head().round(2)
```

<div class="table-responsive"><table class="dataframe table table-hover">
  <thead>
    <tr>
      <th></th>
      <th>label</th>
      <th>area</th>
      <th>convex_area</th>
      <th>eccentricity</th>
      <th>solidity</th>
      <th>perimeter</th>
      <th>centroid-0</th>
      <th>centroid-1</th>
      <th>bbox-0</th>
      <th>bbox-1</th>
      <th>bbox-2</th>
      <th>bbox-3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2352</td>
      <td>2352</td>
      <td>0.20</td>
      <td>1.0</td>
      <td>190.0</td>
      <td>29</td>
      <td>202</td>
      <td>5</td>
      <td>179</td>
      <td>54</td>
      <td>227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2352</td>
      <td>2352</td>
      <td>0.20</td>
      <td>1.0</td>
      <td>190.0</td>
      <td>29</td>
      <td>422</td>
      <td>5</td>
      <td>399</td>
      <td>54</td>
      <td>447</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>272</td>
      <td>272</td>
      <td>0.34</td>
      <td>1.0</td>
      <td>62.0</td>
      <td>29</td>
      <td>139</td>
      <td>21</td>
      <td>132</td>
      <td>38</td>
      <td>148</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>272</td>
      <td>272</td>
      <td>0.34</td>
      <td>1.0</td>
      <td>62.0</td>
      <td>60</td>
      <td>92</td>
      <td>53</td>
      <td>84</td>
      <td>69</td>
      <td>101</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2304</td>
      <td>2304</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>188.0</td>
      <td>107</td>
      <td>29</td>
      <td>84</td>
      <td>6</td>
      <td>132</td>
      <td>54</td>
    </tr>
  </tbody>
</table>
</div>

Fig. 6 illustrates the Felzenszwalb's segmentation results over images containing simple geometric shapes. Little difference can be perceived from the edge-based segmentation strategy.
I used the parameters `scale=1e6, sigma=0.1, min_size=10` to detect the simple geometric shapes, which enforces large objects (high `scale`) and low Gaussian filtering (low `sigma`). Decreasing `scale` had little influence in the results of the last sample (containing 9 objects). However, this resulted in an incorrect segmentation for the objects in the first and second images, in which the objects and their borders were being classified as distinct coinciding objects.
I found `min_size` to be of little importance in these examples, as it is applied as a post-processing stage and can be disregarded when `scale` is sufficiently large.

```python
plot_all_segments(grays, segments, props)
```
{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_20_0.webp.webp"
   caption="Segmentation of the objects in the three original images using the Felzenszwalb's method." %}

Similar results are obtained with Felzenszwalb, with respect to the area of objects:
{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_20_1.webp.webp"
   caption="Histograms of areas detected from objects in each scenario." %}


## BCCD Dataset

To demonstrate in a more realistic scenario, I opted to use the [BCCD Dataset](https://www.tensorflow.org/datasets/catalog/bccd).
This set comprises small-scale image samples used for blood cell detection.

{% include posts/collapse-btn.html id="cv2" %}
```python
import tensorflow as tf
import tensorflow_datasets as tfds

(train,), info = tfds.load('bccd', split=['train'], with_info=True, shuffle_files=False)

int2str = info.features['objects']['label'].int2str

bccd_samples = list(train.take(10))
bccd_images = [s['image'].numpy() for s in bccd_samples]
bccd_grays = [skimage.color.rgb2gray(i) for i in bccd_images]
bccd_cells = sum([e['objects']['label'].shape.as_list() for e in bccd_samples], [])


fig = visualize([s['image'] for s in bccd_samples], rows=2, figsize=(16, 6))
axes = fig.get_axes()

for ax, p in zip(axes, bccd_samples):
    plot_detection_boxes(p, ax)
```
{: class="collapse" id="cv2"}

{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_22_0.webp.webp"
   caption="A few samples from the BCCD dataset. Cells are annotated in RBC (red blood cells), WBC (white blood cells) and Platelets." %}


### Segmentation using Felzenszwalb's Method
Edge-detection methods produce poor results over samples in the BCCD dataset, as they are quite more complex
than the ones previously used --- presenting salt-and-pepper noise and much more densely distributed objects ---.
Thus, I once again employed Felzenszwalb's method.

We first "search" for the best combination of parameters to use in this set.

```python
from sklearn.model_selection import ParameterGrid

def search_felzenszwalb(image, params=None, figsize=(16, 13)):
  if not params:
    params = ParameterGrid({
      'scale': [1000, 10000, 10000],
      'min_size': [75, 150, 300],
    })

  plt.figure(figsize=figsize)

  for ix, param in enumerate(params):
    segment = skimage.segmentation.felzenszwalb(image, sigma=0.01, **param)
    prop = pd.DataFrame(regionprops_table(segment, properties=Config.properties))

    ax = plt.subplot(ceil(len(params) / 3), 3, ix+1)
    plot_segments(image, segment, prop, alpha=0.2, ax=ax)
    plt.title(' '.join((':'.join(map(str, e)) for e in param.items())))

  plt.tight_layout()

search_felzenszwalb(bccd_images[3])
```
{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_25_0.webp.webp"
   caption="Effect of the parameters <code>scale</code> and <code>min_size</code> over segmentation." %}

The segmentation results of samples in the BCCD dataset are shown in Fig. 10. The method has correctly segmented many of the blood cells in some samples (such as in the third and fifth images). However, a few drawbacks are noticeable as well: this method is strongly affected by small grains, and will often recognize small microorganisms that were captured by the microscope while photographing the blood cells. Furthermore, cells that were smashed together seem to have been classified as a single object (first and eighth images).

```python
images = bccd_images

segments = [skimage.segmentation.felzenszwalb(i, scale=1e3, sigma=0.01, min_size=150) for i in images]
props = [pd.DataFrame(regionprops_table(s, properties=Config.properties)) for s in segments]
```

{% include posts/collapse-btn.html id="cv3" %}
```python
def plot_all_segments_and_bboxes(images, segments, props):
  plt.figure(figsize=(16, 5))

  for ix, (image, label, prop) in enumerate(zip(images, segments, props)):
    ax = plt.subplot(2, 5, ix+1)
    plot_segments(image, label, prop, alpha=0.2, ax=ax)

  plt.tight_layout()

plot_all_segments_and_bboxes(images, segments, props)
```
{: class="collapse" id="cv3"}

{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_28_0.webp.webp"
   caption="Segmentation results over samples in the BCCD dataset. " %}

{% include posts/collapse-btn.html id="cv4" %}
```python
def histogram_objects(props, by='area', percentile=None):
  plt.figure(figsize=(16, 6))

  for ix, prop in enumerate(props):
    plt.subplot(2, ceil(len(props)/2), ix +1)

    s = prop[by]
    if percentile: s = s[s < np.percentile(s, 99)]
    sns.histplot(s)

  plt.tight_layout()

histogram_objects(props, percentile=95)
```
{: class="collapse" id="cv4"}

{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_29_0.webp.webp"
   caption="Histogram of areas detected from objects in samples in the BCCD dataset." %}

Finally, Felzenszwalb's method will indistinguishably segment the background sections of the image into regions, as these sections also present color information. It is therefore necessary to employ heuristics that discriminate foreground/background in order to separate it.

#### Post-Processing: Filtering Objects by Overall Size

To remedy the grain noise being detected by Felzenszwalb's, we can sub-select objects based on their properties.
A simple selection could be defined by its overall area:

```python
RBC_AREA = (1e3, 2e4)

props_rbc = [p[(p.area > RBC_AREA[0]) & (p.area < RBC_AREA[1])] for p in props]
segments_rbc = [s*np.isin(s, p.label) for s, p in zip(segments, props_rbc)]

plot_all_segments_and_bboxes(images, bccd_samples, segments_rbc, props_rbc)
```
{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_31_0.webp.webp"
   caption="Segmentation results over samples in the BCCD dataset. Only objects with area between $1e3$ and $2e4$ are shown." %}


```python
histogram_objects(props_rbc)
```
{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_32_0.webp.webp"
   caption="Histogram of areas detected from objects in the samples in the BCCD dataset. Only objects with area between $1e3$ and $2e4$ account for this statistics." %}

#### Pre-Processing: Filtering Noise

The noise present in the samples correspond to dirt present in the lab sample, and do not contribute to the detection of blood cells.
We can use [filters.rank.mean_bilateral](https://scikit-image.org/docs/dev/api/skimage.filters.rank.html#skimage.filters.rank.mean_bilateral)
to remove this noise before applying segmentation, possibly reducing the complexity and helping Felzenszwalb's to focus on the cells:

```python
from skimage import color, img_as_ubyte, img_as_float
from skimage.morphology import disk
from skimage.filters.rank import mean_bilateral

selem20 = disk(20)

def mean_bilateral_fn(image, selem, s0=10, s1=10):
  image = color.rgb2gray(image)
  image = img_as_ubyte(image).astype('uint16')
  image = mean_bilateral(image, selem, s0=s0, s1=s1)

  return image / 255.

grays_bilateral = [mean_bilateral_fn(i, selem20) for i in images]

visualize(grays[:5] + grays_bilateral[:5], figsize=(16, 5));
```
{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_36_0.webp.webp"
   caption="Application of the <code>mean_bilateral</code>. From top to bottom: (a) the original samples in the BCCD dataset, (b) the corresponding sample cleaned from noise." %}

Fig. 14 illustrates the application of `mean_bilateral` over samples in the BCCD dataset.
We observe that much of the noise is either removed or attenuated.

As samples have changed drastically, we search parameters once again. Fig. 15 illustrates
the effect of varying `scale` and `min_size` over a sample in the BCCD dataset. The combination
`scale=1e4` and `min_size=150` seems to produce reasonable results for this sample.

```python
search_felzenszwalb(grays_bilateral[4])
```
{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_37_0.webp.webp"
   caption="Effect of the parameters <code>scale</code> and <code>min_size</code> over segmentation, after preprocessing with <code>mean_bilateral</code> was performed." %}

We can now apply to each sample in our subset:

```python
# Pre-processing
selem20 = disk(20)
grays_bilateral = [mean_bilateral_fn(i, selem20) for i in images]

# Segmentation
segments = [skimage.segmentation.felzenszwalb(i, scale=1e4, sigma=0.01, min_size=150)
            for i in grays_bilateral]
props = [pd.DataFrame(regionprops_table(s, properties=Config.properties))
         for s in segments]

# Post-processing
RBC_AREA = (1e3, 2e4)

props_rbc = [p[(p.area > RBC_AREA[0]) & (p.area < RBC_AREA[1])] for p in props]
segments_rbc = [s*np.isin(s, p.label) for s, p in zip(segments, props_rbc)]

plot_all_segments_and_bboxes(images, bccd_samples, segments_rbc, props_rbc)
```
{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_40_0.webp.webp"
   caption="Segmentation results over samples BCCD dataset, preprocessing samples with <code>mean_bilateral</code> and post-processing them by sub-selecting objects within a reasonable overall area." %}

```python
histogram_objects(props_rbc)
```
{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_41_0.webp.webp"
   caption="Histogram of areas for each image in the BCCD dataset." %}

From Fig. 16 and Fig. 17 show our cleaned segmentation results.
The algorithm seems much more focused on cells now!

### Morphological Boundaries

Going in a different direction, morphology can also be used to segment objects in images. I employed a second segmentation strategy, which consists of applying the Ostu's threshold [2] to separate objects from background and employ morphology to close small holes. The cleaned mask can then be labeled according to its connected regions.

The code below describes the necessary steps to implement this strategy. Notice that the binary mask that will serve as input to the morphological operations is created by retaining the positions in which pixel intensity is below the threshold (`image < t`), as the objects in this problem present a lower pixel intensity than the background's. This differs from the examples in scikit-image documentation, in which silver coins were being compared to a dark background.

```python
from skimage.measure import label
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, square

def morphological_segmentation(image, so=square(5)):
  bw = image < threshold_otsu(image)

  return label(opening(closing(bw, so),  so))
```

```python
%%time

images = bccd_grays
segments = [morphological_segmentation(image) for image in images]
props = [pd.DataFrame(regionprops_table(s, properties=Config.properties)) for s in segments]
```
```shell
CPU times: user 1.13 s, sys: 520 ms, total: 1.65 s
Wall time: 1.11 s
```

```python
plot_all_segments_and_bboxes(images, bccd_samples, segments, props)
```
{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_45_0.webp.webp"
   caption="Segmentation results over samples in the BCCD dataset using Morphological operators." %}

```python
histogram_objects(props)
```
{% include figure.html
   src="/assets/images/posts/cv/segmentation/2021-07-16-segmentation_46_0.webp.webp"
   caption="Histograms of areas detected in objects in each image." %}

The segmentation results over samples in the BCCD dataset are shown in Fig. 18 and Fig. 19. Through inspection, we notice this strategy is robust against small grains in the image, and correctly segments red blood cells presenting light-shaded interiors. It also automatically ignores the background and does not interpret its regions as new objects --- through the usage of Otsu's method {% cite otsu4310076 %} ---. Notwithstanding, long cell chains (which are smashed against each other) are incorrectly segmented as a single object, regardless of their color differences (an effect observed in the first, second, third, fifth, ninth and tenth images).

## References

{% bibliography --file Computer-Vision/segmentation --cited_in_order %}
