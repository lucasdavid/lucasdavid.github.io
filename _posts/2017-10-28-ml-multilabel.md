---
layout: post
title: Multilabel Learning Problems
excerpt: Dealing with ML classification problems that deal where samples aren't mutually disjointed.
date: 2017-10-26 21:43:00
---

# Introduction

In classic classification with networks, samples belong to a single class.
We usually code this relationship using one-hot encoding: a label `i` is
transformed into a vector `[0, 0, ... 1.0, ..., 0, 0]`, where the `1` is at
the position `i` of the vector.



We also define the networks to end with a `softmax` layer with the same number
of units as classes, where the activations are translated intro probabilities:

<center>
  <figure class="equation">
    <img src="/assets/ml/eq-softmax.png"
         alt="Softmax function" />
  </figure>
</center>

Because of the normalization factor, the probabilities will always sum to
`1.0`. That's ideal when dealing with mutually disjointed classes, but what
about when that's not the case?

Textbook ML says we have to use use sigmoid

```python
import numpy as np

from sklearn import datasets, metrics
from keras import Sequential, layers

from sacred import Experiment

ex = Experiment('multi-label-training')


@ex.config
def my_config():
  input_dim = 768
  batch_size = 128
  n_classes = 1001
  test_size = 1 / 3
  random_state = 42


def one_hot_encode(y, n_classes=None):
  if not n_classes: n_classes = len(np.unique(y))
  encoded = np.zeros(len(y), n_classes)
  encoded[y] = 1.
  return encoded


@ex.automain
def main(input_dim, batch_size, n_classes, test_size, random_state):
  dataset = datasets.load_digits(n_classes)
  x, y = dataset.x, one_hot_encode(dataset.target, n_classes)

  x, x_test, y, y_test = train_test_split(x, y,
                                          test_size=test_size,
                                          random_state=random_state)

  model = Sequential([
    layers.Dense(2048, activation='relu', input_dim=input_dim),
    layers.Dense(2048, activation='relu'),
    layers.Dense(n_classes, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  model.fit(x, y, batch_size=batch_size)
  p_test = model.predict(x_test)

  print('roc:', metrics.roc_curve(y_test, p_test))
  print('accuracy:', metrics.accuracy(y_test, p_test > .5))

  print('y:', y_test[:5])
  print('p:', p_test[:5])
```
