---
layout: post
title: Multilabel Learning Problems
excerpt: Dealing with ML classification problems that deal where samples aren't mutually disjointed.
first_p: |-
  In classic classification with networks, samples belong to a single class.
  We usually code this relationship using one-hot encoding: a label <code>i</code> is
  transformed into a vector <code>[0, 0, ... 1.0, ..., 0, 0]</code>, where the <code>1</code> is at
  the position <code>i</code> of the vector.
date: 2017-10-26 21:43:00
lead_image: /assets/ml/deep/dog-cat.jpeg
tags:
  - ML
  - classification
---

In classic classification with networks, samples belong to a single class.
We usually code this relationship using one-hot encoding: a label `i` is
transformed into a vector `[0, 0, ... 1.0, ..., 0, 0]`, where the `1` is at
the position `i` of the vector.

```python
import numpy as np
from keras.utils import to_categorical

samples = 8096
features = 128
classes = 1000

data, target = (np.random.rand(samples, features),
                np.random.randint(classes, size=(samples, 1)))
target_c = to_categorical(target)
```

We also define the networks to end with a `softmax` layer with the same number
of units as classes, where the activations are translated intro probabilities:

<center>
  <figure class="equation">
    <img src="/assets/ml/eq-softmax.png"
         alt="Softmax function" />
  </figure>
</center>

```python
from keras import Model, Input
from keras.layers import Dense

x = Input(shape=[299, 299, 3])
y = Dense(1024, activation='relu')(x)
y = Dense(1024, activation='relu')(y)
y = Dense(classes, activation='softmax')(y)

model = Model(x, y)
model.compile(optimizer='adam', loss='categorical_crossentropy')
```

Because of the normalization factor, the probabilities will always sum to
`1.0`. That's ideal when dealing with mutually disjointed classes, but what
about when that's not the case?

First, we must convert `target` to a binary encoding:

```python
def encode(y, classes=None):
  if classes is None: classses = len(np.unique(y))
  encoded = np.zeros(len(y), classes)
  encoded[y] = 1.
  return encoded

yc = encode(y, classes=classes)
```

This creates a map very much like the one-hot.
For example, let's say there's 5 possible classes:
dog, mammal, primate, feral and domestic.

- the labels `dog`, `mammal` and `domestic`, associated with sample `x0`, would be
  encoded as `(1., 1., 0., 0., 1.)`
- the labels `primate` and `feral`, associated with sample `x1`, would be encoded
  as `(0., 0., 1., 1., 0.)`

Softmax also needs to go. Textbook ML says we can use `sigmoid`:

```python
model.pop()
model.add(Dense(classes, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')
```

Because `sigmoid`'s shape, probabilities are no longer normalized between
the different activation units. This means that `model` might output an
entire vector of ones `(1., 1., ..., 1.)` or zeros `(0., 0., ..., 0.)` -- even
though such situations are unlikely to happen.

Finally, we must replace our `categorical_crossentropy` loss function by the
`binary_crossentropy`. To ensure we are up with the base concepts, this is the
categorical cross-entropy function definition once again:

<center>
{% include figure.html
   src="/assets/ml/eq-crossentropy.png"
   alt="A definição da equação 'categorical cross-entropy loss': 'E(y, p) = -(y log(p))/N'"
   figcaption="A definição da equação 'categorical cross-entropy loss': 'E(y, p) = -(y log(p))/N'" %}
</center>

So let *x* be any given sample from the dataset, associated with the class of index *k*.
From the equation above, we know all *yi* are 0, with exception of *yk*. Hence all terms *i != k*
of the sum will be equal to *0* and will not directly affect the value of the loss function
(the adjacent activation units *yi* s.t. *y != k* are still indirectly related through the softmax function).

We use here a new loss function, that accounts for the independency of each activation unit
of the networks's last layer:
<center>
{% include figure.html
   src="/assets/ml/eq-binary-crossentropy.png"
   alt="A definição da equação 'binary cross-entropy loss': 'E(y, p) = -[y log(p) + (1-y) log(1 - p)]/N'"
   figcaption="A equação da 'binary cross-entropy loss': 'E(y, p) = -[y log(p) + (1-y) log(1 - p)]/N'" %}
</center>

From the figure above, we can see this *loss function* contains two terms.
Differently from the categorical cross-entropy, all units directly contribute to the summation through
one of the terms.

## A Practical Example

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

def encode(y, classes=None):
  if classes is None: classses = len(np.unique(y))
  encoded = np.zeros(len(y), classes)
  encoded[y] = 1.
  return encoded

@ex.automain
def main(input_dim, batch_size, n_classes, test_size, random_state):
  dataset = datasets.load_digits(n_classes)
  x, y = dataset.x, encode(dataset.target, n_classes)

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
