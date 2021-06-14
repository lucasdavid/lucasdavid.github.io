---
layout: post
title: Multilabel Learning Problems
excerpt: Dealing with ML classification problems that deal where samples aren't mutually disjointed.
first_p: |-
  In classic classification with networks, samples belong to a single class.
  We usually code this relationship using one-hot encoding: a label $i$ is
  transformed into a vector $[0, 0, ... 1.0, ..., 0, 0]$, where the $1$ is at
  the position $i$ of the vector.
toc: true
date: 2017-10-26 21:43:00
lead_image: /assets/images/posts/ml/multilabel/dataset.png
tags:
  - ML
  - neural networks
  - multi-label classification
---

<span class="display-6">In</span>
classic classification with networks, samples belong to a single class.
We usually code this relationship using one-hot encoding: a label $i$ is
transformed into a vector $[0, 0, ... 1.0, ..., 0, 0]$, where the $1$ is at
the position $i$ of the vector.

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

$$y(x)_i = \frac{e^x_i}{\sum_k e^x_k} $$

Which, in code, stays like this:
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
$1.0$. That's ideal when dealing with mutually disjointed classes, but what
about when that's not the case?

{% include figure.html
   src="/assets/images/posts/ml/multilabel/dataset.png"
   alt="Samples from the multi-label dataset 'Image Data for Multi-Instance Multi-Label Learning'."
   figcaption="Samples from the multi-label dataset 'Image Data for Multi-Instance Multi-Label Learning'. Note that some instances are associated with more than one label (mountains and sea or sea and sunset). Available at: <a href=\"https://www.lamda.nju.edu.cn/data_MIMLimage.ashx\">lamda.nju.edu.cn</a>" %}

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

$$E(y, p) = -\frac{1}{N} y \cdot \log p = -\frac{1}{N} \sum_i y_i \log p_i $$

So let *x* be any given sample from the dataset, associated with the class of index *k*.
From the equation above, we know all *yi* are 0, with exception of *yk*. Hence all terms *i != k*
of the sum will be equal to *0* and will not directly affect the value of the loss function
(the adjacent activation units *yi* s.t. *y != k* are still indirectly related through the softmax function).

We use here a new loss function, that accounts for the independency of each activation unit
of the networks's last layer:

$$\begin{eqnarray} 
E(y, p) &=& -\frac{1}{N} [y \cdot \log p + (1-y)\log(1-p)] \\
  &=& -\frac{1}{N} \sum_i y_i \log p_i + (1-y_i)\log(1-p_i)
\end{eqnarray}$$

From the figure above, we can see this *loss function* contains two terms.
Differently from the categorical cross-entropy, all units directly contribute to the summation through
one of the terms.

## Multi-label using Tensorflow

It's not commont for recent Tensorflow implementations to add the final layer (either softmax or sigmoid),
as it increases numeric instability when computing gradients. So you should declare the network as:


```python
x = Input(shape=(32, 299, 299, 3), name='inputs')

# using pretrained weights
y = tf.keras.applications.Xception(include_top=False)(x)
y = Dense(classes, name='predictions')(y)

model = Model(x, y, name='multilabel_disc')
```

The metrics need to be tweaked a bit as well, as they are expecting the output to be contained in the $[0, 1]$
interval. We re-declare them to either apply the sigmoid function within them or to expect the decision threshold
to be on top of the point $0$ (where the sigmoid outputs 50%):

```python
from tensorflow.python.keras.metrics import MeanMetricWrapper
from tensorflow.keras import losses, metrics, optimizers

def cosine_similarity_from_logits(y_true, y_pred, axis=-1):
    return losses.cosine_similarity(
        tf.cast(y_true, tf.float32),
        tf.nn.sigmoid(y_pred),
        axis)

model.compile(
    optimizer=optimizers.Adam(lr=Config.training.learning_rate),
    loss=losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        metrics.BinaryAccuracy(threshold=0.),
        cosine_similarity_from_logits,
    ]
)
```

Training happens exactly like we have previously seen:

```python
from tensorflow.keras import callbacks

try:
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.training.epochs,
        initial_epoch=0,
        callbacks=[
            callbacks.TerminateOnNaN(),
            callbacks.ModelCheckpoint(Config.log.tensorboard + '/weights.h5',
                                      save_best_only=True,
                                      save_weights_only=True,
                                      verbose=1),
            callbacks.ReduceLROnPlateau(patience=Config.training.reduce_lr_on_plateau_pacience,
                                        factor=Config.training.reduce_lr_on_plateau_factor,
                                        verbose=1),
            callbacks.EarlyStopping(patience=Config.training.early_stopping_patience, verbose=1),
            callbacks.TensorBoard(Config.log.tensorboard, histogram_freq=1)
        ],
        verbose=2);
except KeyboardInterrupt: print('interrupted')
else: print('done')
```
```shell
Epoch 1/200
25/25 - 28s - loss: 0.4791 - binary_accuracy: 0.7712 - cosine_similarity: -6.3827e-01 - val_loss: 0.3492 - val_binary_accuracy: 0.8850 - val_cosine_similarity: -7.7264e-01

Epoch 00001: val_loss improved from inf to 0.34916, saving model to /tf/logs/d:miml e:200 fte:0 b:32 v:0.3 m:inceptionv3 aug:False/weights.h5
...
Epoch 00123: val_loss did not improve from 0.12995
Epoch 124/200
25/25 - 13s - loss: 0.0369 - binary_accuracy: 0.9912 - cosine_similarity: -9.8772e-01 - val_loss: 0.1314 - val_binary_accuracy: 0.9583 - val_cosine_similarity: -9.3486e-01

Epoch 00124: val_loss did not improve from 0.12995
Epoch 00124: early stopping
done
```

### Testing The Model Trained
First, re-load the best weights found during the training procedure:
```py
disc.load_weights(Config.log.tensorboard + '/weights.h5')
```

Evaluation is pretty straight forward with the keras API:
```py
report = pd.DataFrame([
    disc.evaluate(train_ds, verbose=0),
    disc.evaluate(val_ds, verbose=0),
    disc.evaluate(test_ds, verbose=0),
],
index=['train', 'test', 'val'],
columns=disc.metrics_names)
```
```shell
 	loss 	binary_accuracy 	cosine_similarity
train 	0.049423 	0.986750 	-0.981672
test 	0.129952 	0.956000 	-0.932243
val 	0.131192 	0.958333 	-0.937034
```

It's always a good idea to see a few samples from the validation/test set,
in order to check for obvious inconsistencies:
```py
def plot_predictions(model, ds, take=1):
    figs, titles = [], []
    ls = Data.class_names

    plt.figure(figsize=(16, 16))
    for ix, (x, y) in enumerate(ds.take(take)):
        p = model.predict(x)
        p = tf.nn.sigmoid(p)
        y = tf.cast(y, tf.bool)
        pl = tf.cast(p > 0.5, tf.bool)
        figs.append(x.numpy().astype(int))
        
        titles.append([(f'y: {", ".join(Data.class_names[_y])}\n'
                        f'p: {", ".join(Data.class_names[_p])}')
                       for _y, _p in zip(y.numpy(), pl.numpy())])
    plot(np.concatenate(figs), titles=sum(titles, []), rows=6)
    plt.tight_layout()

plot_predictions(disc, test_ds)
```
{% include figure.html
   src="/assets/images/posts/ml/multilabel/preds.png"
   alt="Samples from the dataset's test split and its predictions."
   figcaption="Samples from the dataset's test split and its predictions." %}

Finally, it might be interesting to verify for the individual results for each label:
```py
def binary_accuracy_per_label(y_true, y_pred, threshold=0.5):
    threshold = tf.cast(threshold, y_pred.dtype)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, y_pred.dtype)
    
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32), axis=0)
    
def calc_acc_per_label(model, ds):
    batches = []
    for ix, (x, y) in enumerate(ds):
        p = model.predict(x)
        p = tf.nn.sigmoid(p)

        batches += [binary_accuracy_per_label(y, p)]
    
    return tf.reduce_mean(tf.stack(batches), axis=0).numpy(), len(batches)

score_per_label, batches = calc_acc_per_label(disc, test_ds)
print('Batches evaluated:', batches)
pd.DataFrame(list(zip(Data.class_names, score_per_label)), columns=['label', 'binary_accuracy'])
```
```shell
Batches evaluated: 19

label 	 binary_accuracy
desert  	0.968750
mountains 	0.958333
sea 	 	0.935307
sunset  	0.973684
trees 	 	0.956689
```

## Final Considerations
In this post, I casually presented the formulation for multi-label classification problems and a way to solve them using networks. Except from the categorical cross-entropy loss function and metrics --- which model mutually disjointed problems, where the classification output is expected to sum to $1$ ---, much of the code that we have learned so far can be reused here. In any case, just a few tweaks can be made in order to bring it home.

Finally, I leave you with following questions as food for thought: if two classes (sea and sunset, for example) always appear together in all of the images, is it possible to achieve 100% test accuracy for both classes without actually
learning how to differentiate them?
If yes, then is there a maximum amount of correlation between two classes such that violating this threshold would create a confusion in the model?
