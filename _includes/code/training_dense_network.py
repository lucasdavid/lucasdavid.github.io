import keras
from keras import callbacks
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import InputLayer, Dense
from sklearn.model_selection import train_test_split

from sacred import Experiment

ex = Experiment('training-dense-network')


@ex.config
def my_config():
    epochs = 20
    batch_size = 128
    num_classes = 10
    valid_size = .25
    early_stopping_patience = 5
    optimizer = 'SGD'
    ckpt = './optimal_weights.hdf5'


@ex.automain
def main(batch_size, num_classes, optimizer, epochs, valid_size, ckpt,
         early_stopping_patience):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=valid_size)

    # one-hot encode train and test
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential([
      InputLayer(shape=[768]),
      Dense(1024, activation='relu', name='fc1'),
      Dense(1024, activation='relu', name='fc2'),
      Dense(num_classes, activation='softmax', name='predictions')
    ])
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    try:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  validation_data=(x_valid, y_valid),
                  callbacks=[
                      callbacks.EarlyStopping(patience=early_stopping_patience),
                      callbacks.ModelCheckpoint(ckpt, save_best_only=True, verbose=True)
                  ])
    except KeyboardInterrupt:
        print('interrupted')
    else:
        print('done')

    print('reloading optimal weights...')
    model.load_weights(ckpt)

    score = model.evaluate(x_test, y_test)
    print('test loss:', score[0])
    print('test accuracy:', score[1])
