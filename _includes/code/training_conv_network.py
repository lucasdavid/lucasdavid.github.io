import keras
from keras import Model, Input, callbacks
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

from sacred import Experiment

ex = Experiment('training-conv-network')


@ex.config
def my_config():
    epochs = 20
    batch_size = 128
    num_classes = 10
    valid_size = .25
    early_stopping_patience = 5
    optimizer = 'adam'
    ckpt = './weights.hdf5'


@ex.automain
def main(batch_size, num_classes, optimizer, epochs, valid_size, ckpt,
         early_stopping_patience):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print('image shapes:', x_train.shape[1:])

    x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
    x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=valid_size)

    # one-hot encode train and test
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x = Input(shape=x_train.shape[1:])
    y = Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv2d_1')(x)
    y = Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv2d_2')(y)
    y = MaxPooling2D(2, name='maxpool_2')(y)
    y = Conv2D(64, kernel_size=5, activation='relu', name='conv2d_3')(y)
    y = Conv2D(64, kernel_size=5, activation='relu', name='conv2d_4')(y)
    y = Flatten()(y)
    y = Dense(1024, activation='relu', name='fc1')(y)
    y = Dense(1024, activation='relu', name='fc2')(y)
    y = Dense(num_classes, activation='softmax', name='predictions')(y)

    model = Model(inputs=x, outputs=y)
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

    score = model.evaluate(x_test, y_test, verbose=0)
    print('test loss:', score[0])
    print('test accuracy:', score[1])
