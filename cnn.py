import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.python.keras.layers import Dense
import tensorflow.python.keras as keras
import tensorflow as tf

DATA_PATH = 'image_test_tst.json'
SAVED_MODEL_PATH = 'test_model_tst.h5'
LEARNING_RATE = 0.0001
EPOCHS = 15
BATCH_SIZE = 2
NUM_IMG_CLASSES = 3


def load_data(data_path):
    with open(data_path, 'r') as fp:
        data = json.load(fp)

        X = np.array(data['PIXs'],dtype=object)
        Y = np.array(data['labels'],dtype=object)

        print(f'X shape : {X.shape}')
        print(f'Y shape: {Y.shape}')
        return X, Y


def get_data_splits(data_path, test_size=0.1, val_size=0.1):
    X, Y = load_data(data_path)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_size)

    print(f'X_train shape : {X_train.shape}')
    print(f'X_test shape : {X_test.shape}')
    print(f'X_val shape : {X_val.shape}')

    # X_train = X_train[...,np.newaxis]
    # X_val = X_val[...,np.newaxis]
    # X_test = X_test[...,np.newaxis]

    print(f'new X_train shape : {X_train.shape}')
    print(f'new X_test shape : {X_test.shape}')
    print(f'new X_val shape : {X_val.shape}')

    return X_train, X_test, X_val, Y_train, Y_test, Y_val

    #kernel_regularizer=tf.keras.regularizers.l2(0.001)


def build_model(input_shape, learning_rate, error='sparse_categorical_crossentropy'):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(2, 2))

    model.add(tf.keras.layers.Conv2D(32, (5, 5), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(2, 2))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(2, 2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(NUM_IMG_CLASSES, activation='softmax'))

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=error, metrics=['accuracy'])
    model.summary()
    return model


def main():
    X_train, X_test, X_val, Y_train, Y_test, Y_val = get_data_splits(DATA_PATH)

    input_shape = (X_train.shape[1], X_train.shape[2],X_train.shape[3])
    print(input_shape)
    print(f"X_train_1 : {X_train.shape[1]}")
    print(f"X_train_2 : {X_train.shape[2]}")
    print(f"X_train_3 : {X_train.shape[3]}")

    model = build_model(input_shape, LEARNING_RATE)
    model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, Y_val))

    test_err, test_acc = model.evaluate(X_test, Y_test)
    print(f'Test error : {test_err} , Test Acc : {test_acc}')
    model.save(SAVED_MODEL_PATH)


if __name__ == '__main__':
    main()