import tensorflow as tf
from tensorflow import keras

def build(arr):
    # arr is a 1D array of layer sizes
    if len(arr) == 0:
        raise ValueError()
    layers = []
    for i in range(len(arr)-1):
        layers.append( keras.layers.Dense(arr[i], activation=tf.nn.relu) )
    layers.append( keras.layers.Dense(arr[i], activation=tf.nn.softmax) )
    model = keras.Sequential(layers)
    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )
    return model
