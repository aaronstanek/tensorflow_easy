import tensorflow as tf
from tensorflow.keras.models import model_from_json as mfj
from .build import build

def load(filename, arr):
    model = build(arr)
    model.load_weights(filename)
    return model
