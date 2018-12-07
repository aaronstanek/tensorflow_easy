import tensorflow as tf
from tensorflow.keras.models import model_from_json as mfj

def load(modelfile, weightsfile):
    with open(modelfile, "r") as file:
        jm = file.read()
    model = mfj(jm)
    model.load_weights(weightsfile)
    return model
