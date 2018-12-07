import numpy as np

def predict(model, x):
    return model.predict(x)

def predict_single(model, item):
    x = np.array([item])
    p = predict(model,x)
    p = np.argmax(p[0])
    return p
