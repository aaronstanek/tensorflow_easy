# suppose we want to learn an XOR function

import tensorflow_easy as te
import numpy as np

x = np.array([[0,0],[0,1],[1,0],[1,1]])
# the inputs

y = np.array([0,1,1,0])
# the outputs

model = te.build([2,4,2])
# makes a neural network
# input layer size 2
# one hidden layer with size 4
# output layer with size 2
# listed from input to output

te.train(model,x,y,2000)
# train for 2000 cycles

percent_correct = te.evaluate(model,x,y)
# tests how good the model is

predicted_answers = te.predict(model,x)
# gives an array of predicted answers
# answers are given in the same order as the inputs

single_predicted_answer = te.predict_single(model,x[2])
# predicts the output for [1,0]
# which should be 1

te.save("./myfile",model)
# saves the model
# first argument is the file where to save it
# second argument if the model itself

loaded_model = te.load("./myfile",[2,4,2])
# loads a model from a file
# first argument is the file to load
# second argument is the shape and size of the neural network
