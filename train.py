def train(model, x, y, epochs, batch_size = 10, verbose = 0):
    model.fit(x,y,epochs=epochs,batch_size=batch_size,verbose=verbose)
