def evaluate(model, x, y):
    test_loss, test_acc = model.evaluate(x,y,verbose=0)
    return test_acc
