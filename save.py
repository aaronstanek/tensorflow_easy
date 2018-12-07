def save(modelfile, weightsfile, model):
    jm = model.to_json()
    with open(modelfile, "w") as file:
        file.truncate(0)
        file.write(jm)
    model.save_weights(weightsfile)
