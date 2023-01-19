
from utils.all_utils import prepare_data, save_plot
import pandas as pd
from utils.model import Perceptron
# Let's for OR gate

def main(data, modelName, plotName, eta, epochs):
    df_AND = pd.DataFrame(data)
    # Let's use the perceptron class

    X, y = prepare_data(df_AND)  # target col = y  default

    model = Perceptron(eta = eta, epochs=epochs)
    # now we call the fit method
    model.fit(X,y)

    # also we calculate the total loss
    _=model.total_loss()

    # to save or model
    model.save(filename=modelName, model_dir = "models")

    save_plot(df_AND,  model, filename = plotName)


if __name__=="__main__":          # it create the entery point it mean code start with here
    AND = {
        "x1" : [0,0,1,1],
        "x2" : [0,1,0,1],
        "y" : [0,0,0,1]
    }

    ETA = 0.3
    EPOCHS = 10

    main(data=AND, modelName = "and.model", plotName = "and.png", eta = ETA, epochs = EPOCHS) # in main function all the rest of the code is written





