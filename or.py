
from utils.all_utils import prepare_data, save_plot
import pandas as pd
from utils.model import Perceptron
# Let's for OR gate

def main(data, modelName, plotName, eta, epochs):
    df_OR = pd.DataFrame(data)
    # Let's use the perceptron class

    X, y = prepare_data(df_OR)  # target col = y  default

    model_or = Perceptron(eta = eta, epochs=epochs)
    # now we call the fit method
    model_or.fit(X,y)

    # also we calculate the total loss
    _=model_or.total_loss()

    # to save or model
    model_or.save(filename=modelName, model_dir = "models")

    save_plot(df_OR, model_or, filename = plotName)


if __name__=="__main__":          # it create the entery point it mean code start with here
    OR = {
        "x1" : [0,0,1,1],
        "x2" : [0,1,0,1],
        "y" : [0,1,1,1]
    }

    ETA = 0.3
    EPOCHS = 10

    main(data=OR, modelName = "or.model", plotName = "or.png", eta = ETA, epochs = EPOCHS) # in main function all the rest of the code is written





