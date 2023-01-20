import os
from utils.all_utils import prepare_data, save_plot
import pandas as pd
from utils.model import Perceptron
# import the logging module
import logging

gate = "OR gate"
log_dir = "logs"
os.makedirs(log_dir, exist_ok= True)
# logging for the basic config
logging.basicConfig(
    filename=os.path.join(log_dir,"running_logs.log"),  # filename - where to store folder name with file name
    level= logging.INFO,
    format='[%(asctime)s: %(levelname)s: %(module)s]:%(message)s',
    filemode='a'                                                    # append mode

)
def main(data, modelName, plotName, eta, epochs):
    df = pd.DataFrame(data)
    logging.info(f"This is raw DataSet: \n{df}")
    # Let's use the perceptron class

    X, y = prepare_data(df)  # target col = y  default

    model = Perceptron(eta = eta, epochs=epochs)
    # now we call the fit method
    model.fit(X,y)

    # also we calculate the total loss
    _= model.total_loss()

    # to save or model
    model.save(filename=modelName, model_dir = "models")

    save_plot(df,  model, filename = plotName)


if __name__=="__main__":          # it create the entery point it mean code start with here
    OR = {
        "x1" : [0,0,1,1],
        "x2" : [0,1,0,1],
        "y" : [0,1,1,1]
    }

    ETA = 0.3
    EPOCHS = 10

    # there is high change of error occuring  so we use try and error block

    try: 
        logging.info(f">>>>>>>>>>> Starting the Training for {gate} <<<<<<<<<<<<<<")
        main(data=OR, modelName = "or.model", plotName = "or.png", eta = ETA, epochs = EPOCHS) # in main function all the rest of the code is written
        logging.info(f">>>>>>>>>>>>Ending of Training for {gate}<<<<<<<<<<<<<<<<<<\n\n\n")
    except Exception as e:
        logging.exception(e)
        raise e






