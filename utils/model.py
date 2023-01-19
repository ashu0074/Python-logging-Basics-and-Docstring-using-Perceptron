# model contain all model related code perceptron class present

import numpy as np
import pandas as pd
import os
import joblib




class Perceptron:
    def __init__(self, eta : float = None, epochs: int = None):
        self.weights = np.random.randn(3) * 1e-4  # for small random weight we get after multiply with tiny no. 1e-4
        training = (eta is not None) and (epochs is not None)
        if training:
            print(f"initial weights before training: \n{self.weights}\n")
        self.eta  = eta
        self.epochs = epochs
        
    # (_z_outcome is the internal function and it not utilize from outside / hidden )
    def _z_outcome(self, inputs, weights):
        return np.dot(inputs, weights)
    
    def activation_function(self, z):
        return np.where(z>0, 1, 0)
    
    def fit(self, X , y):  
        self.X = X
        self.y = y
        
        # X with the bias weight x1w1, x2w2...
        X_with_bias = np.c_[self.X, -np.ones((len(self.X),1))]
        print(f"X with bias: \n{X_with_bias}")
        
        # Epoch
        for epoch in range(self.epochs):
            print("--"*10)
            print(f"for epoch >> {epoch}")
            print("--"*10)
            
            # Z dection function 
            z = self._z_outcome(X_with_bias, self.weights)
            y_hat = self.activation_function(z)
            print(f"predicted value after forward pass:\n{y_hat}")
            
            
            # Error:- y-yhat
            self.error = self.y - y_hat
            print(f"Error: \n{self.error}")
            
            
            # update weight:- (wnew = wold + eta.error.x)
            
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error) # X_with_bias.transpose
            print(f"update weights after epoch : {epoch + 1}/{self.epochs}:\n{self.weights}")
            print("##"*10)
            
        
    def predict(self, X):
        X_with_bias = np.c_[X, -np.ones((len(X),1))]
        z = self._z_outcome(X_with_bias,self.weights)  # current x and updated weight
        return self.activation_function(z)

        
    # total_loss is equal to sum of all error
    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"\n total loss: {total_loss}\n")
        return total_loss

    def _create_dir_return_path(self, model_dir, filename):
        os.makedirs(model_dir, exist_ok= True)
        return os.path.join(model_dir, filename)


    def save(self, filename, model_dir = None):
        if model_dir is not None:
            model_file_path = self._create_dir_return_path(model_dir, filename)
            joblib.dump(self,model_file_path)                                    # dump the model into binary path
        else:
             # if we did not given any file name in that case we will create the model dir amd filepath for you 
            model_file_path = self._create_dir_return_path("model",filename)
            joblib.dump(self, model_file_path )


    def load(self, filepath):
        return joblib.load(filepath)    



        