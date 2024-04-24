import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np

# weak models:
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

#boosting models:
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
#bagging models:
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier
#stacking models:
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import StackingClassifier
#voting models:
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import VotingClassifier


class MLTrainer:
    _samples = None
    _X = None
    _Y = None
    
    @classmethod
    def initialize_samples(cls, samples):
        if cls._samples is None:
            cls._samples = samples
            cls._X = np.array([sample[0] for sample in samples])
            cls._Y = np.array([sample[1] for sample in samples])

    def __init__(self):        
        if self._samples is None:
            raise ValueError("Samples must be initialized before creating an instance of MLTrainer.")
        
        self.X = self._X
        self.Y = self._Y

    def train(self, model):

        # Split data into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        
        #time - start measuring time:
        start_train = time.time()

        # Train the model
        model.fit(X_train, Y_train)
        
        start_predict = time.time()

        # Make predictions
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)
        
        end = time.time()
        print(f"Training time: {start_predict - start_train}, Prediction time: {end - start_predict}")

        # Evaluate the model
        train_rmse = mean_squared_error(Y_train, Y_pred_train, squared=False)
        test_rmse = mean_squared_error(Y_test, Y_pred_test, squared=False)
        # accuracy - prediction is correct if both values are over 8 or both are under 8:
        train_accuracy = accuracy_score(Y_train >= 8, Y_pred_train >= 8)
        test_accuracy = accuracy_score(Y_test >= 8, Y_pred_test >= 8)

        # Loss
        print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
        # Accuracy
        print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        return train_rmse, test_rmse, train_accuracy, test_accuracy

def LinearRegressionTrainer():
    trainer = MLTrainer()
    model = LinearRegression()
    trainer.train(model)
    return model

def LogisticRegressionTrainer():
    trainer = MLTrainer()
    model = LogisticRegression()
    trainer.train(model)
    return model

def RandomForestRegressorTrainer():
    trainer = MLTrainer()
    model = RandomForestRegressor()
    trainer.train(model)
    return model

def RandomForestClassifierTrainer():
    trainer = MLTrainer()
    model = RandomForestClassifier()
    trainer.train(model)
    return model

def SVRTrainer():
    trainer = MLTrainer()
    model = SVR()
    trainer.train(model)
    return model

def SVCTrainer():
    trainer = MLTrainer()
    model = SVC()
    trainer.train(model)
    return model

def MLPRegressorTrainer():
    trainer = MLTrainer()
    model = MLPRegressor()
    trainer.train(model)
    return model

def MLPClassifierTrainer():
    trainer = MLTrainer()
    model = MLPClassifier()
    trainer.train(model)
    return model

def GradientBoostingRegressorTrainer():
    trainer = MLTrainer()
    model = GradientBoostingRegressor()
    trainer.train(model)
    return model

def GradientBoostingClassifierTrainer():
    trainer = MLTrainer()
    model = GradientBoostingClassifier()
    trainer.train(model)
    return model

def AdaBoostRegressorTrainer():
    trainer = MLTrainer()
    model = AdaBoostRegressor()
    trainer.train(model)
    return model

def AdaBoostClassifierTrainer():
    trainer = MLTrainer()
    model = AdaBoostClassifier()
    trainer.train(model)
    return model

def BaggingRegressorTrainer():
    trainer = MLTrainer()
    model = BaggingRegressor()
    trainer.train(model)
    return model

def BaggingClassifierTrainer():
    trainer = MLTrainer()
    model = BaggingClassifier()
    trainer.train(model)
    return model

def StackingRegressorTrainer():
    trainer = MLTrainer()
    model = StackingRegressor()
    trainer.train(model)
    return model

def StackingClassifierTrainer():
    trainer = MLTrainer()
    model = StackingClassifier()
    trainer.train(model)
    return model

def VotingRegressorTrainer():
    trainer = MLTrainer()
    model = VotingRegressor()
    trainer.train(model)
    return model

def VotingClassifierTrainer():
    trainer = MLTrainer()
    model = VotingClassifier()
    trainer.train(model)
    return model

