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

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron, RidgeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import warnings


class MLTrainer:
    _samples = None
    _X = None
    _Y = None

    @classmethod
    def initialize_samples(cls, X, Y):
        if cls._X is None:
            cls._X = np.array(X)
            cls._Y = np.array(Y)
        else:
            raise ValueError("Samples have already been initialized.")

    def __init__(self):
        if self._X is None:
            raise ValueError("Samples must be initialized before creating an instance of MLTrainer.")

        self.X = self._X
        self.Y = self._Y

    def train(self, model):


        # Split data into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.2)

        #time - start measuring time:
        start_train = time.time()

        # Train the model
        model.fit(X_train, Y_train)

        start_predict = time.time()

        # Make predictions
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

        end = time.time()
        print(
            f"Training time: {start_predict - start_train}, Prediction time: {end - start_predict}, Total time: {end - start_train}")

        print(f"Y_pred_train length: {len(Y_pred_train)}, Y_pred_test length: {len(Y_pred_test)}")

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

    # def discrete_train(self, model):
    #
    #     # Split data into train and test sets
    #     X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.2)
    #
    #     #time - start measuring time:
    #     start_train = time.time()
    #
    #     # print(f'X_train: {X_train}, Y_train: {Y_train}')
    #     # Train the model
    #     model.fit(X_train, Y_train)
    #
    #     start_predict = time.time()
    #
    #     # Make predictions
    #     Y_pred_train = model.predict(X_train)
    #     Y_pred_test = model.predict(X_test)
    #
    #     end = time.time()
    #     print(f"Training time: {start_predict - start_train}, Prediction time: {end - start_predict}")
    #
    #     # Evaluate the model
    #     train_rmse = 0
    #     test_rmse = 0
    #     # accuracy - prediction is correct if both values are over 8 or both are under 8:
    #     train_accuracy = accuracy_score(Y_train, Y_pred_train)
    #     test_accuracy = accuracy_score(Y_test, Y_pred_test)
    #
    #     # Loss
    #     print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    #     # Accuracy
    #     print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    #
    #     return train_rmse, test_rmse, train_accuracy, test_accuracy

    def discrete_train(self, model):
        # same as discrete_train but with cross validation
        # implement cross validation
        scores = np.mean(cross_val_score(model, self.X, self.Y, cv=10, scoring='accuracy', n_jobs=-1))
        print(f"Cross validation scores: {scores}")
        return scores


def LinearRegressionTrainer():
    trainer = MLTrainer()
    model = LinearRegression()
    trainer.train(model)
    return model


def LogisticRegressionTrainer():
    trainer = MLTrainer()
    model = LogisticRegression()
    model.fit(trainer.X, trainer.Y)
    return model


def RandomForestRegressorTrainer():
    trainer = MLTrainer()
    model = RandomForestRegressor()
    trainer.train(model)
    return model


def RandomForestClassifierTrainer():
    trainer = MLTrainer()
    model = RandomForestClassifier()
    model.fit(trainer.X, trainer.Y)
    return model


def SVRTrainer():
    trainer = MLTrainer()
    model = SVR()
    trainer.train(model)
    return model


def SVCTrainer():
    trainer = MLTrainer()
    model = SVC()
    model.fit(trainer.X, trainer.Y)
    return model


def MLPRegressorTrainer():
    trainer = MLTrainer()
    model = MLPRegressor()
    model.fit(trainer.X, trainer.Y)
    return model


def MLPClassifierTrainer():
    trainer = MLTrainer()
    model = MLPClassifier()
    model.fit(trainer.X, trainer.Y)
    return model


def GradientBoostingRegressorTrainer():
    trainer = MLTrainer()
    model = GradientBoostingRegressor()
    trainer.train(model)
    return model


def GradientBoostingClassifierTrainer():
    trainer = MLTrainer()
    model = GradientBoostingClassifier()
    model.fit(trainer.X, trainer.Y)
    return model


def AdaBoostRegressorTrainer():
    trainer = MLTrainer()
    model = AdaBoostRegressor()
    trainer.train(model)
    return model


def AdaBoostClassifierTrainer():
    trainer = MLTrainer()
    model = AdaBoostClassifier()
    model.fit(trainer.X, trainer.Y)
    return model


def BaggingRegressorTrainer():
    trainer = MLTrainer()
    model = BaggingRegressor()
    trainer.train(model)
    return model


def BaggingClassifierTrainer():
    trainer = MLTrainer()
    model = BaggingClassifier()
    model.fit(trainer.X, trainer.Y)
    return model


def KNeighborsClassifierTrainer():
    trainer = MLTrainer()
    model = KNeighborsClassifier()
    model.fit(trainer.X, trainer.Y)
    return model


def DecisionTreeClassifierTrainer():
    trainer = MLTrainer()
    model = DecisionTreeClassifier()
    model.fit(trainer.X, trainer.Y)
    return model


def ExtraTreeClassifierTrainer():
    trainer = MLTrainer()
    model = ExtraTreeClassifier()
    model.fit(trainer.X, trainer.Y)
    return model


def GaussianNBTrainer():
    trainer = MLTrainer()
    model = GaussianNB()
    model.fit(trainer.X, trainer.Y)
    return model


def BernoulliNBTrainer():
    trainer = MLTrainer()
    model = BernoulliNB()
    model.fit(trainer.X, trainer.Y)
    return model


def MultinomialNBTrainer():
    trainer = MLTrainer()
    model = MultinomialNB()
    model.fit(trainer.X, trainer.Y)
    return model


def SGDClassifierTrainer():
    trainer = MLTrainer()
    model = SGDClassifier()
    model.fit(trainer.X, trainer.Y)
    return model


def PassiveAggressiveClassifierTrainer():
    trainer = MLTrainer()
    model = PassiveAggressiveClassifier()
    model.fit(trainer.X, trainer.Y)
    return model


def PerceptronTrainer():
    trainer = MLTrainer()
    model = Perceptron()
    model.fit(trainer.X, trainer.Y)
    return model


def RidgeClassifierTrainer():
    trainer = MLTrainer()
    model = RidgeClassifier()
    model.fit(trainer.X, trainer.Y)
    return model


def QuadraticDiscriminantAnalysisTrainer():
    trainer = MLTrainer()
    model = QuadraticDiscriminantAnalysis()
    model.fit(trainer.X, trainer.Y)
    return model
