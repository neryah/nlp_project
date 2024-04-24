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
    def __init__(self, samples):        
        self.X = np.array([sample[0] for sample in samples])
        self.Y = np.array([sample[1] for sample in samples])


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

        self.trained_model = model

        return train_rmse, test_rmse, train_accuracy, test_accuracy

    # def predict(bot_message, previous_rounds, review_features)
    #     return self.trained_model.predict([bot_message] + previous_rounds + review_features.values())


# dm_strategy = LinearRegression(bot_message, previous_rounds, review_features, hotel_value)

def LinearRegressionTrainer(samples):
    trainer = MLTrainer(samples)
    model = LinearRegression()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model

def LogisticRegressionTrainer(samples):
    trainer = MLTrainer(samples)
    model = LogisticRegression()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model

def RandomForestRegressorTrainer(samples):
    trainer = MLTrainer(samples)
    model = RandomForestRegressor()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model

def RandomForestClassifierTrainer(samples):
    trainer = MLTrainer(samples)
    model = RandomForestClassifier()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model

def SVRTrainer(samples):
    trainer = MLTrainer(samples)
    model = SVR()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model

def SVCTrainer(samples):
    trainer = MLTrainer(samples)
    model = SVC()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model

def MLPRegressorTrainer(samples):
    trainer = MLTrainer(samples)
    model = MLPRegressor()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model

def MLPClassifierTrainer(samples):
    trainer = MLTrainer(samples)
    model = MLPClassifier()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model

def GradientBoostingRegressorTrainer(samples):
    trainer = MLTrainer(samples)
    model = GradientBoostingRegressor()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model

def GradientBoostingClassifierTrainer(samples):
    trainer = MLTrainer(samples)
    model = GradientBoostingClassifier()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model

def AdaBoostRegressorTrainer(samples):
    trainer = MLTrainer(samples)
    model = AdaBoostRegressor()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model

def AdaBoostClassifierTrainer(samples):
    trainer = MLTrainer(samples)
    model = AdaBoostClassifier()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model

def BaggingRegressorTrainer(samples):
    trainer = MLTrainer(samples)
    model = BaggingRegressor()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model


def BaggingClassifierTrainer(samples):
    trainer = MLTrainer(samples)
    model = BaggingClassifier()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model

def StackingRegressorTrainer(samples):
    trainer = MLTrainer(samples)
    model = StackingRegressor()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model


def StackingClassifierTrainer(samples):
    trainer = MLTrainer(samples)
    model = StackingClassifier()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model


def VotingRegressorTrainer(samples):
    trainer = MLTrainer(samples)
    model = VotingRegressor()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model

def VotingClassifierTrainer(samples):
    trainer = MLTrainer(samples)
    model = VotingClassifier()
    train_rmse, test_rmse, train_accuracy, test_accuracy = trainer.train(model)
    # Loss
    print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    # Accuracy
    print(f"Training Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    return model





# Example usage:
# trainer = MLTrainer(bot_message, previous_rounds, review_features, hotel_value)
# model = LinearRegression()
# train_rmse, test_rmse = trainer.train(model)
# print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")






# class LinearRegressionTrainer(MLTrainer):
#     def __init__(self, bot_message, previous_rounds, review_features, hotel_value):
#         super().__init__(bot_message, previous_rounds, review_features, hotel_value)
        

#     def train(self):
#         model = LinearRegression()
#         return super().train(model)

# Example usage:
# trainer = LinearRegressionTrainer(bot_message, previous_rounds, review_features, hotel_value)
# train_rmse, test_rmse = trainer.train()
# print(f"Linear Regression - Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")

