from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np

class MLTrainer:
    def __init__(self, samples):        
        self.X = np.array([sample[0] for sample in samples])
        self.Y = np.array([sample[1] for sample in samples])


    def train(self, model):

        # Split data into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)

        # Train the model
        model.fit(X_train, Y_train)

        # Make predictions
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

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

