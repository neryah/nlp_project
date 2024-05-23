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
from Simulation.smart_dm import MLTrainer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron, RidgeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



def LinearRegressionTrainer():
    trainer = MLTrainer()
    model = LinearRegression()
    return trainer.train(model)


def LogisticRegressionTrainer():
    trainer = MLTrainer()
    model = LogisticRegression()
    return trainer.cross_val_train(model)


def RandomForestRegressorTrainer():
    trainer = MLTrainer()
    model = RandomForestRegressor()
    return trainer.train(model)


def RandomForestClassifierTrainer():
    trainer = MLTrainer()
    model = RandomForestClassifier()
    return trainer.cross_val_train(model)


def SVRTrainer():
    trainer = MLTrainer()
    model = SVR()
    return trainer.train(model)


def SVCTrainer():
    trainer = MLTrainer()
    model = SVC()
    return trainer.cross_val_train(model)


def MLPRegressorTrainer():
    trainer = MLTrainer()
    model = MLPRegressor()
    return trainer.train(model)


def MLPClassifierTrainer():
    trainer = MLTrainer()
    model = MLPClassifier()
    return trainer.cross_val_train(model)


def GradientBoostingRegressorTrainer():
    trainer = MLTrainer()
    model = GradientBoostingRegressor()
    return trainer.train(model)


def GradientBoostingClassifierTrainer():
    trainer = MLTrainer()
    model = GradientBoostingClassifier()
    return trainer.cross_val_train(model)


def AdaBoostRegressorTrainer():
    trainer = MLTrainer()
    model = AdaBoostRegressor()
    return trainer.train(model)


def AdaBoostClassifierTrainer():
    trainer = MLTrainer()
    model = AdaBoostClassifier()
    return trainer.cross_val_train(model)


def BaggingRegressorTrainer():
    trainer = MLTrainer()
    model = BaggingRegressor()
    return trainer.train(model)


def BaggingClassifierTrainer():
    trainer = MLTrainer()
    model = BaggingClassifier()
    return trainer.cross_val_train(model)

def KNeighborsClassifierTrainer():
    trainer = MLTrainer()
    model = KNeighborsClassifier()
    return trainer.cross_val_train(model)


def DecisionTreeClassifierTrainer():
    trainer = MLTrainer()
    model = DecisionTreeClassifier()
    return trainer.cross_val_train(model)


def ExtraTreeClassifierTrainer():
    trainer = MLTrainer()
    model = ExtraTreeClassifier()
    return trainer.cross_val_train(model)


def GaussianNBTrainer():
    trainer = MLTrainer()
    model = GaussianNB()
    return trainer.cross_val_train(model)


def BernoulliNBTrainer():
    trainer = MLTrainer()
    model = BernoulliNB()
    return trainer.cross_val_train(model)


def MultinomialNBTrainer():
    trainer = MLTrainer()
    model = MultinomialNB()
    return trainer.cross_val_train(model)


def SGDClassifierTrainer():
    trainer = MLTrainer()
    model = SGDClassifier()
    return trainer.cross_val_train(model)


def PassiveAggressiveClassifierTrainer():
    trainer = MLTrainer()
    model = PassiveAggressiveClassifier()
    return trainer.cross_val_train(model)


def PerceptronTrainer():
    trainer = MLTrainer()
    model = Perceptron()
    return trainer.cross_val_train(model)


def RidgeClassifierTrainer():
    trainer = MLTrainer()
    model = RidgeClassifier()
    return trainer.cross_val_train(model)


def QuadraticDiscriminantAnalysisTrainer():
    trainer = MLTrainer()
    model = QuadraticDiscriminantAnalysis()
    return trainer.cross_val_train(model)
