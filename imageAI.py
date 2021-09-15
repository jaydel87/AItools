import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle


# class newModel:
#
#     def __init__(self):
#         self.modelType = "Classifier"
#         self.modelName = "RandomForest"
#         self.n = 1000
#         self.learningRate = 0.1
#         self.randomState = 1
#         self.library = "sklearn"
#         self.earlyStopping = True
#         self.earlyStopRounds = 10
#         self.L1Regularization = 0


def DecisionTree(**kwargs):
    return DecisionTreeClassifier(**kwargs)


def RandomForest(**kwargs):
    print(kwargs)
    return RandomForestClassifier(**kwargs)


def GradientBooster(library, **kwargs):

    if library == "sklearn":
        return GradientBoostingClassifier(**kwargs)
    elif library == "xgboost":
        return XGBClassifier(**kwargs)
    else:
        print("Library not recognised")
        return 0


def train_model(model, training_features, training_class, **kwargs):
    model.fit(training_features, training_class, **kwargs)


def predict(model, prediction_features, **kwargs):
    prediction = model.predict(prediction_features, **kwargs)
    return prediction


def model_probability(model, prediction_features, prediction):
    raw_output = model.predict_proba(prediction_features)
    probability_by_category = np.transpose(raw_output, [1, 0])
    no_categories = probability_by_category.shape[0]
    probability_all = np.zeros(len(prediction))

    for i in range(no_categories):
        idx = np.where(prediction == i+1)
        probability_all[idx] = probability_by_category[i, idx]

    return probability_by_category, probability_all

def save_model(model, name, features, sizes):
    featureDict = {
        "Features": features,
        "Kernel sizes": sizes
    }

    model_ext = ".ai"
    features_ext = ".ft"
    folder = "savedModels/"
    modelFileName = folder + name + model_ext
    featureFileName = folder + name + features_ext
    with open(modelFileName, 'wb') as file:
        pickle.dump(model, file)
    with open(featureFileName, 'wb') as file:
        pickle.dump(featureDict, file)

    return


def load_model(name):
    model_ext = ".ai"
    features_ext = ".ft"
    folder = "savedModels/"
    modelFileName = folder + name + model_ext
    featureFileName = folder + name + features_ext
    with open(modelFileName, 'rb') as file:
        model = pickle.load(file)
    with open(featureFileName, 'rb') as file:
        features = pickle.load(file)

    return model, features


