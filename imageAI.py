from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error


class newModel:

    def __init__(self):
        self.modelType = "Classifier"
        self.modelName = "RandomForest"
        self.n = 1000
        self.learningRate = 0.1
        self.randomState = 0
        self.library = "sklearn"
        self.earlyStopping = True
        self.earlyStopRounds = 5


def init_model(self):

    if self.modelName == "DecisionTree":
        model = DecisionTreeClassifier(random_state=self.randomState)
    elif self.modelName == "RandomForest":
        model = RandomForestClassifier(random_state=self.randomState)
    elif self.modelName == "GradientBooster":
        if self.library == "sklearn":
            model = GradientBoostingClassifier(random_state=self.randomState, n_estimators=self.n,
                                               learning_rate=self.learningRate)
        elif self.library == "xgboost":
            model = XGBClassifier(random_state=self.randomState, n_estimators=self.n,
                                               learning_rate=self.learningRate)

    return model
