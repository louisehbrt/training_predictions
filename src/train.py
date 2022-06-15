from src import config
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pandas as pd

def train(data_train,data_test):
    df_train = pd.read_csv(data_train)

    MODELS = {
        'linearregration': LinearRegression(),
        'randomforestregressor': RandomForestRegressor(n_estimators=100),
        'svr': SVR(kernel='rbf')

    }

    X_train = df_train.drop(columns=['Participants'])
    y_train = df_train["Participants"]


    df_test = pd.read_csv(data_test)
    X_test = df_test.drop(columns=['Participants'])
    y_test = df_test["Participants"]

    model = MODELS['svr']
    model.fit(X_train,y_train)

    y_predict = model.predict(X_test)
    #train other models
    print(r2_score(y_test, y_predict))

train('/Users/louise.hubert/PycharmProjects/training_predictions/data/train_data.csv','/Users/louise.hubert/PycharmProjects/training_predictions/data/test_data.csv')