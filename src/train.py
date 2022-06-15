from src import config
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
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
    X_train = df_train.drop(columns=['Participants'])
    y_train = df_train["Participants"]

    df_test = pd.read_csv(data_test)
    X_test = df_test.drop(columns=['Participants'])
    y_test = df_test["Participants"]

    MODELS_REGRESSION = {
        'linearregration': LinearRegression(),
        'ridgeregression': Ridge(10),
        'lassoregression': Lasso(5.),
        'randomforestregressor': RandomForestRegressor(n_estimators=100),
        'svr': SVR(kernel='rbf')
    }

    print('REGRESSION MODELS :')
    for name, function in MODELS_REGRESSION.items():
        model = function
        model.fit(X_train,y_train)
        y_predict = model.predict(X_test)
        print(name, ' r2_score : ', r2_score(y_test, y_predict), 'and mse : ', mean_squared_error(y_test,y_predict))

    MODELS_CLASSIFICATION = {
        #'logisticregration': LogisticRegression(random_state=7, max_iter=10000),
        'decisiontree': DecisionTreeClassifier(random_state=7),
        'randomforestclassifier': RandomForestClassifier(random_state=7),
        'svc': SVC(random_state=7),
        'naivebaysianclass': GaussianNB(),
        'adaboost': AdaBoostClassifier(base_estimator=RandomForestClassifier(), random_state=7),
        'gradientboost': GradientBoostingClassifier(random_state=7),
        'baggin' : BaggingClassifier(base_estimator=RandomForestClassifier(), random_state=7),
    }

    print('\n CLASSIFICATION MODELS :')
    for name, function in MODELS_CLASSIFICATION.items():
        model = function
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        print(name, ' precision : ', model.score(X_test, y_test)*100, 'and f1-score : ', f1_score(y_test, y_predict, average='weighted'))


train('/Users/louise.hubert/PycharmProjects/training_predictions/data/train_data.csv','/Users/louise.hubert/PycharmProjects/training_predictions/data/test_data.csv')