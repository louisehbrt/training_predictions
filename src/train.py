from src import config
from src import feature_encoding
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



def train_with_2021_test(data_train,data_test):

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

    print('With 2021 as test set :\n')
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




def train_split_function(data):

    df = pd.read_csv(data)
    df = df[df['Year'] < 2022]  # on ne prend pas l'année 2022 qui n'est pas complète
    X = df.drop(columns=['Participants'])
    y = df["Participants"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)


    MODELS_REGRESSION = {
        'linearregration': LinearRegression(),
        'ridgeregression': Ridge(10),
        'lassoregression': Lasso(5.),
        'randomforestregressor': RandomForestRegressor(n_estimators=100),
        'svr': SVR(kernel='rbf')
    }
    print('With the split function :\n')
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
        'bagging': BaggingClassifier(base_estimator=RandomForestClassifier(), random_state=7),
    }

    print('\n CLASSIFICATION MODELS :')
    for name, function in MODELS_CLASSIFICATION.items():
        model = function
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        print(name, ' precision : ', model.score(X_test, y_test)*100, 'and f1-score : ', f1_score(y_test, y_predict, average='weighted'))


def keep_best(method, eval, X_train, X_test, y_train, y_test):

    if method == 'regression':

        if eval == 'r2':

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            r2_best = r2_score(y_test, y_predict)
            name_best = 'linearregression'

            MODELS_REGRESSION = {
                'ridgeregression': Ridge(10),
                'lassoregression': Lasso(5.),
                'randomforestregressor': RandomForestRegressor(n_estimators=100),
                'svr': SVR(kernel='rbf')
            }

            for name, function in MODELS_REGRESSION.items():
                model = function
                model.fit(X_train, y_train)
                y_predict = model.predict(X_test)
                r2_temp = r2_score(y_test, y_predict)

                if r2_best < r2_temp :
                    r2_best = r2_temp
                    name_best = name

            return name_best, r2_best

        elif eval == 'mse':

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            mse_best = mean_squared_error(y_test, y_predict)
            name_best = 'linearregression'

        MODELS_REGRESSION = {
            'ridgeregression': Ridge(10),
            'lassoregression': Lasso(5.),
            'randomforestregressor': RandomForestRegressor(n_estimators=100),
            'svr': SVR(kernel='rbf')
        }

        for name, function in MODELS_REGRESSION.items():
            model = function
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            mse_temp = mean_squared_error(y_test, y_predict)

            if mse_best < mse_temp:
                mse_best = mse_temp
                name_best = name

            return (name_best, mse_best)

    if method == 'classification':

        if eval == 'precision':

            model = DecisionTreeClassifier(random_state=7)
            model.fit(X_train, y_train)
            precision_best = model.score(X_test, y_test) * 100
            name_best = 'decisiontree'

            MODELS_CLASSIFICATION = {
                # 'logisticregration': LogisticRegression(random_state=7, max_iter=10000),
                'randomforestclassifier': RandomForestClassifier(random_state=7),
                'svc': SVC(random_state=7),
                'naivebaysianclass': GaussianNB(),
                'adaboost': AdaBoostClassifier(base_estimator=RandomForestClassifier(), random_state=7),
                'gradientboost': GradientBoostingClassifier(random_state=7),
                'bagging': BaggingClassifier(base_estimator=RandomForestClassifier(), random_state=7),
            }

            for name, function in MODELS_CLASSIFICATION.items():
                model = function
                model.fit(X_train, y_train)
                precision_temp = model.score(X_test, y_test) * 100

                if precision_best < precision_temp:
                    precision_best = precision_temp
                    name_best = name

            return name_best, precision_best

        if eval == 'f1':

            model = DecisionTreeClassifier(random_state=7)
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            f1_best = f1_score(y_test, y_predict, average='weighted')
            name_best = 'decisiontree'

            MODELS_CLASSIFICATION = {
                # 'logisticregration': LogisticRegression(random_state=7, max_iter=10000),
                'randomforestclassifier': RandomForestClassifier(random_state=7),
                'svc': SVC(random_state=7),
                'naivebaysianclass': GaussianNB(),
                'adaboost': AdaBoostClassifier(base_estimator=RandomForestClassifier(), random_state=7),
                'gradientboost': GradientBoostingClassifier(random_state=7),
                'bagging': BaggingClassifier(base_estimator=RandomForestClassifier(), random_state=7),
            }

            for name, function in MODELS_CLASSIFICATION.items():
                model = function
                model.fit(X_train, y_train)
                y_predict = model.predict(X_test)
                f1_temp = f1_score(y_test, y_predict, average='weighted')

                if f1_best < f1_temp:
                    f1_best = f1_temp
                    name_best = name

            return name_best, f1_best




def compare_encoding(data):
    df = pd.read_csv(data)

    encoders = ['label', 'onehot', 'basen', 'similarity', 'minhash', 'gap']

    # On initialise tous les encodages avec label encoder
    columns = ['Course Code', 'Course Name', 'Course Type', 'Course Status', 'Country/Territory', 'Course Skill',
               'Main Domain', 'Priority', 'Training Provider', 'Managed Type', 'Delivery Tool Platform',
               'Display Course Type', 'Specialization']



    for index_enc, enc in enumerate(encoders):
        df_enc = df.copy()
        for index_col, col in enumerate(columns):
            df_enc = feature_encoding.chose_feature_encoding(df_enc, col, enc)

        test = df_enc[df_enc['Year'] == 2021]
        train = df_enc[df_enc['Year'] < 2021]

        test.to_csv('/Users/louise.hubert/PycharmProjects/training_predictions/data/test_data.csv', index=False)
        train.to_csv('/Users/louise.hubert/PycharmProjects/training_predictions/data/train_data.csv', index=False)

        X_train = train.drop(columns=['Participants'])
        y_train = train["Participants"]

        X_test = test.drop(columns=['Participants'])
        y_test = test["Participants"]

        name, score = keep_best('classification', 'f1', X_train, X_test, y_train, y_test)

        print(enc, name, score)


    return (name, score)

NAME, SCORE = compare_encoding('/Users/louise.hubert/PycharmProjects/training_predictions/data/processed_data.csv')


#train_split_function(DF)
#train_with_2021_test(TRAIN,TEST)
#train_with_2021_test('/Users/louise.hubert/PycharmProjects/training_predictions/data/train_data.csv','/Users/louise.hubert/PycharmProjects/training_predictions/data/test_data.csv')
#train_split_function('/Users/louise.hubert/PycharmProjects/training_predictions/data/encoded_data.csv')