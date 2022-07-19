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
from plotnine import *





def train_with_2021_test(data):
    df = pd.read_csv(data)

    df_test = df[df['Year'] == 2021]
    df_train = df[df['Year'] < 2021]

    X_train = df_train.drop(columns=['Participants'])
    y_train = df_train["Participants"]

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


def keep_best(method, eval, df):
    test = df[df['Year'] == 2021]
    train = df[df['Year'] < 2021]

    X_train = train.drop(columns=['Participants'])
    y_train = train["Participants"]

    X_test = test.drop(columns=['Participants'])
    y_test = test["Participants"]

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

                if r2_best < r2_temp:
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



def apply_random_forest(df,eval):
    test = df[df['Year'] == 2021]
    train = df[df['Year'] < 2021]

    X_train = train.drop(columns=['Participants'])
    y_train = train["Participants"]

    X_test = test.drop(columns=['Participants'])
    y_test = test["Participants"]

    if eval == 'precision':
        model = RandomForestClassifier(random_state=7)
        model.fit(X_train, y_train)
        precision_best = model.score(X_test, y_test) * 100
        name_best = 'randomforestclassifier'

        return name_best, precision_best

    if eval == 'f1':
        model = RandomForestClassifier(random_state=7)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        f1_best = f1_score(y_test, y_predict, average='weighted')
        name_best = 'randomforestclassifier'

        return name_best, f1_best

def apply_decision_tree(df,eval):
    test = df[df['Year'] == 2021]
    train = df[df['Year'] < 2021]

    X_train = train.drop(columns=['Participants'])
    y_train = train["Participants"]

    X_test = test.drop(columns=['Participants'])
    y_test = test["Participants"]

    if eval == 'precision':
        model = DecisionTreeClassifier(random_state=7)
        model.fit(X_train, y_train)
        precision_best = model.score(X_test, y_test) * 100
        name_best = 'decisiontree'

        return name_best, precision_best

    if eval == 'f1':
        model = DecisionTreeClassifier(random_state=7)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        f1_best = f1_score(y_test, y_predict, average='weighted')
        name_best = 'decisiontree'

        return name_best, f1_best

def apply_svc(df,eval):
    test = df[df['Year'] == 2021]
    train = df[df['Year'] < 2021]

    X_train = train.drop(columns=['Participants'])
    y_train = train["Participants"]

    X_test = test.drop(columns=['Participants'])
    y_test = test["Participants"]

    if eval == 'precision':
        model = SVC(random_state=7)
        model.fit(X_train, y_train)
        precision_best = model.score(X_test, y_test) * 100
        name_best = 'svc'

        return name_best, precision_best

    if eval == 'f1':
        model = SVC(random_state=7)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        f1_best = f1_score(y_test, y_predict, average='weighted')
        name_best = 'svc'

        return name_best, f1_best

def apply_adaboost(df,eval):
    test = df[df['Year'] == 2021]
    train = df[df['Year'] < 2021]

    X_train = train.drop(columns=['Participants'])
    y_train = train["Participants"]

    X_test = test.drop(columns=['Participants'])
    y_test = test["Participants"]

    if eval == 'precision':
        model = AdaBoostClassifier(base_estimator=RandomForestClassifier(), random_state=7)
        model.fit(X_train, y_train)
        precision_best = model.score(X_test, y_test) * 100
        name_best = 'adaboost'

        return name_best, precision_best

    if eval == 'f1':
        model = AdaBoostClassifier(base_estimator=RandomForestClassifier(), random_state=7)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        f1_best = f1_score(y_test, y_predict, average='weighted')
        name_best = 'adaboost'

        return name_best, f1_best

def apply_gradientboost(df,eval):
    test = df[df['Year'] == 2021]
    train = df[df['Year'] < 2021]

    X_train = train.drop(columns=['Participants'])
    y_train = train["Participants"]

    X_test = test.drop(columns=['Participants'])
    y_test = test["Participants"]

    if eval == 'precision':
        model = GradientBoostingClassifier(random_state=7),
        model.fit(X_train, y_train)
        precision_best = model.score(X_test, y_test) * 100
        name_best = 'gradientboost'

        return name_best, precision_best

    if eval == 'f1':
        model = GradientBoostingClassifier(random_state=7)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        f1_best = f1_score(y_test, y_predict, average='weighted')
        name_best = 'gradientboost'

        return name_best, f1_best




def encode_high_mod_features(df):
    encoders = ['basen', 'label', 'minhash'] #'onehot','gap', 'similarity'
    high_mod_col = ['Training Provider', 'Specialization', 'Course Skill', 'Course Name', 'Course Code']
    #high_mod_col = ['Training Provider', 'Course Name']

    encoded_high_dfs = []

    for index_enc_high, enc_high in enumerate(encoders):
        df_enc_high = df.copy()
        print('HIGH ENCODER', enc_high)
        for index_col_high, col_high in enumerate(high_mod_col):
            df_enc_high = feature_encoding.chose_feature_encoding(df_enc_high, col_high, enc_high)
        encoded_high_dfs.append(df_enc_high)

    return encoded_high_dfs


def encode_low_mod_features(dfs):
    encoders = ['basen', 'label', 'onehot', 'minhash']#'similarity','gap'
    low_mod_col = ['Course Type', 'Priority', 'Managed Type', 'Display Course Type', 'Course Status',
                   'Country/Territory', 'Delivery Tool Platform', 'Main Domain']

    models = []
    scores = []
    low_encoders = []

    for index_dfs, high_df in enumerate(dfs):
        df_enc = high_df.copy()
        #print('FIRST', df_enc)
        for index_enc_low, enc_low in enumerate(encoders):
            df_enc_low = df_enc.copy()
            print('LOW ENCODER',enc_low)

            for index_col_low, col_low in enumerate(low_mod_col):
                #print(col_low)
                df_enc_low = feature_encoding.chose_feature_encoding(df_enc_low, col_low, enc_low)
                #print(df_enc_low)
            #print('SECOND', df_enc_low)
            #print(df_enc_low)
            #name, score = keep_best('regression', 'r2', df_enc_low)
            name, score = apply_random_forest(df_enc_low,'precision')
            models.append(name)
            scores.append(score)
            low_encoders.append(enc_low)
    return models, scores, low_encoders



def compare_encoding(data):

    df = pd.read_csv(data)

    # Les encodeurs utilisés
    high_encoders = ['basen', 'label', 'minhash']#'onehot','gap', ,'similarity',

    DFS = encode_high_mod_features(df)
    MODELS, SCORES, LOW_ENC = encode_low_mod_features(DFS)
    HIGH_ENC = high_encoders * 4
    print(MODELS)
    print(SCORES)
    print(LOW_ENC)
    print(HIGH_ENC)

    df_low_enc = pd.DataFrame(LOW_ENC, columns=['Low_Encoder'])
    df_high_enc = pd.DataFrame(HIGH_ENC, columns=['High_Encoder'])
    df_models = pd.DataFrame(MODELS, columns=['Name_Model'])
    df_scores = pd.DataFrame(SCORES, columns=['Score'])

    df_result = ((df_models.join(df_scores)).join(df_high_enc)).join(df_low_enc)

    print(df_result)
    #df_result.to_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_random_forest_precision_mod.csv', index=False)

    return df_result



    """"# On initialise une liste vide pour récupérer le nom du modèle et une autre pour le score
    models = []
    scores = []

    #for index_enc_low, enc_low in enumerate(encoders):
        #df_enc_low = df.copy()

        #for index_col_low, col_low in enumerate(low_mod_col):
            #df_enc_low = feature_encoding.chose_feature_encoding(df_enc_low, col_low, enc_low)
            #print(index_col_low,col_low)
            #print(df_enc_low.columns)

            #for index_enc_high, enc_high in enumerate(encoders):
                #df_enc_high = df_enc_low.copy()

                #for index_col_high, col_high in enumerate(high_mod_col):

                    #df_enc_high = feature_encoding.chose_feature_encoding(df_enc_high, col_high, enc_high)
                    #print(col_high, enc_high)
        #print(df_enc_high)



        #name, score = keep_best('regression', 'r2',df_enc_high)
            # models.append(name)
            # scores.append(score)
            #print(df_enc_low)
        #print(enc_low, enc_high, name, score)

    #df_encoders = pd.DataFrame(encoders, columns=['Encoder'])
    #df_models = pd.DataFrame(models, columns=['Name_Model'])
    #df_scores = pd.DataFrame(scores, columns=['Score'])
    #df_result = (df_encoders.join(df_models)).join(df_scores)


    #print(df_result)
    #df_result.to_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_f1.csv', index=False)
    #p = ggplot(df_result, aes('Encoder', 'Score')) + geom_histogram()
    #print(p)
    #return (name, score)"""

#DF_RESULT = compare_encoding('/Users/louise.hubert/PycharmProjects/training_predictions/data/processed_data.csv')
#train_with_2021_test('/Users/louise.hubert/PycharmProjects/training_predictions/data/encoded_data.csv')
#train_split_function('/Users/louise.hubert/PycharmProjects/training_predictions/data/encoded_data.csv')
compare_encoding('/Users/louise.hubert/PycharmProjects/training_predictions/data/processed_data_classes.csv')

"""data = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/data/little_processed_data.csv')
data_low = data.drop(columns = ['Training Provider', 'Course Name'])
data_high = data.drop(columns =['Course Type', 'Priority', 'Managed Type', 'Display Course Type', 'Course Status',
                   'Country/Territory', 'Delivery Tool Platform', 'Main Domain', 'Year', 'Participants', 'Course Hours'])

data_high_enc = pd.get_dummies(data_high)

data_high_enc = data_high_enc.join(data_low)

encoders = ['basen', 'onehot', 'label', 'similarity', 'minhash', 'gap']
low_mod_col = ['Course Type', 'Priority', 'Managed Type', 'Display Course Type', 'Course Status',
                   'Country/Territory', 'Delivery Tool Platform', 'Main Domain']

models = []
scores = []
low_encoders = []


for index_enc_low, enc_low in enumerate(encoders):
    df_enc_low = data_high_enc.copy()
    print(enc_low)

    for index_col_low, col_low in enumerate(low_mod_col):
        #print(col_low)
        df_enc_low = feature_encoding.chose_feature_encoding(df_enc_low, col_low, enc_low)
        #print(df_enc_low)
    print('SECOND', df_enc_low)
    name, score = keep_best('regression', 'r2', df_enc_low)
    models.append(name)
    scores.append(score)
    low_encoders.append(enc_low)




HIGH_ENC = ['onehot']*6


df_low_enc = pd.DataFrame(low_encoders, columns=['Low_Encoder'])
df_high_enc = pd.DataFrame(HIGH_ENC, columns=['High_Encoder'])
df_models = pd.DataFrame(models, columns=['Name_Model'])
df_scores = pd.DataFrame(scores, columns=['Score'])

df_new_result = ((df_models.join(df_scores)).join(df_high_enc)).join(df_low_enc)

data_all = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/regression_little_r2_mod.csv')

df_new_result = df_new_result.append(data_all)

print(df_new_result)
df_new_result.to_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/regression_little_r2_mod.csv', index=False)"""
