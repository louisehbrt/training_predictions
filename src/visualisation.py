import pandas as pd
import plotly.express as px
from math import *


class_f1 = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_f1.csv')
class_prec = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_precision.csv')
regr_mse = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/regression_mse.csv')
regr_r2 = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/regression_r2.csv')
#regr_r2_mod = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/regression_r2_mod.csv')
regr_r2_mod = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/regression_little_r2_mod.csv')

class_f1_mod_class = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_f1_mod_classes.csv')
class_prec_mod_class = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_precision_mod_classes.csv')

class_f1_mod = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_f1_mod.csv')
class_pre_mod = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_precision_mod.csv')

regr_mse_mod = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/regression_mse_mod.csv')

list_of_df = [class_f1,class_prec,regr_mse,regr_r2,regr_r2_mod]


for i in range (len(list_of_df)):
    fig = px.bar(list_of_df[i],
             x='Name_Model',
             y='Score',
             color="Name_Model"
              )
    fig.show()

def plot_graph(df):
    Pair_Encoders = []
    for i in range (len(df)):
        Pair_Encoders.append((df['High_Encoder'][i],df['Low_Encoder'][i]))

    df = df.join(pd.DataFrame(Pair_Encoders))
    df['Pair_Encoders'] = df[0]+' + '+df[1]

    df=df.drop(columns=[0,1])

    fig = px.bar(df,
                 x='Pair_Encoders',
                y='Score',
                color="Name_Model"
                 )
    fig.show()

plot_graph(class_prec_mod_class)

df = pd.concat([regr_mse_mod,df_new_result],ignore_index=True)

regr_rmse_mod = regr_mse_mod.copy()

regr_rmse_mod['Score'] = regr_rmse_mod['Score'].apply(sqrt)

regr_mse_mod.to_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/regression_mse_mod.csv', index=False)
regr_rmse_mod.to_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/regression_rmse_mod.csv', index=False)