import pandas as pd
import plotly.express as px


class_f1 = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_f1.csv')
class_prec = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_precision.csv')
regr_mse = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/regression_mse.csv')
regr_r2 = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/regression_r2.csv')

list_of_df = [class_f1,class_prec,regr_mse,regr_r2]

for i in range (len(list_of_df)):
    fig = px.bar(list_of_df[i],
             x='Encoder',
             y='Score',
             color="Name_Model"
              )
    fig.show()