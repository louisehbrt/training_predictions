import pandas as pd
import plotly.express as px


class_f1 = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_f1.csv')
class_prec = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_precision.csv')
regr_mse = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/regression_mse.csv')
regr_r2 = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/regression_r2.csv')
regr_r2_mod = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/regression_r2_mod.csv')


list_of_df = [class_f1,class_prec,regr_mse,regr_r2,regr_r2_mod]

for i in range (len(list_of_df)):
    fig = px.bar(list_of_df[i],
             x='Name_Model',
             y='Score',
             color="Name_Model"
              )
    fig.show()

Pair_Encoders = []
for i in range (len(regr_r2_mod)):
    Pair_Encoders.append((regr_r2_mod['High_Encoder'][i],regr_r2_mod['Low_Encoder'][i]))

regr_r2_mod = regr_r2_mod.join(pd.DataFrame(Pair_Encoders))
regr_r2_mod['Pair_Encoders'] = regr_r2_mod[0]+' + '+regr_r2_mod[1]

regr_r2_mod=regr_r2_mod.drop(columns=[0,1])

fig = px.bar(regr_r2_mod,
             x='Pair_Encoders',
             y='Score',
             color="Name_Model"
              )
fig.show()
