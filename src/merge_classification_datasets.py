import pandas as pd


# F1-SCORE
ada_f1 = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_adaboost_f1_mod_classes.csv')
dec_f1 = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_decision_tree_f1_mod_classes.csv')
rand_f1 = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_random_forest_f1_mod_classes.csv')
svc_f1 = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_svc_f1_mod_classes.csv')
#grad_f1 = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_gradientboost_f1_mod_classes.csv')


df = ada_f1

best_score = df['Score'][0]


for i in range (len(df)):
    best_score = df['Score'][i]

    if dec_f1['Score'][i]>best_score:
        best_score = dec_f1['Score'][i]
        df['Score'][i] = best_score
        df['Name_Model'][i] = dec_f1['Name_Model'][i]

    if rand_f1['Score'][i]>best_score:
        best_score = rand_f1['Score'][i]
        df['Score'][i] = best_score
        df['Name_Model'][i] = rand_f1['Name_Model'][i]

    if svc_f1['Score'][i] > best_score:
        best_score = svc_f1['Score'][i]
        df['Score'][i] = best_score
        df['Name_Model'][i] = svc_f1['Name_Model'][i]

print(df)

df.to_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_f1_mod_classes.csv')



# PRECISION
ada_prec = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_adaboost_precision_mod_classes.csv')
dec_prec = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_decision_tree_precision_mod_classes.csv')
rand_prec = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_random_forest_precision_mod_classes.csv')
svc_prec = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_svc_precision_mod_classes.csv')
#grad_f1 = pd.read_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_gradientboost_f1_mod_classes.csv')


df_prec = ada_prec
print(df_prec)
best_score = df_prec['Score'][0]


for i in range (len(df)):
    best_score = df_prec['Score'][i]

    if dec_prec['Score'][i]>best_score:
        best_score = dec_prec['Score'][i]
        df_prec['Score'][i] = best_score
        df_prec['Name_Model'][i] = dec_prec['Name_Model'][i]

    if rand_prec['Score'][i]>best_score:
        best_score = rand_prec['Score'][i]
        df_prec['Score'][i] = best_score
        df_prec['Name_Model'][i] = rand_prec['Name_Model'][i]

    if svc_prec['Score'][i] > best_score:
        best_score = svc_prec['Score'][i]
        df_prec['Score'][i] = best_score
        df_prec['Name_Model'][i] = svc_prec['Name_Model'][i]

print(df_prec)

df_prec.to_csv('/Users/louise.hubert/PycharmProjects/training_predictions/models/classification_precision_mod_classes.csv')