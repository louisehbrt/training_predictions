import pandas as pd
from src import config


def feature_process(data1, data2, main_domains, mode, encoder_name):
    France = pd.read_excel(data1)
    Ben_Mau = pd.read_excel(data2)

    # Retirer les inutiles/vides
    for i in range(42204, 42213):
        Ben_Mau = Ben_Mau.drop(Ben_Mau.index[42204])

    for i in range(34212, 34221):
        France = France.drop(France.index[34212])

    # Fusionner les datasets
    df = pd.concat([France, Ben_Mau], ignore_index=True)

    # Supprimer les variables inutiles
    df1 = df.copy()
    df1 = df1.drop(columns=['Session Code', 'Session Start Date', 'Session End Date', 'Course Skill Relevancy',
                            'Service/Group Level 1', 'Total Hours'])

    # Retirer les formations 'Online', 'Video' et 'Mobile'
    df1.drop(df1.loc[df1['Display Course Type'] == 'Mobile'].index, inplace=True)
    df1.drop(df1.loc[df1['Display Course Type'] == 'Online'].index, inplace=True)
    df1.drop(df1.loc[df1['Display Course Type'] == 'Video'].index, inplace=True)

    # Créer la variable Year
    df1['Year'] = df1['Completion Date'].dt.year
    df1 = df1.drop(columns=['Completion Date'])  # on retirer la variable Completion Date

    # Regrouper les lignes par formation et par année
    grouped_by1 = pd.DataFrame(df1.groupby(by=['Course Code', 'Year']).sum().groupby(level=[
        0]).cumsum())  # on regroupe le df avec les variables course code et year pour sommer le nombre de participants par formation par an
    participants_by_course = grouped_by1['No. of Completions']  # on conserve le nombre de participants
    participants_by_course = participants_by_course.reset_index()  # on reset l'index pour ensuite merge
    participants_by_course = participants_by_course.rename(
        columns={'No. of Completions': 'Participants'})  # on rename le nom de la colonne
    df2 = df1.copy()
    df2.drop_duplicates(subset=["Course Code", "Year"], keep='first', inplace=True)  # on retire les doublons
    df2 = df2.merge(participants_by_course, on=["Course Code", "Year"],
                    how='left')  # on merge les dataframes, on a la nouvelle colonne des participants
    df2 = df2.drop(columns=['No. of Completions'])  # on peut retirer la colonne de participations aux formations

    # Ajout de la variable Main Domain
    df3 = df2.copy()
    domains = pd.read_excel(main_domains)  # on récupère la table contenant les main domains
    domains = domains.drop(columns=['Course', 'Training Domain'])  # on retire les colonnes inutiles
    df4 = df3.merge(domains, on='Course Code', how='inner')  # on ajoute la nouvelle variable au dataframe

    df4['Main Domain'].isna().sum()  # on vérifie les valeurs manquantes

    # df_2021 = df4[df4['Year'] == 2021]
    # course_code_2021 = df_2021['Course Code']

    df4.to_csv('/Users/louise.hubert/PycharmProjects/training_predictions/data/processed_data.csv', index=False)

    return df4

    # LabelEncoder
    #labelencoders = {}
    #for c in ['Course Type', 'Course Status', 'Country/Territory', 'Priority', 'Managed Type',
    #          'Delivery Tool Platform', 'Display Course Type']:
    #    labelencoders[c] = LabelEncoder()
    #    if mode == 'train' :
    #        df4[c] = labelencoders[c].fit_transform(df4[c])
    #    else :
    #        labelencoders = joblib.load(config.MODELS_PATH+'feature_encoders.pkl')
    #        df4[c] = labelencoders[c].transform(df4[c])



    #df4 = chose_encoder(df4, encoder_name, mode)
    # BaseNEncoder
    #encoder = ce.BaseNEncoder(
        #cols=['Course Name', 'Course Skill', 'Specialization', 'Training Provider', 'Main Domain', 'Course Code'],
        #return_df=True, base=5)
    #df4 = encoder.fit_transform(df4)

    #df4.to_csv('/Users/louise.hubert/PycharmProjects/training_predictions/data/encoded_data.csv', index=False)

    #if mode == 'train':
    #    joblib.dump(labelencoders, '/Users/louise.hubert/PycharmProjects/training_predictions/models' + 'feature_encoders.pkl')



data1 = config.FRANCE_DATA
data2 = config.BEN_MAU_DATA
main_domains = config.MAIN_DOMAINS
feature_process('/Users/louise.hubert/PycharmProjects/training_predictions/data/training_list_France.xlsx', '/Users/louise.hubert/PycharmProjects/training_predictions/data/training_list_Benelux_Maurice.xlsx', '/Users/louise.hubert/PycharmProjects/training_predictions/data/main_domains.xlsx','train','minhash')
