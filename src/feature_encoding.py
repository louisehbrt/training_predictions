from src import config
from src import feature_processing
import numpy as np
import pandas as pd
import joblib
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from dirty_cat import SimilarityEncoder, MinHashEncoder, GapEncoder


def chose_feature_encoding(df, feature, name_encoder):
    if name_encoder == 'label':
        encoder = LabelEncoder()
        df[feature] = encoder.fit_transform(df[feature])

    if name_encoder == 'onehot':
        df = pd.get_dummies(df, columns=[feature])

    elif name_encoder == 'basen':
        encoder = ce.BaseNEncoder(cols=[feature], return_df=True, base=5)
        df = encoder.fit_transform(df)

    elif name_encoder == 'similarity':
        encoder = SimilarityEncoder(similarity='ngram')
        df[feature] = encoder.fit_transform(np.array(df[feature]).reshape(-1, 1))

    elif name_encoder == 'minhash':
        encoder = MinHashEncoder(n_components=100)
        df[feature] = encoder.fit_transform(np.array(df[feature]).reshape(-1, 1))

    elif name_encoder == 'gap':
        encoder = GapEncoder(n_components=100)
        df[feature] = encoder.fit_transform(np.array(df[feature]).reshape(-1, 1))

    return df


def chose_encoder(df, name, mode):
    # BaseNEncoder
    if name == 'basen':
        encoder = ce.BaseNEncoder(
            cols=['Course Name', 'Course Skill', 'Specialization', 'Training Provider', 'Main Domain', 'Course Code'],
            return_df=True, base=5)
        df = encoder.fit_transform(df)

    # SimilarityEncoder
    elif name == 'similarity':
        encoders = {}
        for c in ['Course Name', 'Course Skill', 'Specialization', 'Training Provider', 'Main Domain', 'Course Code']:
            encoders[c] = SimilarityEncoder(similarity='ngram')
            if mode == 'train':
                df[c] = encoders[c].fit_transform(np.array(df[c]).reshape(-1, 1))
            else:
                encoders = joblib.load(config.MODELS_PATH + 'feature_encoders.pkl')
                df[c] = encoders[c].transform(df[c])

    # MinHashEncoder
    elif name == 'minhash':
        encoders = {}
        for c in ['Course Name', 'Course Skill', 'Specialization', 'Training Provider', 'Main Domain', 'Course Code']:
            encoders[c] = MinHashEncoder(n_components=100)
            if mode == 'train':
                df[c] = encoders[c].fit_transform(np.array(df[c]).reshape(-1, 1))
            else:
                encoders = joblib.load(config.MODELS_PATH + 'feature_encoders.pkl')
                df[c] = encoders[c].transform(df[c])

    # GapEncoder
    elif name == 'gap':
        encoders = {}
        for c in ['Course Name', 'Course Skill', 'Specialization', 'Training Provider', 'Main Domain', 'Course Code']:
            encoders[c] = GapEncoder(n_components=100)
            if mode == 'train':
                df[c] = encoders[c].fit_transform(np.array(df[c]).reshape(-1, 1))
            else:
                encoders = joblib.load(config.MODELS_PATH + 'feature_encoders.pkl')
                df[c] = encoders[c].transform(df[c])

    return df



