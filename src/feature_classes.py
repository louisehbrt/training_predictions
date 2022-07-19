import pandas as pd
from src import config
from plotnine import *

def define_classes(x):
    if x < 5 :
        return "Class 0"
    elif x < 10 :
        return "Class 1"
    elif x < 20 :
        return "Class 2"
    elif x < 30 :
        return "Class 3"
    elif x < 40 :
        return "Class 4"
    elif x < 50 :
        return "Class 5"
    elif x < 60 :
        return "Class 6"
    elif x < 70 :
        return "Class 7"
    elif x < 80 :
        return "Class 8"
    elif x < 100 :
        return "Class 9"
    elif x < 200 :
        return "Class 10"
    else :
        return "Class 11"



def create_classes(data):

    df = pd.read_csv(data)

    Classes = df["Participants"]

    Classes = (Classes.apply(define_classes)).to_list()
    Classes = pd.DataFrame(Classes, columns=['Classes'])

    print(Classes)

    df = df.join(Classes)

    df = df.drop(columns=['Participants'])

    df.to_csv('/Users/louise.hubert/PycharmProjects/training_predictions/data/processed_data_classes.csv', index=False)

create_classes('/Users/louise.hubert/PycharmProjects/training_predictions/data/processed_data.csv')