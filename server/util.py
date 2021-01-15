import json
import pickle
import numpy as np

__locations = None
__model = None
__data_columns = None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index=-1

    X=np.zeros(len(__data_columns))
    X[0]=sqft
    X[1]= bath
    X[2]=bhk
    if loc_index>=0:
        X[loc_index]=1
    return round(__model.predict([X])[0],2)


def get_location_names():
    load_saved_artifacts()
    return __locations

def load_saved_artifacts():
    print("Loading saved Artifacts....start")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json",'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    global __model
    with open("./artifacts/banglore_home_price_model.pickle",'rb') as f:
        __model = pickle.load(f)
    print("Loading Saved artifactes..... done")






