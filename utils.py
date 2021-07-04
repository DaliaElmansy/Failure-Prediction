import os
import joblib
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

EXPECTED_DATAPOINTS = 10000

#Loading Models
def load_ensemble(model_dir):
    ensemble = []
    for model_file in os.listdir(model_dir):
        model = joblib.load(f"{model_dir}{model_file}")
        ensemble.append(model)

    return ensemble

#Saves trained model
def save_model(model, model_dir, model_name):
    joblib.dump(model, model_dir + model_name)

#Loading Data
def load_data(data_dir):
    data_extension = data_dir.split(".")[-1]
    #Check extension type
    if data_extension == "csv":
        df = pd.read_csv(data_dir)
    elif data_extension == "tsv":
        df = pd.read_csv(data_dir, sep="\t")
    else:
        df = pd.read_excel(data_dir)

    return df

#Normalizing data
def normalize_data(df):
    for column in df:
        # Start MinMaxScaler
        scaler = MinMaxScaler()

        # Fit Scaler on column
        scaler.fit(df[column].values.reshape(-1, 1))

        # Replace column data with transformed data
        df[column] = scaler.transform(df[column].values.reshape(-1, 1))

    return df

#Preprocessing data
def preproccess_data(df, target_col, feature_col_filter):
    #Get Target column
    target = df[target_col]

    #Get feature columns
    features = df[[i for i in df.columns if feature_col_filter in i]]
    #Normalize Features
    features = normalize_data(features)

    #Imbalanced data
    # Undersample Non-Failure Data
    non_failure_indicies = np.where(target == 0)[0]

    # Sample non failure indicies
    non_failure_indicies = np.random.choice(non_failure_indicies, EXPECTED_DATAPOINTS)

    # Collect all failure data
    failure_indicies = np.where(target == 1)[0]

    # Filter data to limited non failure and all failure data
    features = features.iloc[np.append(non_failure_indicies, failure_indicies)]
    target = target.iloc[np.append(non_failure_indicies, failure_indicies)]

    # Oversample Failure Data using SMOTE
    oversample = SMOTE()
    features, target = oversample.fit_resample(features, target)

    return features, target