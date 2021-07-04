import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils import load_data, preproccess_data, save_model

TARGET_COLUMN = "failure"
FEATURE_FILTER = "metric"
#Function to make prediction with ensemble
def ensebmle_predict(ensemble, features):
    predictions = []

    # Go over all models in our ensemble
    for model in ensemble:
        # Get Prediction
        prediction = model.predict([features])[0]

        # Store prediction
        predictions.append(prediction)

    # Get Max Prediction
    prediction = np.argmax(np.bincount(predictions))
    return prediction

#Loads data,
def train_ensemble(data_file, model_dir, number_of_models=10):

    #Iterating over number of models in ensemble
    for index in range(number_of_models):
        #laod data into df
        df = load_data(data_file)

        #prerpocess data
        features, target = preproccess_data(df, TARGET_COLUMN, FEATURE_FILTER)

        model = train_model(features, target)

        save_model(model, model_dir, f"forest_model_{index+1}.sav")

#Function to train new model on provided data
def train_model(features, target):
    forest = RandomForestClassifier(bootstrap=False,
                                    max_depth=50,
                                    min_samples_leaf=1,
                                    min_samples_split=5,
                                    n_estimators=1200)

    forest.fit(features, target)

    return forest