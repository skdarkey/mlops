# Scenario 3: deployment from mlflow 
# Where we saved the dict-vectorizer and the model in a scikit learn pipeline object
# Here we want to load the model directly from the artefact store without relying on the tracking server.
# This circumvents the problem of relying on tracking server when the server has an issue


import pickle
import mlflow
import os
from flask import Flask, request, jsonify

# instead of loading our model directly, we use the run_id
# MLFLOW_TRACKING_URI = 'https://127.0.0.1:5000'
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# RUN_ID = "7c9fe822ae5847dfa5c930012541f86d/models_mlflow"     # you can set the run id as env variable and pass it when deploying.
RUN_ID = os.getenv('RUN_ID')   # run export RUN_ID="7c9fe822ae5847dfa5c930012541f86d"  in terminal when using k8 

logged_model = f's3://mlflow-models-selorm/1/{RUN_ID}/artefacts/model'  # here the 1 refers to the experiment id. Get the full path to
# the model from the mlflow path 

model = mlflow.pyfunc.load_model(logged_model)  # loading the model directly from artefact store


# create function to prepare the raw features into engineered features we used during the model training
def prepare_features(ride):
    features = {}   # empty dictionary to start with
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])  # This is to concatenate the values to skip python complaining
    features['trip_distance'] = ride['trip_distance'] # we did not transform this feature
    return features


# create a function to predict on new features
def predict(features):
    preds = model.predict(X)
    
    return float(preds[0])

# creating a flask application
app =  Flask('duration-prediction')  # name of the app


# creating the function to use flask app to create web-service for the application.
@app.route('/predict', methods=['POST'])  # adding flask decorator to add extra functionalities from flask framework
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)

    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID,    # to return the id of the run.
    }
    return result

# run the flask application and expose port
if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=9696)