# Scenario 2: deployment from mlflow 
# Where we saved the dict-vectorizer and the model in a scikit learn pipeline object
# This is a scenario where we run a tracking server with mlflow, set an artifact store for models and other files
# and using the model registry to get model by run id. the tracking server, experiment name and artefact stores are set as below.



import pickle
import mlflow
from flask import Flask, request, jsonify

# instead of loading our model directly, we use the run_id
MLFLOW_TRACKING_URI = 'https://127.0.0.1:5000'
RUN_ID = "7c9fe822ae5847dfa5c930012541f86d/models_mlflow"     # you can change the run id and immediately use another model

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
 

logged_model = f"runs:/{RUN_ID}/model"
model = mlflow.pyfunc.load_model(logged_model)  # look at the mlflow pyfunc documentation for more options to load model


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