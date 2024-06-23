import pickle
from flask import Flask, request, jsonify

with open('lin_reg2.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)  # this loads the dictiionary vectorizer and the lr model


# create function to prepare the raw features into engineered features we used during the model training
def prepare_features(ride):
    features = {}   # empty dictionary to start with
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])  # This is to concatenate the values to skip python complaining
    features['trip_distance'] = ride['trip_distance'] # we did not transform this feature
    return features


# create a function to predict on new features
def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    
    return preds[0]

# creating a flask application
app =  Flask('duration-prediction')  # name of the app


# creating the function to use flask app to create web-service for the application.
@app.route('/predict', methods=['POST'])  # adding flask decorator to add extra functionalities from flask framework
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)

    pred = predict(features)

    result = {
        'duration': pred
    }
    return result

# run the flask application and expose port
if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=9696)