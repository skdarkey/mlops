# this script is to manually test the prediction file to see if everything works.
import predict
import requests

# testing before webservice
ride = {
    "PULocationID" : 20,  # this is an example case for a ride to predict on.
    "DOLocationID" : 100,
    "trip_distance" : 100,
}
# features = predict.prepare_features(ride)

# pred = predict.predict(features)
# print(pred)

# testing after web-service
url = 'http://localhost:9696/predict'    # specify the endpoiint "predict" which is used in the flask app
response = requests.post(url, json=ride)
print(response.json())

