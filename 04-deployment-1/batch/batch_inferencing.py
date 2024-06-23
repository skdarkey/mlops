import pandas as pd
import uuid
import pickle
import mlflow
import os
import sys

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline


# functions
# function to generate uuid
def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids
    

def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # assigning unique ids to the rows in the df
    df['ride_id'] = generate_uuids(len(df))

    return df

def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


# functionize the model application steps

def load_model(run_id):
    logged_model = f'02-exp-tracking/mlruns/1/{run_id}/artifacts/models_mlflow/model.xgb'  # the 1 is for the experiment id
    model = mlflow.pyfunc.load_model(logged_model)
    return model 


def apply_model(input_file, run_id, model_version, output_file):
    df = read_dataframe(input_file) # getting the data
    dicts = prepare_dictionaries(df)

    model = load_model(run_id)
    y_pred = model.predict(dicts)

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id
    # save new data 
    df_result.to_parquet(output_file, index=False)
    
def run():
    # parameterizing the input and output data 
    taxi_type = sys.argv[1]  # 'green'
    year = int(sys.argv[2])  # 2021
    month = int(sys.argv[3]) # 3

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'

    # other parameters
    # RUN_ID = os.getenv('RUN_ID', '74de625dfdf14d32a8a939330f51c35a')
    run_id = sys.argv[4] # 


    # apply the model
    apply_model(input_file= input_file,
                run_id= run_id,
                output_file=output_file)
    

if __name__ == "__main__":
    run()

# run the script in terminal and pass in arguments for taxi_type, year, month and run_id. save the 
# output in a good storage location like in an s3 bucket.
# python batch_inferencing.py green 2021 3 74de625dfdf14d32a8a939330f51c35a


