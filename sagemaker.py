import sagemaker
import sagemaker.amazon.common as smac
from sagemaker import get_execution_role
from sagemaker.predictor import csv_serializer, json_deserializer
# Import the get_image_uri module from the sagemaker library
from sagemaker.amazon.amazon_estimator import get_image_uri
# Import AWS Python SDK
import boto3

# Import support libraries
import io
import os
import json
import numpy as np

# function to store both the train and test data into S3 bucket using special format
def store_data_s3(bucket, prefix, role, data_dictionary):

    # Encode the training data as Protocol Buffer
    X_train = data_dictionary['X_train']
    y_train = data_dictionary['y_train']
    s3_train_data = encode_data(bucket, prefix, role, "model_train.data", X_train, y_train, "train")
    
    # Encode the testing data as Protocol Buffer
    X_test = data_dictionary['X_test']
    y_test = data_dictionary['y_test']
    s3_test_data = encode_data(bucket, prefix, role, "model_test.data", X_test, y_test, "test")
    
    return s3_train_data,s3_test_data 
        
# function to Encode the data as Protocol Buffer
def encode_data(bucket, prefix, role, key, df_X_data, df_y_data, train_or_test):
    
    buf = io.BytesIO()
    vectors = np.array(df_X_data).astype("float32")
    labels = np.array(df_y_data).astype("float32")
    smac.write_numpy_to_dense_tensor(buf, vectors, labels)
    buf.seek(0)        

    boto3.resource("s3").Bucket(bucket).Object(os.path.join(prefix, train_or_test, key)).upload_fileobj(buf)
    s3_data = "s3://{}/{}/{}/{}".format(bucket, prefix, train_or_test,key)
    print("{} data uploaded to: {}".format(train_or_test,s3_data))
    return s3_data
        

# function to prepare learner model
def create_model(bucket, prefix, role,model_type, instance_type,feature_dim):
    
    sess = sagemaker.Session()
    # Import the container image
    container = get_image_uri(boto3.Session().region_name, model_type)

    model_learner = sagemaker.estimator.Estimator(
      container,
      role,
      train_instance_count=1,
      train_instance_type="ml.m4.xlarge",
      output_path="s3://{}/{}/output".format(bucket, prefix),
      sagemaker_session=sess
      )

    # Define linear learner hyperparameters
    model_learner.set_hyperparameters(
        feature_dim=feature_dim,
        mini_batch_size=200,
        predictor_type="binary_classifier"
    )
    
    return model_learner


# function to fit the leaner model with train and test data
def fit_model(model_learner, s3_train_data,s3_test_data):
    # Fitting the learner model
    model_learner = model_learner.fit({"train": s3_train_data, "test": s3_test_data})
    return model_learner
    
# Deploy an instance of the learner model to create a predictor model
def deploy_model(model_learner, instance_type):
    model_predictor = model_learner.deploy(initial_instance_count=1, instance_type=instance_type)
    
# predict using predictor model
def predict(model_predictor,X_test_scaled):
    # predictor configurations
    model_predictor.serializer = csv_serializer
    model_predictor.deserializer = json_deserializer
    # Making some predictions using the test data
    model_predictions = model_predictor.predict(X_test_scaled)
    # Creating a list with the predicted values
    y_predictions = [np.uint8(value["predicted_label"]) for value in model_predictions["predictions"]]
    # Transforming the list into an array
    y_predictions = np.array(y_predictions)

    return y_predictions

# Delete Amazon SageMaker endpoint
def delete_endpoint(model_predictor):
    sagemaker.Session().delete_endpoint(model_predictor.endpoint)