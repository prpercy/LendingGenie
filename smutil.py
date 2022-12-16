'''
    SageMaker modelling file
    
    Steps to follow:
    1. store data in s3 using special format (use function store_data_s3)
    2. create model (use create_model function to create it)
    3. fit the model
    4. deploy model
    5. predict using the model
    6. Evaluate
    7. Delete the SageMaker endpoint
    
'''

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
    s3_train_data = encode_data_pbuf(bucket, prefix, role, "model_train.data", X_train, y_train, "train")
    
    # Encode the testing data as Protocol Buffer
    X_test = data_dictionary['X_test']
    y_test = data_dictionary['y_test']
    s3_test_data = encode_data_pbuf(bucket, prefix, role, "model_test.data", X_test, y_test, "test")
    
    return s3_train_data,s3_test_data 
        
# function to Encode the data as Protocol Buffer
def encode_data_pbuf(bucket, prefix, role, key, df_X_data, df_y_data, train_or_test):
    
    buf = io.BytesIO()
    vectors = np.array(df_X_data).astype("float32")
    labels = np.array(df_y_data).astype("float32")
    smac.write_numpy_to_dense_tensor(buf, vectors, labels)
    buf.seek(0)        

    boto3.resource("s3").Bucket(bucket).Object(os.path.join(prefix, train_or_test, key)).upload_fileobj(buf)
    s3_data = "s3://{}/{}/{}/{}".format(bucket, prefix, train_or_test,key)
    print("{} data uploaded to: {}".format(train_or_test,s3_data))
    return s3_data
        

# function to prepare model
def create_model(bucket, prefix, role,model_type, instance_type,hyperparams):
    
    sess = sagemaker.Session()
    # Import the container image
    container = get_image_uri(boto3.Session().region_name, model_type)

    model_learner = sagemaker.estimator.Estimator(
      container,
      role,
      train_instance_count=1,
      train_instance_type=instance_type,
      output_path="s3://{}/{}/output".format(bucket, prefix),
      sagemaker_session=sess
      )

    # Define learner hyperparameters
    model_learner.set_hyperparameters(**hyperparams)
    
    return model_learner


# function to fit the leaner model with train and test data
def fit_model(model_learner, s3_train_data,s3_test_data):
    # Fitting the learner model
    model_learner.fit({"train": s3_train_data, "test": s3_test_data})
    return model_learner
    
# Deploy an instance of the learner model to create a predictor model
def deploy_model(model_predictor, instance_type):
    model_predictor.deploy(initial_instance_count=1, instance_type=instance_type)
    # predictor configurations
    model_predictor.serializer = csv_serializer
    model_predictor.deserializer = json_deserializer
    return model_predictor
    
# predict using predictor model
def predict(model_predictor,X_test_scaled):
    # Making some predictions using the test data
    model_predictions = model_predictor.predict(X_test_scaled)
    # Creating a list with the predicted values
    y_predictions = [np.uint8(value["predicted_label"]) for value in model_predictions["predictions"]]
    # Transforming the list into an array
    y_predictions = np.array(y_predictions)
    return y_predictions

# evaluation of the model
def evaluate(model_predictor, X_test_scaled, y_test, model_name, verbose=True):
    """
    Evaluate a model on a test set given the prediction endpoint.  Return binary classification metrics.
    """
    y_predictions = predict(model_predictor,X_test_scaled)

    # calculate true positives, false positives, true negatives, false negatives
    tp = np.logical_and(y_test, y_predictions).sum()
    fp = np.logical_and(1 - y_test, y_predictions).sum()
    tn = np.logical_and(1 - y_test, 1 - y_predictions).sum()
    fn = np.logical_and(y_test, 1 - y_predictions).sum()

    # calculate binary classification metrics
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    f1 = 2 * precision * recall / (precision + recall)

    if verbose:
        print(pd.crosstab(y_test, y_predictions, rownames=["actuals"], colnames=["predictions"]))
        print("\n{:<11} {:.3f}".format("Recall:", recall))
        print("{:<11} {:.3f}".format("Precision:", precision))
        print("{:<11} {:.3f}".format("Accuracy:", accuracy))
        print("{:<11} {:.3f}".format("F1:", f1))

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "F1": f1,
        "Model": model_name,
    }

# Delete Amazon SageMaker endpoint
def delete_endpoint(model_predictor):
    try:
        model_predictor.delete_model()
        sagemaker.Session().delete_endpoint(model_predictor.endpoint)
        print("Deleted {}".format(model_predictor.endpoint))
    except:
        print("Already deleted: {}".format(model_predictor.endpoint))
