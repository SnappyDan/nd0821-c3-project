# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pandas as pd
from joblib import dump, load
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Add code to load in the data.
data = pd.read_csv('../data/census_cor.csv')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
logger.info("Splitting data ...")
train, test = train_test_split(data, test_size=0.20)
logger.inof("Done")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
logger.info("Preprocessing training data ...")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
logger.info("SUCCESS: Train data is preprocessed.")
# Proces the test data with the process_data function.
logger.info("Preprocessing test data ...")
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)
logger.info("SUCCCES: Test data is preprocessed")
# Train and save model and encoder.
logger.info("Training the model ...")
model = train_model(X_train, y_train)
logger.info("SUCCESS")
MODEL_PATH = '../model'
MODEL_NAME = 'cl_model.joblib'
ENCODER_NAME = 'encoder.joblib'
logger.info("Saving model and encoder ...")
dump(model, os.path.join(MODEL_PATH, MODEL_NAME))
dump(encoder, os.path.join(MODEL_PATH, ENCODER_NAME))

# Load the model, get the inference on the test set and model metrics
loaded_model = load(os.path.join(MODEL_PATH, MODEL_NAME))
preds = inference(loaded_model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)
logger.info(f"Precision: {precision}, Recall: {recall}, Fbeta: {fbeta}")
