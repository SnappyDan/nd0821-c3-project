import numpy as np
from starter.ml.model import train_model, inference, compute_model_metrics
from sklearn.linear_model import LogisticRegression


def test_train_model(data_train):
    X_train, y_train = data_train
    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression)

def test_inference(model, data_test):
    X_test, _ = data_test
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)

def test_compute_model_metrics(data_test, preds):
    _, y_test = data_test
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(recall, float)




