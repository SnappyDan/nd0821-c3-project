import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from joblib import load


@pytest.fixture(scope="session")
def data():

    data = pd.read_csv('data/census_cor.csv')

    return data


@pytest.fixture(scope="session")
def data_train():

    train, _ = get_train_test()
    encoder = load('model/encoder.joblib')
    lb = load('model/lb.joblib')
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
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    data_train = (X_train, y_train)
    return data_train


@pytest.fixture(scope="session")
def data_test():

    _, test = get_train_test()
    encoder = load('model/encoder.joblib')
    lb = load('model/lb.joblib')
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
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    data_test = (X_test, y_test)
    return data_test


@pytest.fixture(scope="session")
def model():

    model = load('model/cl_model.joblib')

    return model


@pytest.fixture(scope="session")
def preds():

    model = load('model/cl_model.joblib')
    _, test = get_train_test()
    encoder = load('model/encoder.joblib')
    lb = load('model/lb.joblib')
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
    X_test, _, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    preds = model.predict(X_test)

    return preds


def get_train_test():
    data = pd.read_csv('data/census_cor.csv')
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    return train, test
