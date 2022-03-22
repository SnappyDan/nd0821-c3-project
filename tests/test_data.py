from starter.ml.data import process_data
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def test_raw_data(data):

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

    assert set(data.columns.values).issuperset(set(cat_features))


def test_process_data(data):

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

    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)

    X, y, n_encoder, n_lb = process_data(
        data, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )

    assert n_encoder == encoder
    assert n_lb == n_lb
