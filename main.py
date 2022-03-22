# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.encoders import jsonable_encoder
from starter.ml.data import process_data
from starter.ml.model import inference

import pandas as pd

from joblib import load
import logging


logging.basicConfig(level=logging.DEBUG, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class CensusDataItem(BaseModel):
    age: int
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str = Field(alias="native-country")
    workclass: str

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "capital-gain": 1234,
                "capital-loss": 0,
                "hours-per-week": 40,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "native-country": "United-States",
                "workclass": "State-gov"
            }
        }


app = FastAPI()


def load_inf_pipeline():
    """Helper function to load model and encoder"""
    model = load('model/cl_model.joblib')
    encoder = load('model/encoder.joblib')
    return encoder, model


@app.get('/')
def root():
    return {"message": "Hello! Please use /model entrypoint to get predictions"}


@app.post('/model')
def predict(cen_data: CensusDataItem):
    df_request = pd.DataFrame.from_dict([jsonable_encoder(cen_data)])
    logger.info(df_request)

    encoder, model = load_inf_pipeline()

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

    X_input, _, _, _ = process_data(
        df_request, categorical_features=cat_features, training=False, encoder=encoder)

    logger.debug(X_input)
    logger.debug(X_input.shape)

    preds = inference(model, X_input)

    logger.debug(f"Predictions: {preds}")
    logger.debug(type(preds))

    return preds.tolist()[0]
