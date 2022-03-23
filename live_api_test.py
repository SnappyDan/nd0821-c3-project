import requests
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

data_item = {
        "age": 45,
        "fnlgt": 50567,
        "education": "HS-grad",
        "education-num": 9,
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "native-country": "United-States",
        "workclass": "State-gov"  
    }
response = requests.post('https://udacity-mlops2.herokuapp.com/model', json=data_item)

logger.debug(response.json())

print(f"Response Status Code: {response.status_code}")
if response.status_code == 200:
    print(f"Result provided by the model: {response.json()}")
    if response.json() == 1:
        print("Expected salary for this person is more than 50K per year")
    else:
        print("Expected salary for this person is less than 50K per year")