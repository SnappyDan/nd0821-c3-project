from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello! Please use /model entrypoint to get predictions"}


def test_predict_positive():
    request_item = {
        "age": 42,
        "fnlgt": 159449,
        "education": "Bachelors",
        "education-num": 13,
        "capital-gain": 5178,
        "capital-loss": 0,
        "hours-per-week": 40,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native-country": "United-States",
        "workclass": "Private"         
    }
    response = client.post("/model", json=request_item)
    assert response.status_code == 200
    assert response.json() == 1


def test_predict_negative():
    response = client.get("/model")

    request_item = {
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
    response = client.post("/model", json=request_item)
    print(response)
    assert response.status_code == 200
    assert response.json() == 0