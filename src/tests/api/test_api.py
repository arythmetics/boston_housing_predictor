import pytest
import random
from pathlib import Path
from src.main import app
from src.model.model import Model
from starlette.testclient import TestClient
from starlette.status import HTTP_200_OK, HTTP_422_UNPROCESSABLE_ENTITY
from itertools import product

# Importing training feature dimensions to use as an input for the tests.
n_features = Model().n_inputs


def test_predict(test_client: TestClient):
    # Ensures that the model can predict on the right data
    fake_data = [random.random() for _ in range(n_features)]
    response = test_client.post("/predict", json={"data": fake_data})
    assert response.status_code == HTTP_200_OK


# Parameterize the dimensions for the wrong datasets
@pytest.mark.parametrize(
    "n_instances, test_data_n_features",
    product(range(1, 10), [n for n in range(1, 20) if n != n_features]),
)
def test_predict_with_wrong_input(
    n_instances: int, test_data_n_features: int, test_client: TestClient
):
    # Ensures the API will reject data that does not match the appropriate dimensions
    fake_data = [[random.random() for _ in range(test_data_n_features)] for _ in range(n_instances)]
    response = test_client.post("/predict", json={"data": fake_data})
    assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY
