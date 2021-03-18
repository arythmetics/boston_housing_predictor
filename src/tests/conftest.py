import pytest
from starlette.testclient import TestClient

from src.main import app

from src.model.model import Model
from src.tests.mocks import MockModel


def model_override():
    model = MockModel()
    return model


app.dependency_overrides[Model] = model_override


@pytest.fixture()
def test_client():
    return TestClient(app)
