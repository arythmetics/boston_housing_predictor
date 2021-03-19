import pytest
from starlette.testclient import TestClient

from src.main import app

from src.model.model import Model
from src.tests.mocks import MockModel


def model_override():
    model = MockModel()
    return model

# Overriding the model dependency (Model = Depends(get_model)) in main.py with MockModel for testing
app.dependency_overrides[Model] = model_override

# Redefines test_client when pytest is initiated
@pytest.fixture()
def test_client():
    return TestClient(app)
