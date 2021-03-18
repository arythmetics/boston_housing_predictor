import numpy as np


class MockModel:
    def __init__(self):
        self.model = None

    def train(self, X: np.ndarray, y: np.ndarray):
        return self

    def test(self):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_instances = len(X)
        return np.random.rand(n_instances)

    def save(self):
        pass

    def load(self):
        return self