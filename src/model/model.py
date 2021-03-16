import mlflow
from mlflow.tracking import MlflowClient

import joblib
import importlib
from pathlib import Path

from src.model.nn import nn
from src.data.dataset import inputDataset


class Model:
    
    def __init__(self):
        self.model = nn()
        self.X_train, self.X_test = inputDataset().prepare_data()
        self.root_path = Path(__file__).parent.parent.parent
        self.load("76f2e23fe49748a4b1e65f6aa1560e27")
    
    def train(self):
        self.model.train(self.X_train)
    
    def test(self):
        # Could possibly be multiple metrics - left room for that.
        self.metrics = self.model.test(self.X_test)
    
    def predict(self, row):
        self.model.predict(row)

    def save(self):
        if self.model is not None:
            joblib.dump(self.model, Path(__file__).parent / "model_objects" /"ml_model.joblib")
        else:
            raise TypeError("There is no model object. Train the model with model.train() first.")
    
    def load(self, artifact_uri):
        try:
            self.model = joblib.load(self.root_path / "mlruns" / "0" / artifact_uri / "artifacts" / "ml_model.joblib")
        except:
            self.model = None


if __name__ == "__main__":
    model = Model()
    model.load("76f2e23fe49748a4b1e65f6aa1560e27")
    row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
    model.predict(row)
    # with mlflow.start_run() as run:
    #     model.train()
    #     model.test()
    #     model.save()
    #     mlflow.log_artifact(Path(__file__).parent / "model_objects" /"ml_model.joblib")
    #     mlflow.log_metrics(model.metrics)
    #     row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
    #     model.predict(row)