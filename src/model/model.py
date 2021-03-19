import mlflow
from mlflow.tracking import MlflowClient

import joblib
import importlib
from pathlib import Path

from src.model.nn import nn
from src.data.dataset import inputDataset


class Model:

    # Model class is the main abstraction that interfaces with the API.
    # All future models should match the methods here and mirror them respectively.
    
    def __init__(self, load=True):
        self.X_train, self.X_test, self.n_inputs = inputDataset().prepare_data()
        self.model = nn(self.n_inputs)
        self.root_path = Path(__file__).parent.parent.parent
        # This loads 
        if load==True:
            self.load("9123751989ba44508a0a1b2d5a2cb8bb")
    
    def train(self):
        self.model.train(self.X_train)
    
    def test(self):
        # Store metrics that will be saved when model is trained.
        self.metrics = self.model.test(self.X_test)
    
    def predict(self, row):
        pred = self.model.predict(row)
        return pred

    def save(self):
        if self.model is not None:
            joblib.dump(self.model, Path(__file__).parent / "model_objects" /"ml_model.joblib")
        else:
            raise TypeError("There is no model object. Train the model with model.train() first or ensure artifact is the correct uri.")
    
    def load(self, artifact_uri):
        # will try and load an existing trained model artifact if it exists.
        try:
            self.model = joblib.load(self.root_path / "mlruns" / "0" / artifact_uri / "artifacts" / "ml_model.joblib")
        except:
            self.model = None
            print("No model at this artifact_uri. Please train a new model or select an existing model artifact")

def get_model():
    # Easy to import function that can retrieve the model. Primarily used to load into the API
    model = Model()
    return model


if __name__ == "__main__":
    # Model can be retrained by running this file, if required. NN has a stochastic quality and results can vary.
    model = Model(load=False)
    with mlflow.start_run() as run:
        model.train()
        model.test()
        model.save()
        mlflow.log_artifact(Path(__file__).parent / "model_objects" /"ml_model.joblib")
        mlflow.log_metrics(model.metrics)
        row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
        model.predict(row)
