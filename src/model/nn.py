from src.model.mlp import MLP

from torch.nn import MSELoss
from torch.optim import SGD
from torch import Tensor

from sklearn.metrics import mean_squared_error
from numpy import vstack


class nn():
    def __init__(self, n_inputs):
        self.model = MLP(n_inputs)

    def train(self, train_dl):
        criterion = MSELoss()
        optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # enumerate epochs
        for epoch in range(100):
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(train_dl):
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = self.model(inputs)
                # calculate loss
                loss = criterion(yhat, targets)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()
    
    def test(self, test_dl):
        predictions, actuals = list(), list()
        for i, (inputs, targets) in enumerate(test_dl):
            # evaluate the model on the test set
            yhat = self.model(inputs)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            actual = actual.reshape((len(actual), 1))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        # calculate mse
        mse = mean_squared_error(actuals, predictions)
        return {"mse": mse}
    
    def predict(self, row):
        # convert row to data
        row = Tensor([row])
        # make prediction
        yhat = self.model(row)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        return yhat
