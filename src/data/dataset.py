from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.datasets import load_boston

class inputDataset(Dataset):
    # load the dataset
    def __init__(self):
        # store the inputs and outputs
        self.X, self.y = load_boston(return_X_y=True)
        self.X, self.y = self.X.astype('float32'), self.y.astype('float32')
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 1))
 
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
 
    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
 
    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])
    
    def prepare_data(self):
        # calculate split
        train, test = self.get_splits()
        # prepare data loaders
        train_dl = DataLoader(train, batch_size=32, shuffle=True)
        test_dl = DataLoader(test, batch_size=1024, shuffle=False)
        return train_dl, test_dl
