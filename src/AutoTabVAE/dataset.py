import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    """
    Generic dataset wrapper for tabular data compatible with AutoTabVAE.

    This dataset accepts NumPy arrays or PyTorch tensors for inputs (X)
    and optionally for targets (y), enabling use in both supervised
    and unsupervised settings.

    Parameters
    ----------
    X : array-like (numpy.ndarray or torch.Tensor)
        Input features.
    y : array-like (numpy.ndarray or torch.Tensor), optional
        Target values (for regression). If None, only X is returned.
    """
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]
