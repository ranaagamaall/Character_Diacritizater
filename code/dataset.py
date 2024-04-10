import torch
from torch import tensor
import pickle


class Letter_Train_Dataset(torch.utils.data.Dataset):

    def __init__(self, x, e, y):
        """
        This is the constructor of the NERDataset
        Inputs:
        - x: a list of lists where each list contains the ids of the tokens
        - e: a list of lists where each list contains hidden states of the tokens
        - y: a list of lists where each list contains the label of each token in the sentence
        """
        self.x = torch.tensor(x)
        self.e = torch.tensor(e)
        self.y = torch.tensor(y)

        #################################################################################################################

    def __len__(self):
        """
        This function should return the length of the dataset (the number of sentences)
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        This function returns a subset of the whole dataset
        """
        return self.x[idx], self.e[idx], self.y[idx]
