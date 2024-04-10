import torch.nn as nn


class Letter_Model(nn.Module):
    def __init__(self, input_size=768, hidden_size=50, n_classes=100):
        """
        The constructor of our NER model
        Inputs:
        - input_size: the number of input features
        - hidden_size: the size of the hidden state
        - n_classes: the number of final classes (tags)
        """
        super(Letter_Model, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size,
                            batch_first=True)

        self.linear = nn.Linear(
            hidden_size, n_classes)

    def forward(self, X, hidden=None):
        """
        This function does the forward pass of our model
        Inputs:
        - X: tensor of shape (batch_size, max_length)
        - hidden: initial hidden state of the LSTM

        Returns:
        - final_output: tensor of shape (batch_size, max_length, n_classes)
        - hidden: the last hidden state of the LSTM
        """

        final_output, hidden = self.lstm(X, hidden)
        final_output = self.linear(final_output)

        return final_output, hidden


class Model(nn.Module):
    def __init__(self, input_size=768, hidden_size=50, n_classes=100, bidirectional=False):
        """
        The constructor of our NER model
        Inputs:
        - vacab_size: the number of unique words
        - embedding_dim: the embedding dimension
        - n_classes: the number of final classes (tags)
        """
        super(Model, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size,
                            batch_first=True, bidirectional=bidirectional)

        self.linear = nn.Linear(
            hidden_size * (2 if bidirectional else 1), n_classes)

    def forward(self, X):
        """
        This function does the forward pass of our model
        Inputs:
        - sentences: tensor of shape (batch_size, max_length)

        Returns:
        - final_output: tensor of shape (batch_size, max_length, n_classes)
        """

        final_output, hidden = self.lstm(X)
        final_output = self.linear(final_output)

        return final_output, hidden
