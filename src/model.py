from utils import one_hot_encode

import numpy as np

import torch
import torch.nn as nn
import torch.functional as F


class LSTM(nn.Module):

    def __init__(self, tokens, n_hidden_units=256, n_layers=2, dropout=0.5, learning_rate=0.001):
        super().__init__()
        self.n_hidden_units = n_hidden_units
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate

        self.chars = tokens
        self.int_to_char = dict(enumerate(self.chars))
        self.char_to_int = {char: integer for integer, char in self.int_to_char.items()}

        # Define LSTM with dropout
        self.lstm = nn.LSTM(len(self.chars), n_hidden_units, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # Fully connected output layer | input: outputs from LSTM, output: softmax over chars
        self.fc = nn.Linear(n_hidden_units, len(self.chars))

    def forward(self, x, hc):

        # Get output, hidden state and cell state from the output of LSTM, apply dropout
        output, (h, c) = self.lstm(x, hc)
        output = self.dropout(x)

        # Reshape output for fully connected layer
        output = output.reshape(output.shape[0]*output.shape[1], self.n_hidden)
        output = self.fc(output)

        return output, (h, c)

    def init_hidden(self, n_seqs: int):
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, n_seqs, self.n_hidden).zero_(),
                weight.new(self.n_layers, n_seqs, self.n_hidden).zero_())

    def predict(self, char, h=None):
        """
        Given a character x predict the next character.

        Parameters:
        char -- input character
        h -- hidden state

        Returns:
        encoded_value -- encoded value of predicted character
        h -- new hidden state
        """
        if h is None:
            h = self.init_hidden(1)

        x = np.array([[self.char_to_int[char]]])    # needs to be 2dim array
        x = one_hot_encode(x, len(self.chars))

        inputs = torch.from_numpy(x)
        h = tuple([each.data for each in h])
        output, h = self.forward(inputs, h)

        p = F.softmax(output, dim=1).data
        top_ch = np.arange(len(self.chars))
        char = np.random.choice(top_ch, p=p/p.sum())
        encoded_value = self.int_to_char[char]
        return encoded_value, h