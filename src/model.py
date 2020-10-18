from src.utils import one_hot_encode
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, tokens, n_hidden_units=512, n_layers=4, dropout=0.5):
        super().__init__()
        self.n_hidden_units = n_hidden_units
        self.n_layers = n_layers
        self.dropout = dropout

        self.chars = tokens
        self.int_to_char = dict(enumerate(self.chars))
        self.char_to_int = {char: integer for integer, char in self.int_to_char.items()}

        self.lstm = nn.LSTM(len(self.chars), n_hidden_units, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_hidden_units, len(self.chars))
        print(f'Number of parameters: {sum(p.numel() for p in self.parameters())}')

    def forward(self, x, hc):

        # Get output, hidden state and cell state from the output of LSTM, apply dropout
        output, (h, c) = self.lstm(x, hc)
        output = self.dropout(output)

        # Reshape output for fully connected layer
        output = output.reshape(output.shape[0]*output.shape[1], self.n_hidden_units)
        output = self.fc(output)
        return output, (h, c)

    def init_hidden(self, n_seqs: int):
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, n_seqs, self.n_hidden_units).zero_(),
                weight.new(self.n_layers, n_seqs, self.n_hidden_units).zero_())

    def predict(self, char, h=None, top_k=None):
        if h is None:
            h = self.net.init_hidden(1)

        x = np.array([[self.char_to_int[char]]])    # needs to be 2dim array
        x = one_hot_encode(x, len(self.chars))
        input = torch.from_numpy(x)
        h = tuple([each.data for each in h])
        output, h = self.forward(input, h)

        p = F.softmax(output, dim=1).data

        if top_k is None:
            top_ch = np.arange(len(self.net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p / p.sum())
        return self.int_to_char[char], h