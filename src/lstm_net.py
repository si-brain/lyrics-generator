import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, tokens, n_hidden_units=256, n_layers=2, dropout=0.5):
        super().__init__()
        self.n_hidden_units = n_hidden_units
        self.n_layers = n_layers
        self.dropout = dropout

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
        output = self.dropout(output)

        # Reshape output for fully connected layer
        output = output.reshape(output.shape[0]*output.shape[1], self.n_hidden_units)
        output = self.fc(output)

        return output, (h, c)

    def init_hidden(self, n_seqs: int):
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, n_seqs, self.n_hidden_units).zero_(),
                weight.new(self.n_layers, n_seqs, self.n_hidden_units).zero_())