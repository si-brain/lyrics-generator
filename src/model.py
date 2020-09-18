from src.utils import one_hot_encode, get_batches   # function pointers
from src.lstm_net import LSTM   # LSTM net

import numpy as np

import torch
import torch.nn as nn


class Model(object):

    def __init__(self, tokens, n_hidden=256, n_layers=2, dropout=0.5):
        """
        Model for training and evaluating LSTM net

        Parameters:
        tokens -- characters
        n_hidden -- number of hidden units
        n_layers -- number of stacked LSTM layers
        dropout -- dropout probability

        Attributes:
        net -- LSTM network
        """
        super(Model, self).__init__()
        self.net = LSTM(tokens, n_hidden_units=256, n_layers=2, dropout=0.5)

    def load(self, path: str):
        """
        Load neural network model

        Parameters:
        path -- file path from which neural network is loaded
        """
        self.net.load_state_dict(torch.load(path))

    def train(self, data, n_epochs=50, n_seqs=10, n_steps=50, learning_rate=0.001, clip=5, data_frac=0.9, print_every=100):
        """
        Trains and saves (every print_every epochs) LSTM network

        Parameters:
        data -- characters to train the network
        n_epochs -- number of training epochs
        n_seqs -- number of sequences per batch
        n_steps -- number of steps per sequence (sequence length)
        learning_rate -- learning rate for the optimizer
        clip -- clipping rate
        data_frac -- train-dev set fraction
        print_every -- number of epochs before printing stats
        """
        self.net.train()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        valid_idx = int(len(data) * (1 - data_frac))
        data, val_data = data[:valid_idx], data[valid_idx:]

        counter = 0
        n_chars = len(self.net.chars)

        for epoch in range(n_epochs):

            h = self.net.init_hidden(n_seqs)

            for x, y in get_batches(data, n_seqs, n_steps):
                counter += 1
                x = one_hot_encode(x, n_chars)

                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                h = tuple([each.data for each in h])

                self.net.zero_grad()
                output, h = self.net.forward(inputs, h)
                loss = criterion(output, targets.view(n_seqs * n_steps).type(torch.LongTensor))
                loss.backward()

                # Clip the gradients
                nn.utils.clip_grad_norm_(self.net.parameters(), clip)

                optimizer.step()

                if counter % print_every == 0:

                    # Get validation loss
                    h_valid = self.net.init_hidden(n_seqs)
                    valid_losses = []

                    for x, y in get_batches(val_data, n_seqs, n_steps):
                        x = one_hot_encode(x, n_chars)
                        x, y = torch.from_numpy(x), torch.from_numpy(y)

                        h_valid = tuple([each.data for each in h_valid])
                        inputs, targets = x, y

                        output, h_valid = self.net.forward(inputs, h_valid)
                        loss_valid = criterion(output, targets.view(n_seqs * n_steps).type(torch.LongTensor))

                        valid_losses.append(loss_valid.item())

                    print(f"Epoch: {epoch + 1} / {n_epochs}, "
                          f"Step: {counter}..., "
                          f"Loss: {loss.item()}, "
                          f"Validation Loss: {np.mean(valid_losses)}")

            torch.save(self.net.state_dict(), 'models/model.pth')

    def predict(self, char, h=None):
        """
        Given a character x predict the next character.

        Parameters:
        char -- input character
        h -- hidden state

        Returns:
        encoded_value -- encoded value of predicted character (integer)
        h -- new hidden state
        """
        if h is None:
            h = self.net.init_hidden(1)

        x = np.array([[self.net.char_to_int[char]]])    # needs to be 2dim array
        x = one_hot_encode(x, len(self.net.chars))

        inputs = torch.from_numpy(x)
        h = tuple([each.data for each in h])
        output, h = self.net.forward(inputs, h)

        softmax = nn.Softmax(dim=1)
        prob = softmax(output).data
        prob = np.array(prob[0])

        top_ch = np.arange(len(self.net.chars))
        char = np.random.choice(top_ch, p=prob)
        encoded_value = self.net.int_to_char[char]
        return encoded_value, h

    def sample(self, size: int, starting_word: str = 'The') -> str:
        """
        Sample from LSTM model

        Parameters:
        size -- length of the output
        starting_word -- word which will be used as first word in the sampled output

        Returns: a bunch of characters that hopefully make sense
        """
        self.net.eval()

        chars = [ch for ch in starting_word]
        h = self.net.init_hidden(1)
        for ch in starting_word:
            char, h = self.predict(ch, h)

        chars.append(char)

        for i in range(size):
            char, h = self.predict(chars[-1], h)
            chars.append(char)

        return ''.join(chars)