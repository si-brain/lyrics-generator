from src.utils import get_batches, one_hot_encode

import torch
import torch.nn as nn

import numpy as np


def train(model, data, n_epochs=50, n_seqs=10, n_steps=50, learning_rate=0.001, clip=5, data_frac=0.9, print_every=10):
    """
    Trains the LSTM model

    Parameters:
    model -- LSTM model
    data -- characters to train the network
    n_epochs -- number of training epochs
    n_seqs -- number of sequences per batch
    n_steps -- number of steps per sequence (sequence length)
    learning_rate -- learning rate for the optimizer
    clip -- clipping rate
    data_frac -- train-dev set fraction
    print_every -- number of epochs before printing stats
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    valid_idx = int(len(data) * (1 - data_frac))
    data, val_data = data[:valid_idx], data[valid_idx:]

    counter = 0
    n_chars = len(model.chars)

    for epoch in range(n_epochs):

        h = model.init_hidden(n_seqs)

        for x, y in get_batches(data, n_seqs, n_steps):
            counter += 1
            x = one_hot_encode(x, n_chars)

            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            h = tuple([each.data for each in h])

            model.zero_grad()
            output, h = model.forward(inputs, h)
            loss = criterion(output, targets.view(n_seqs * n_steps).type(torch.LongTensor))
            loss.backward()

            # Clip the gradients
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            if counter % print_every == 0:

                # Get validation loss
                h_valid = model.init_hidden(n_seqs)
                valid_losses = []

                for x, y in get_batches(val_data, n_seqs, n_steps):

                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    h_valid = tuple([each.data for each in h_valid])
                    inputs, targets = x, y

                    output, h_valid = model.forward(inputs, h_valid)
                    loss_valid = criterion(output, targets.view(n_seqs * n_steps).type(torch.LongTensor))

                    valid_losses.append(loss_valid.item())

                print(f"Epoch: {epoch + 1} / {n_epochs}, "
                      f"Step: {counter}..., "
                      f"Loss: {loss.item()}, "
                      f"Validation Loss: {np.mean(valid_losses)}")
