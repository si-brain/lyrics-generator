import torch
import torch.nn as nn

from utils import get_batches, one_hot_encode


def train(model, data, n_epochs=10, n_seqs=10, n_steps=50, learning_rate=0.001, clip=5):
    """
    Trains the LSTM model

    Parameters:
    model -- LSTM model
    data -- characters to train the network
    n_epochs -- number of training epochs
    n_seqs -- numbber of sequences per batch
    n_steps -- number of steps per sequence (sequence length)
    learning_rate -- learning rate for the optimizer
    clip -- clipping rate

    TODO: treba da dodam neku validaciju ovo ono pa da lepo ispisujem loss, ovo je samo test
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

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
            print(loss.item())