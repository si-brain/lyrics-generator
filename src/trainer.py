from src.utils import get_batches, one_hot_encode

import numpy as np
import torch
import torch.nn as nn


class TrainerConfig:
    # optimization parameters
    n_epochs = 150
    batch_size = 64  # = number of sequences
    seq_length = 160
    learning_rate = 0.001
    clip = 5
    validation_fraction = 0.1
    # checkpoint parameters
    print_every = 10
    save_path = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, epoch, model_dict, optimizer_dict):
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_dict,
            'optimizer_state_dict': optimizer_dict,
        }, self.config.save_path)

    def train(self):
        model, config, data = self.model, self.config, self.data
        valid_idx = int(len(data) * (1 - config.validation_fraction))
        train_data, test_data = data[:valid_idx], data[valid_idx:]

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        step_counter = 0
        n_chars = len(model.chars)
        for epoch in range(config.n_epochs):

            h = model.init_hidden(config.batch_size)
            for x, y in get_batches(train_data, config.batch_size, config.seq_length):
                x = one_hot_encode(x, n_chars)
                inputs, targets = torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)
                h = tuple([each.data for each in h])
                model.zero_grad()
                output, h = model(inputs, h)
                loss = loss_fn(output, targets.view(config.batch_size * config.seq_length).long())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.clip)
                optimizer.step()

                if step_counter % config.print_every == 0:
                    print(f"Epoch: {epoch} / {config.n_epochs}, "
                          f"Step: {step_counter}..., "
                          f"Loss: {loss.item()}, ")
                step_counter += 1

            # Get validation loss and save checkpoint every epoch
            model.eval()
            h_valid = model.init_hidden(config.batch_size)
            valid_losses = []
            for x, y in get_batches(test_data, config.batch_size, config.seq_length):
                x = one_hot_encode(x, n_chars)
                inputs, targets = torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)
                h_valid = tuple([each.data for each in h_valid])
                output, h_valid = model(inputs, h_valid)
                loss_valid = loss_fn(output, targets.view(config.batch_size * config.seq_length).long())
                valid_losses.append(loss_valid.item())

            print(f"Epoch: {epoch + 1} / {config.n_epochs}, "
                  f"Validation Loss: {np.mean(valid_losses)}")
            self.save_checkpoint(epoch, model.state_dict(), optimizer.state_dict())
            model.train()  # back to train mode after validation
