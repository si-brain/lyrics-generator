from src.model import LSTM

import numpy as np
from src.train import train


def get_txt():
    text = None
    with open('data/test.txt', 'r') as f:
        text = f.read()

    return text


if __name__ == '__main__':
    text = get_txt()
    chars = tuple(set(text))
    int_to_char = dict(enumerate(chars))
    char_to_int = {char: integer for integer, char in int_to_char.items()}

    encoded = np.array([char_to_int[ch] for ch in text])

    model = LSTM(chars)

    n_seqs, n_steps = 18, 30
    train(model, encoded, n_seqs=n_seqs, n_steps=n_steps)