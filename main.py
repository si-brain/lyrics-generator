import numpy as np
from src.model import Model


def get_txt():
    text = None
    with open('data/sekspir.txt', 'r') as f:
        text = f.read()

    return text


if __name__ == '__main__':

    # Testing on some random data
    text = get_txt()

    chars = tuple(set(text))
    int_to_char = dict(enumerate(chars))
    char_to_int = {char: integer for integer, char in int_to_char.items()}

    encoded = np.array([char_to_int[ch] for ch in text])

    model = Model(chars, dropout=0, n_hidden=128, n_layers=1)

    # model.train(encoded, n_epochs=1000)

    model.load('models/model.pth')
    output = model.sample(70, 'The')
    print(output)