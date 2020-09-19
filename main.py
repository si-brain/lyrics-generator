import numpy as np
from src.model import Model
from src.dataset import SongDataset
import src.preprocessing as preprocessing


def get_txt():
    with open('data/sekspir.txt', 'r') as f:
        text = f.read()
    return text


def example_train():
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


def example_dataset():
    dataset = SongDataset(output_file_path='data/raw/all_songs.txt',
                          output_dir_path='data/raw/singers',
                          num_singers=4,
                          thread_count=4,
                          preprocessing_ops=[
                              preprocessing.ToLowercaseOp(),
                              preprocessing.FilterLinesOp(["ref.", "("]),
                              preprocessing.LambdaOp(lambda x: x + '\n')
                          ])
    dataset.prepare()


def train_on_song_dataset():
    with open('data/raw/all_songs.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    chars = tuple(set(text))
    int_to_char = dict(enumerate(chars))
    char_to_int = {char: integer for integer, char in int_to_char.items()}

    encoded = np.array([char_to_int[ch] for ch in text])

    model = Model(chars, dropout=0, n_hidden=128, n_layers=1)

    # print('>> Starting training...\n\n')

    # model.train(encoded, n_epochs=1000, save_path='models/test_model.pth')

    model.load('models/test_model.pth')
    output = model.sample(70, 'ja sam')
    print(output)


if __name__ == '__main__':
    # example_train()
    example_dataset()
    # train_on_song_dataset()