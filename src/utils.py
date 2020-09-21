import numpy as np


def one_hot_encode(arr: np.ndarray, n_labels: int) -> np.ndarray:
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot


def get_batches(arr: list, n_seqs: int, seq_length: int) -> (list, list):

    batch_size = n_seqs * seq_length
    n_batches = len(arr) // batch_size

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size]

    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], seq_length):

        # n - starting index of the mini batch
        x = arr[:, n:n + seq_length]

        # The targets, shifted by one
        y = np.zeros_like(x)

        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


def sample(model, size: int, starting_word: str = 'ja', top_k=None) -> str:
    chars = [ch for ch in starting_word]
    h = model.init_hidden(1)

    model.eval()
    for ch in starting_word:
        char, h = model.predict(ch, h, top_k=top_k)

    chars.append(char)

    for i in range(size):
        char, h = model.predict(chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)


