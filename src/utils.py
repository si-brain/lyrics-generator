import numpy as np


def one_hot_encode(arr: list, n_labels: int):
    """
    Encode (2dim) list into one hot representation

    Parameters:
    arr -- 2dim array [n_seqs, n_steps]
    n_labels -- number of labels

    Returns:
    one_hot -- one hot representation of arr
    """
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot


def get_batches(arr: list, n_seqs: int, n_steps: int):
    """
    Create a generator that returns batches of size n_seqs x n_steps from arr.

    Parameters:
    arr -- Array you want to make batches from
    n_seqs -- Number of sequences per batch
    n_steps -- Number of sequence steps per batch
    """
    batch_size = n_seqs * n_steps
    n_batches = len(arr) // batch_size

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size]

    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):

        # n - starting index of the mini batch
        x = arr[:, n:n + n_steps]

        # The targets, shifted by one
        y = np.zeros_like(x)

        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_steps]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y