import numpy as np


def list_to_padded_array(x_list, pad_type='edge', dt='float32'):
    """takes a list of np arrays of different lengths
    where other dimensions are of same size, returns padded np array w/
    shape (n_items, max_sequence_length, ) + (n_dimensions,)
    pads evenly on start and end w/ greater pads on end if not %2
    """
    n_items = len(x_list)
    max_sequence_length = max(map(lambda x: x.shape[0], x_list))
    other_dims = x_list[0].shape[1:]
    X = np.zeros((n_items, max_sequence_length) + other_dims, dt)
    for i, x in enumerate(x_list):
        pad_start = (max_sequence_length - x.shape[0]) // 2
        pad_end = max_sequence_length - (pad_start + x.shape[0])
        X[i] = np.pad(x, ((pad_start, pad_end), (0, 0)), pad_type)
    return X
