from math import ceil

import numpy as np 



def split_byc(X, Y, split=.2):
    """split best you can : tries to do the right 
    thing and split by given split parameter, otherwise always puts
    at least one value in validation set. 
    """
    if len(Y.shape) > 1: #its one hot
        flat_y = Y.argmax(axis=1)
    else:
        flat_y = Y
    
    train_indices = []
    val_indices = []
    
    for indices in (np.argwhere(i==flat_y) for i in np.unique(flat_y)):
        indices = indices.flatten()
        n = indices.shape[0]
        n_val = ceil(n*split)
        vi = np.random.choice(indices, n_val, replace=False)
        train_indices+= list(indices[~np.isin(indices, vi)])
        val_indices += list(vi)
    return (X[train_indices], Y[train_indices]), (X[val_indices], Y[val_indices])
