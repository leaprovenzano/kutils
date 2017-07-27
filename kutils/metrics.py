from keras import backend as K


def d_precision(y_true, y_pred):
    '''this is basically precision metric from keras 1. but 
    I've attempted to make it differentiable
    '''
    true_positives = K.sum(K.clip(y_true * y_pred, 0, 1))
    predicted_positives = K.sum(K.clip(y_pred, 0, 1))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def d_recall(y_true, y_pred):
    '''this is basically reall metric from keras 1. but 
    I've attempted to make it differentiable.
    '''
    true_positives = K.sum(K.clip(y_true * y_pred, 0, 1))
    possible_positives = K.sum(K.clip(y_true, 0, 1))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def d_fbeta_score(y_true, y_pred, beta=1):
    """this is basically fbeta from keras 1. but 
    I've attempted to make it differentiable.
    """
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = d_precision(y_true, y_pred)
    r = d_recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def dice_coef(y_true, y_pred):
    intersection = K.sum(K.flatten(y_true) * K.flatten(y_pred), axis=[0, 1, 2])
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


