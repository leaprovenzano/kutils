from kutils import metrics
from keras.losses import binary_crossentropy


def soft_f1_loss(y_true, y_pred):
    return 1 - metrics.d_fbeta_score(y_true, y_pred)


def soft_f2_loss(y_true, y_pred):
    return 1 - metrics.d_fbeta_score(y_true, y_pred, 2)


def dice_coef_loss(y_true, y_pred):
    return -metrics.dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    """this was written by keras user bguberfain see: https://www.kaggle.com/bguberfain/naive-keras"""
    return binary_crossentropy(y_true, y_pred) - metrics.dice_coef(y_true, y_pred)
