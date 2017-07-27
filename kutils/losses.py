from kutils import metrics



def soft_f1_loss(y_true, y_pred):
    return 1 - metrics.d_fbeta_score(y_true, y_pred)

def soft_f2_loss(y_true, y_pred):
    return 1 - metrics.d_fbeta_score(y_true, y_pred, 2)

def dice_coef_loss(y_true, y_pred):

    return -metrics.dice_coef(y_true, y_pred)
