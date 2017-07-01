from kutils.metrics import d_fbeta_score



def soft_f1_loss(y_true, y_pred):
    return 1 - d_fbeta_score(y_true, y_pred)

def soft_f2_loss(y_true, y_pred):
    return 1 - d_fbeta_score(y_true, y_pred, 2)
