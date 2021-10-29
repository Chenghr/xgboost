import numpy as np


class Loss(object):
    def loss(self, y_label, y_pred):
        return NotImplementedError()

    def gradient(self, y_label, y_pred):
        """Calculate the gradient of y_pred
        """
        return NotImplementedError()

    def hessian(self, y_label, y_pred):
        """Calculate the Second derivative of y_pred
        """
        return NotImplementedError()


class SquareLoss(Loss):
    """The loss function of the regression task
    """

    def __init__(self):
        pass

    def loss(self, y_label, y_pred):
        return np.array(0.5 * np.power((y_label - y_pred), 2))

    def gradient(self, y_label, y_pred):
        return np.array(-(y_label - y_pred))

    def hessian(self, y_label, y_pred):
        return np.ones_like(y_label)


class BinaryLoss(Loss):
    """LogLoss , The loss function of Binary classification
    """

    def __init__(self):
        pass

    def loss(self, y_label, y_pred):
        return y_label * np.log(1 + np.exp(-y_pred)) + (1-y_label) * np.log(1+np.exp(y_pred))

    def gradient(self, y_label, y_pred):
        return np.array(y_pred - y_label)

    def hessian(self, y_label, y_pred):
        hessian = y_pred * (1 - y_pred)
        hessian = np.array(hessian)
        
        # Avoid having all the second derivatives equal to 0
        hessian[hessian == 0] = float(1e-16)
        return hessian
