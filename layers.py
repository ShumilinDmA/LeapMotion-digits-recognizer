import numpy as np


class Dense:
    """
    Dense or Linear layer of Neural Net
    """
    def __init__(self, in_dim, out_dim):
        """
        :param in_dim: Input dimension of data
        :param out_dim: Output dimension of data
        """
        self.in_dim = in_dim
        self.W = np.random.normal(0, 0.1, (in_dim + 1, out_dim))  # Weight matrix
        self.out = None
        self.grad_X = None
        self.grad_W = None
        self.X = None

    def forward(self, X):
        """
        Forward step through layer
        :param X: Data to process
        :return: Result of processing data
        """
        self.X = np.hstack((X, np.ones((X.shape[0], 1))))  # Add bias term to data matrix
        self.out = self.X.dot(self.W)
        return self.out

    def _backward(self, grad, lr, reg):
        """
        Calculate gradient and make backward step
        :param grad: Gradient from previous layer
        :param lr: Learning rate
        :param reg: Regularization coefficient
        :return: Gradient to a next layer
        """
        self.grad_W = self.X.T.dot(grad)  # Gradient to weight matrix
        self.grad_W[:-1] += 2 * reg * self.W[:-1]  # L2 regularization

        self.W = self.W - lr * self.grad_W  # Update weight

        self.grad_X = grad.dot(self.W[:-1].T)  # Calculate grad to next layer
        return self.grad_X


class LeakyReLU:
    """
    Activation layer
    Function f(x) = X>0, X  |
                    X<0, a*x|
    """
    def __init__(self, a):
        """
        :param a: Coefficient to negative data
        """
        self.a = a
        self.grad = None
        self.out = None

    def forward(self, X):
        """
        Forward step through layer
        :param X: Data to process
        :return: Result of processing data
        """
        temp = X.copy()  # Make sure that we will not change X
        temp[np.where(temp < 0)] = temp[np.where(temp < 0)] * self.a  # multiplication on a
        self.out = temp
        return self.out

    def _backward(self, grad):
        """
        Calculate gradient and make backward step
        :param grad: Gradient from previous layer
        :return: Gradient to the next layer
        """
        temp = grad.copy()
        temp[np.where(self.out < 0)] *= self.a
        self.grad = temp
        return temp


class Softmax:
    """
    Process softmax function to the given data
    """
    def __init__(self):
        self.inp = None
        self.out = None

    def forward(self, X):
        """
        Forward step through layer
        :param X: Data to process
        :return: Chances/probability
        """
        self.inp = X
        exp = np.exp(X)
        self.out = (exp.T / exp.sum(axis=1)).T
        return self.out


class CrossEntropy:
    """
    Cross entropy loss function
    """
    def __init__(self):
        self.loss = None
        self.grad = None

    def forward(self, pred, y):
        """
        Forward step through layer to get loss value
        :param pred: Output from softmax layer
        :param y: Ground truth labels
        :return: Loss
        """
        self.loss = np.mean(-y * np.log(pred))
        return self.loss

    def _backward(self, pred, y):
        """
        Calculate gradient and make backward step.
        Gradient from Loss function and softmax.
        :param pred: Output from previous layer
        :param y: Ground truth label
        :return: Gradient to the next layer
        """
        self.grad = pred - y
        return self.grad
