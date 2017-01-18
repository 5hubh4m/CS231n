import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    num_train = X.shape[0]
    num_classes = W.shape[1]

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # Calculate scores for each class
    scores = np.dot(X, W)
    scores -= np.max(scores, axis=1)[:, np.newaxis]

    # Iterate for each image in input tensor X
    for i in np.arange(0, num_train):
        exp_sum = np.sum(np.exp(scores[i]))
        loss += -scores[i][y[i]] + np.log(exp_sum)

        # Calculate the contribution of each class to overall gradient
        for j in np.arange(0, num_classes):
            dW[:, j] += X[i] * ((np.exp(scores[i][j]) / exp_sum) - (j == y[i]))

    # Average and add a regularisation term
    loss = loss / num_train + 0.5 * reg * np.sum(W * W)
    dW = dW / num_train + reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    num_train = X.shape[0]
    num_classes = W.shape[1]

    # Initialize the loss and gradient to zero.
    loss = np.float64(0)
    dW = np.zeros_like(W, dtype=np.float64)
    scores = np.zeros((num_train, num_classes), dtype=np.float64)

    scores += np.dot(X, W)
    scores -= np.max(scores, axis=1)[:, np.newaxis]

    exp_scores = np.exp(scores)
    loss = np.sum(-scores[np.arange(0, num_train), y] + np.log(np.sum(exp_scores, axis=1)))
    loss = loss / num_train + 0.5 * reg * np.sum(np.square(W))

    exp_scores /= np.sum(exp_scores, axis=1)[:, np.newaxis]
    exp_scores[np.arange(0, num_train), y] -= 1

    dW = np.dot(X.T, exp_scores)
    dW = dW / num_train + reg * W

    return loss, dW