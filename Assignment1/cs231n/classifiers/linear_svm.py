import numpy as np
from random import shuffle, randint

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    num_class_incorrect = 0

    for j in xrange(num_classes):
      if j == y[i]:
        continue

      margin = scores[j] - scores[y[i]] + 1 # note delta = 1
      if margin > 0:
        num_class_incorrect += 1
        loss += margin
        dW[:, j] += X[i]

    dW[:, y[i]] += -num_class_incorrect * X[i]

  # Average and add regularization to the loss and gradient.
  loss = loss / num_train + 0.5 * reg * np.sum(W * W)
  dW = dW / num_train + reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = np.dot(X, W)                                 # Calculate scores

  # Calculate an ndarray of correct scores
  correct_scores = np.ones(scores.shape) * scores[np.arange(0, num_train), y, np.newaxis]

  # Calculate margin matrix
  margin = np.maximum(0, scores - correct_scores + 1)
  margin[np.arange(0, num_train), y] = 0

  # Calculate the mask martix over training data
  X_mask = np.zeros(margin.shape)
  np.copyto(X_mask, margin)
  X_mask[X_mask > 0] = 1
  X_mask[np.arange(0, num_train), y] = -np.sum(X_mask, axis=1)
  
  # Average across samples and add regularisation term
  loss = (np.sum(margin) / num_train) + (0.5 * reg * np.sum(W * W))
  dW = np.dot(X.T, X_mask) / num_train + reg * W
    
  return loss, dW
