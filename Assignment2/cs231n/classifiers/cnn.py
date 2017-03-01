import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

from fc_net import *


def conv_sbn_relu_pool_forward(x, w, b, gamma, beta,
                               conv_param,
                               bn_param,
                               pool_param):

    out, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, sbn_cache = spatial_batchnorm_forward(out, gamma, beta, bn_param)
    out, relu_cache = relu_forward(out)
    out, pool_cache = max_pool_forward_fast(out, pool_param)

    cache = (conv_cache, sbn_cache, relu_cache, pool_cache)

    return out, cache


def conv_sbn_relu_pool_backward(dout, cache):
    conv_cache, sbn_cache, relu_cache, pool_cache = cache

    dx = max_pool_backward_fast(dout, pool_cache)
    dx = relu_backward(dx, relu_cache)
    dx, dgamma, dbeta = spatial_batchnorm_backward(dx, sbn_cache)
    dx, dw, db = conv_backward_fast(dx, conv_cache)

    return dx, dw, db, dgamma, dbeta


def conv_sbn_relu_forward(x, w, b, gamma, beta,
                               conv_param,
                               bn_param):

    out, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, sbn_cache = spatial_batchnorm_forward(out, gamma, beta, bn_param)
    out, relu_cache = relu_forward(out)

    cache = (conv_cache, sbn_cache, relu_cache)

    return out, cache


def conv_sbn_relu_backward(dout, cache):
    conv_cache, sbn_cache, relu_cache = cache

    dx = relu_backward(dout, relu_cache)
    dx, dgamma, dbeta = spatial_batchnorm_backward(dx, sbn_cache)
    dx, dw, db = conv_backward_fast(dx, conv_cache)

    return dx, dw, db, dgamma, dbeta


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """

        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim
        pad = (filter_size - 1) / 2
        _H = 1 + (H + 2 * pad - filter_size) / 1
        _W = 1 + (W + 2 * pad - filter_size) / 1
        pool_H = (_H - 2) / 2 + 1
        pool_W = (_W - 2) / 2 + 1

        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
        self.params['W2'] = np.random.randn(num_filters * pool_H * pool_W, hidden_dim) * weight_scale
        self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale

        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        x, conv_cache = conv_relu_pool_forward(X,
                                               self.params['W1'],
                                               self.params['b1'],
                                               conv_param,
                                               pool_param)
        x, fc_cache = affine_relu_forward(x,
                                         self.params['W2'],
                                         self.params['b2'])
        scores, affine_cache = affine_forward(x,
                                              self.params['W3'],
                                              self.params['b3'])
        if y is None:
            return scores

        grads = {}

        loss, dout = softmax_loss(scores, y)
        dout, grads['W3'], grads['b3'] = affine_backward(dout, affine_cache)
        dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, fc_cache)
        _, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, conv_cache)

        for key in grads:
            if key[0] == 'W':
                loss += self.reg * np.sum(np.square(self.params[key]))
                grads[key] += 0.5 * self.reg * self.params[key]

        return loss, grads


class ChaudharyNet(object):
    """
    A custom many-layer convolutional network with the following architecture:

    (convrelu - conv - batchnorm - relu - 2x2 maxpool) x 3 - convrelu - affine - batchnorm - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dim=(3, 32, 32),
                 filter_size=3,
                 num_filters=8,
                 num_classes=10,
                 hidden_dim=200,
                 reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """

        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.filter_size = filter_size

        C, H, W = input_dim
        pad = (filter_size - 1) / 2

        H_1 = 1 + (H + 2 * pad - filter_size) / 1
        W_1 = 1 + (W + 2 * pad - filter_size) / 1
        pool_H_1 = (H_1 - 2) / 2 + 1
        pool_W_1 = (W_1 - 2) / 2 + 1

        H_2 = 1 + (pool_H_1 + 2 * pad - filter_size) / 1
        W_2 = 1 + (pool_W_1 + 2 * pad - filter_size) / 1
        pool_H_2 = (H_2 - 2) / 2 + 1
        pool_W_2 = (W_2 - 2) / 2 + 1

        H_3 = 1 + (pool_H_2 + 2 * pad - filter_size) / 1
        W_3 = 1 + (pool_W_2 + 2 * pad - filter_size) / 1
        pool_H_3 = (H_3 - 2) / 2 + 1
        pool_W_3 = (W_3 - 2) / 2 + 1

        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) / np.sqrt(filter_size * filter_size * C / 2)
        self.params['W2'] = np.random.randn(num_filters * 2, num_filters, filter_size, filter_size) / np.sqrt(filter_size * filter_size * num_filters / 2)
        self.params['W3'] = np.random.randn(num_filters * 4, num_filters * 2, filter_size, filter_size) / np.sqrt(filter_size * filter_size * num_filters)
        self.params['W4'] = np.random.randn(num_filters * 6, num_filters * 4, filter_size, filter_size) / np.sqrt(filter_size * filter_size * num_filters * 2)
        self.params['W5'] = np.random.randn(num_filters * 8, num_filters * 6, filter_size, filter_size) / np.sqrt(filter_size * filter_size * num_filters * 3)
        self.params['W6'] = np.random.randn(num_filters * 16, num_filters * 8, filter_size, filter_size) / np.sqrt(filter_size * filter_size * num_filters * 4)
        self.params['W7'] = np.random.randn(num_filters * 32, num_filters * 16, filter_size, filter_size) / np.sqrt(filter_size * filter_size * num_filters * 8)
        self.params['W8'] = np.random.randn(pool_H_3 * pool_W_3 * num_filters * 32, hidden_dim) / np.sqrt(pool_H_3 * pool_W_3 * num_filters * 16)
        self.params['W9'] = np.random.randn(hidden_dim, num_classes) / np.sqrt(hidden_dim / 2)

        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(num_filters * 2)
        self.params['b3'] = np.zeros(num_filters * 4)
        self.params['b4'] = np.zeros(num_filters * 6)
        self.params['b5'] = np.zeros(num_filters * 8)
        self.params['b6'] = np.zeros(num_filters * 16)
        self.params['b7'] = np.zeros(num_filters * 32)
        self.params['b8'] = np.zeros(hidden_dim)
        self.params['b9'] = np.zeros(num_classes)

        self.params['gamma1'] = np.ones(num_filters * 2)
        self.params['gamma2'] = np.ones(num_filters * 6)
        self.params['gamma3'] = np.ones(num_filters * 16)
        self.params['gamma4'] = np.ones(hidden_dim)

        self.params['beta1'] = np.zeros(num_filters * 2)
        self.params['beta2'] = np.zeros(num_filters * 6)
        self.params['beta3'] = np.zeros(num_filters * 16)
        self.params['beta4'] = np.zeros(hidden_dim)

        self.bn_params = [{
            'mode': 'train'
        } for _ in np.arange(4)]

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the many-layer convolutional network.
        """

        # pass conv_param to the forward pass for the convolutional layer
        conv_param = {'stride': 1, 'pad': (self.filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        mode = 'test' if y is None else 'train'
        for bn_param in self.bn_params:
            bn_param[mode] = mode

        conv_cache = [() for _ in np.arange(7)]

        params = self.params
        bn_params = self.bn_params
        
        # Forward pass
        x, conv_cache[0] = conv_relu_forward(X, params['W1'], params['b1'], conv_param)
        x, conv_cache[1] = conv_sbn_relu_pool_forward(x, params['W2'], params['b2'], params['gamma1'], params['beta1'], conv_param, bn_params[0], pool_param)
        x, conv_cache[2] = conv_relu_forward(x, params['W3'], params['b3'], conv_param)
        x, conv_cache[3] = conv_sbn_relu_pool_forward(x, params['W4'], params['b4'], params['gamma2'], params['beta2'], conv_param, bn_params[1], pool_param)
        x, conv_cache[4] = conv_relu_forward(x, params['W5'], params['b5'], conv_param)
        x, conv_cache[5] = conv_sbn_relu_pool_forward(x, params['W6'], params['b6'], params['gamma3'], params['beta3'], conv_param, bn_params[2], pool_param)
        x, conv_cache[6] = conv_relu_forward(x, self.params['W7'], self.params['b7'], conv_param)
        x, fc_cache = affine_batchnorm_relu_forward(x, self.params['W8'], self.params['b8'], self.params['gamma4'], self.params['beta4'], self.bn_params[3])
        x, affine_cache = affine_forward(x, self.params['W9'], self.params['b9'])

        if y is None:
            return x

        grads = {}

        # Backward pass
        loss, dout = softmax_loss(x, y)
        dout, grads['W9'], grads['b9'] = affine_backward(dout, affine_cache)
        dout, grads['W8'], grads['b8'], grads['gamma4'], grads['beta4'] = affine_batchnorm_relu_backward(dout, fc_cache)
        dout, grads['W7'], grads['b7'] = conv_relu_backward(dout, conv_cache[6])
        dout, grads['W6'], grads['b6'], grads['gamma3'], grads['beta3'] = conv_sbn_relu_pool_backward(dout, conv_cache[5])
        dout, grads['W5'], grads['b5'] = conv_relu_backward(dout, conv_cache[4])
        dout, grads['W4'], grads['b4'], grads['gamma2'], grads['beta2'] = conv_sbn_relu_pool_backward(dout, conv_cache[3])
        dout, grads['W3'], grads['b3'] = conv_relu_backward(dout, conv_cache[2])
        dout, grads['W2'], grads['b2'], grads['gamma1'], grads['beta1'] = conv_sbn_relu_pool_backward(dout, conv_cache[1])
        _, grads['W1'], grads['b1'] = conv_relu_backward(dout, conv_cache[0])

        for key in grads:
            if key[0] == 'W':
                loss += self.reg * np.sum(np.square(self.params[key]))
                grads[key] += 0.5 * self.reg * self.params[key]

        return loss, grads