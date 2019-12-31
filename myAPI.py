import tensorflow as tf
import numpy as np


def weight_variable_2D(shape, wInit):
    n1, n2 = shape
    std = np.sqrt(2.0 / (n1 + n2))
    if wInit == 'u':
        initial = tf.random_uniform(shape, -np.sqrt(3)*std, np.sqrt(3)*std)
    elif wInit == 'n':
        initial = tf.random_normal(shape, mean=0.0, stddev=std)

    return tf.Variable(initial)


def bias_variable(shape, bInit):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(float(bInit), shape=shape)

    return tf.Variable(initial)


class MLP():
    def __init__(
            self, inNum, hiddenLayers, outNum,
            actiFunc, outActiFunc,
            wInit, bInit):
        self.actiFunc = actiFunc
        self.outActiFunc = outActiFunc
        self.layers = [inNum] + hiddenLayers + [outNum]
        self.W = []
        self.b = []
        for i in range(len(self.layers)-1):
            self.W.append(weight_variable_2D([self.layers[i], self.layers[i+1]], wInit))
            self.b.append(bias_variable([self.layers[i+1]], bInit))

    def __call__(self, x, keepProb, useRes):
        h = x
        # non-output layers
        for i in range(len(self.layers)-2):
            h = tf.matmul(h, self.W[i]) + self.b[i]
            if i == len(self.layers) - 2 - 1:
                h = tf.cond(useRes,
                            lambda: h + tf.pad(x, [[0, 0], [0, self.layers[-2]-self.layers[0]]]),
                            lambda: h)
            h = self.actiFunc(h)
            h = tf.nn.dropout(h, keepProb)
        # output layer
        i = len(self.layers) - 2
        h = self.outActiFunc(tf.matmul(h, self.W[i]) + self.b[i])
        return h


def setActiFunc(funcName):
    if funcName.startswith('sigm'):
        if funcName == 'sigm':
            return tf.nn.sigmoid
        else:
            try:
                a = float(funcName[4:])
                return lambda x: tf.nn.sigmoid(a*x)
            except:
                raise ValueError('Coef error for sigm: ' + funcName)
    elif funcName == 'identity':
        return tf.identity
    raise ValueError('Unsupported activation function: ' + funcName)
