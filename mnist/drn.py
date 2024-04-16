import numpy as np
import tensorflow as tf


def init_d(q_upper, q_lower):
    lower = np.arange(q_lower, dtype=np.float32)
    upper = np.arange(q_upper, dtype=np.float32)
    D = lower.reshape(1, q_lower)/q_lower - upper.reshape(q_upper, 1)/q_upper
    D = np.exp(-np.power(D, 2))
    return D


def cal_logexp_bias(B, s0, q):
    b_a, b_q, lamb_a, lamb_q = B
    # b_a : absolute bias
    # b_q : quadratic bias
    B = -b_q * tf.pow(s0/q - lamb_q, 2) - b_a * tf.abs(s0 / q - lamb_a)
    return B


class DRNLayer(tf.keras.layers.Layer):
    def __init__(self, n_lower, n_upper, q_lower, q_upper, loadparam=False, loadW=None, loadB=None, fixWb=False, w_init=0.1, w_init_method='glorot_uniform'):
        super().__init__()
        self.n_lower, self.n_upper, self.q_lower, self.q_upper = n_lower, n_upper, q_lower, q_upper
        self.D = tf.tile(tf.constant(init_d(q_lower=self.q_lower, q_upper=self.q_upper).reshape([self.q_upper, self.q_lower, 1, 1])), [1, 1, self.n_upper, self.n_lower])
        self.s0 = tf.constant(np.arange(self.q_upper, dtype=np.float32).reshape((1, self.q_upper)))

        # self.register_buffer('D' ,torch.tile(torch.from_numpy(init_D(q_upper, q_lower)).reshape(
        #     [q_upper, q_lower, 1, 1]), [1, 1, n_upper, n_lower]))
        # self.register_buffer('s0', torch.from_numpy(np.arange(self.q_upper, dtype=np.float32).reshape((1, self.q_upper))))
        if w_init_method=="uniform":
            self.initializer = tf.keras.initializers.RandomUniform(minval=-w_init, maxval=w_init, seed=0)
        elif w_init_method=="glorot_uniform":
            self.initializer = tf.keras.initializers.GlorotUniform(seed=0)
        elif w_init_method=="glorot_normal":
            self.initializer = tf.keras.initializers.GlorotNormal(seed=0)


    def build(self, input_shapes):
        self.weight = self.add_weight(
            name="weight",
            shape=[self.n_upper, self.n_lower],
            initializer=self.initializer
            )
        self.bias_abs = self.add_weight(
            name="bias_abs",
            shape=[self.n_upper, 1],
            initializer=self.initializer
            )
        self.bias_q = self.add_weight(
            name="bias_q",
            shape=[self.n_upper, 1],
            initializer=self.initializer
            )
        self.lambda_abs = self.add_weight(
            name="lambda_abs",
            shape=[self.n_upper, 1],
            initializer=self.initializer
            )
        self.lambda_q = self.add_weight(
            name="lambda_q",
            shape=[self.n_upper, 1],
            initializer=self.initializer
            )

    def call(self, P):
        P_tile = tf.tile(
            tf.reshape(P, [-1, 1, self.n_lower, self.q_lower, 1]),
            [1, self.n_upper, 1, 1, 1]
        )
        T = tf.transpose(
            tf.pow(
                self.D,
                self.weight,
            ), [2, 3, 0, 1]
        )
        Pw = tf.squeeze(
            tf.einsum(
                'jklm,ijkmn->ijkln', T, P_tile
            ),
            axis=4
        )

        Pw = tf.clip_by_value(Pw, 1e-15, 1e+15)
        logPw = tf.math.log(Pw)
        logsum = tf.math.reduce_sum(logPw, axis=[2])
        exponent_B = cal_logexp_bias(
            [self.bias_abs, self.bias_q, self.lambda_abs, self.lambda_q],
            self.s0,
            self.q_upper
        )
        logsumB = tf.add(logsum, exponent_B)
        
        max_logsum = tf.reduce_max(logsumB, axis=2, keepdims=True)
        expm_P = tf.exp(tf.subtract(logsumB, max_logsum))
        Z = tf.reduce_sum(expm_P, 2, keepdims=True)
        ynorm = tf.divide(expm_P, Z)
        return ynorm
