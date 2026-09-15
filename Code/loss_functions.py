import tensorflow as tf
import numpy as np


def mse(y, y_, reduce=True):
    ax = list(range(1, len(y.shape)))

    mse = tf.reduce_mean((y - y_) ** 2, axis=ax)
    return tf.reduce_mean(mse) if reduce else mse


def gaussian_nll_logvar(y, mu, logvar, reduce=True):
    ax = list(range(1, len(y.shape)))

    log_liklihood = 0.5 * (
            -tf.exp(-logvar) * (mu - y) ** 2 - tf.math.log(2 * tf.constant(np.pi, dtype=logvar.dtype)) - logvar
    )
    loss = tf.reduce_mean(-log_liklihood, axis=ax)
    return tf.reduce_mean(loss) if reduce else loss


def nig_nll(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2 * beta * (1 + v)

    nll = 0.5 * tf.math.log(np.pi / v) \
          - alpha * tf.math.log(twoBlambda) \
          + (alpha + 0.5) * tf.math.log(v * (y - gamma) ** 2 + twoBlambda) \
          + tf.math.lgamma(alpha) \
          - tf.math.lgamma(alpha + 0.5)

    return tf.reduce_mean(nll) if reduce else nll


def kl_nig(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5 * (a1 - 1) / b1 * (v2 * tf.square(mu2 - mu1)) \
         + 0.5 * v2 / v1 \
         - 0.5 * tf.math.log(tf.abs(v2) / tf.abs(v1)) \
         - 0.5 + a2 * tf.math.log(b1 / b2) \
         - (tf.math.lgamma(a1) - tf.math.lgamma(a2)) \
         + (a1 - a2) * tf.math.digamma(a1) \
         - (b1 - b2) * a1 / b1
    return KL


def nig_reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    error = tf.abs(y - gamma)

    if kl:
        kl = kl_nig(gamma, v, alpha, beta, gamma, omega, 1 + omega, beta)
        reg = error * kl
    else:
        evi = 2 * v + (alpha)
        reg = error * evi

    return tf.reduce_mean(reg) if reduce else reg


def evidential_regression(y_true, evidential_output, lam, epsilon=1e-2):
    gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=-1)
    gamma, v, alpha, beta = tf.squeeze(gamma), tf.squeeze(v), tf.squeeze(alpha), tf.squeeze(beta)
    loss_nll = nig_nll(tf.cast(y_true, tf.float32), gamma, v, alpha, beta)
    loss_reg = nig_reg(tf.cast(y_true, tf.float32), gamma, v, alpha, beta)
    loss = loss_nll + lam * (loss_reg - epsilon)
    return loss, loss_nll, loss_reg


def dropout_regression(y_true, evidential_output):
    mu, logvar = tf.split(evidential_output, 2, axis=-1)
    mu, logvar = tf.squeeze(mu), tf.squeeze(logvar)
    loss = gaussian_nll_logvar(tf.cast(y_true, tf.float32), mu, logvar)
    return loss


def regression(y_true, y_pred):
    y_pred = tf.squeeze(y_pred)
    loss = mse(tf.cast(y_true, tf.float32), y_pred)
    return loss