import random

import numpy as np
import scipy
import tensorflow as tf
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from config import *
from generate_dataset import build_dataset
from model.rcan import build_model


random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


data_type = dataset["type_choices"][0]
degradation_type = dataset["degradation_choices"][0]

_, _, x_test, y_test, xmax, xmin = build_dataset(data_type, degradation_type)

lr_shape = train_parameters[data_type]["input_size"]
scale = train_parameters[data_type]["scale"]

num_filters = model_parameters["number_filters"]
num_group = model_parameters["number_groups"]
num_block = model_parameters["number_blocks"]

model = build_model(lr_shape, scale, num_filters, num_group, num_block, 'dropout')
model.summary()

initial_lr = train_parameters[data_type]["initial_learning_rate"]

optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr,
                                     beta_1=0.9,
                                     beta_2=0.999,
                                     epsilon=1e-07,
                                     amsgrad=False,
                                     global_clipnorm=1.0)


path = f"checkpoints/{data_type}/MCDN_GISRR/{degradation_type}/"
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, path + "model", max_to_keep=15)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

batch_size = 50
y_pred, au, eu = [], [], []
for i in range(int(x_test.shape[0] / batch_size)):
    begin, end = i * batch_size, (i + 1) * batch_size
    x = x_test[begin:end, :, :]
    y1, sigma = [], []
    for k in range(50):
        y = model(x, training=True)
        mu, u = tf.split(y, 2, axis=-1)
        y1.append(mu)
        sigma.append(u)
    y1 = np.array(y1)
    y1 = np.squeeze(y1, 4)
    sigma = np.array(sigma)
    sigma = np.squeeze(sigma, 4)
    y_mean = np.mean(y1, axis=0)
    y_std = np.var(y1, axis=0)
    sigma_mean = np.mean(sigma, axis=0)
    y_pred.append(y_mean)
    au.append(sigma_mean)
    eu.append(y_std)

if (x_test.shape[0] % batch_size) != 0:
    x = x_test[end:, :]
    y1, sigma = [], []
    for k in range(50):
        y = model(x, training=True)
        mu, u = tf.split(y, 2, axis=-1)
        y1.append(mu)
        sigma.append(u)
    y1 = np.array(y1)
    y1 = np.squeeze(y1, 4)
    sigma = np.array(sigma)
    sigma = np.squeeze(sigma, 4)
    y_mean = np.mean(y1, axis=0)
    y_std = np.var(y1, axis=0)
    sigma_mean = np.mean(sigma, axis=0)
    y_pred.append(y_mean)
    au.append(sigma_mean)
    eu.append(y_std)

y_pred = np.concatenate(y_pred)
au = np.concatenate(au)
eu = np.concatenate(eu)

psnr = []
for i in range(x_test.shape[0]):
    a = peak_signal_noise_ratio(y_test[i, :, :], y_pred[i, :, :], data_range=1.0)
    psnr.append(a)

ssim = []
for i in range(x_test.shape[0]):
    a = structural_similarity(y_test[i, :, :], y_pred[i, :, :], data_range=1.0)
    ssim.append(a)

y_pred = y_pred * (xmax - xmin) + xmin
eu = np.sqrt(eu) * (xmax - xmin)
au = np.sqrt(np.exp(au)) * (xmax - xmin)
y_test = y_test * (xmax - xmin) + xmin
x_test = x_test * (xmax - xmin) + xmin

mae = np.abs(y_test - y_pred)
mse = np.square(y_test - y_pred)

cc = []
for i in range(x_test.shape[0]):
    r2 = np.corrcoef(y_test[i, :, :].flatten(), y_pred[i, :, :].flatten())
    # a = np.corrcoef(ones_like(y_test[i, :, :].flatten()), gamma[i, :, :].flatten())
    if np.isnan(r2).any():
        cc.append(1.0)
    else:
        cc.append(r2[0, 1])

rfne = []
for i in range(x_test.shape[0]):
    a = np.linalg.norm(y_test[i, :, :]-y_pred[i, :, :], 'fro') / np.linalg.norm(y_test[i, :, :], 'fro')
    rfne.append(a)

data = {'y_test': y_test, 'y_pred': y_pred, 'x_test': x_test, 'eu': eu, 'au': au}
scipy.io.savemat(path + 'data.mat', data)

mae = np.mean(mae, axis=(1, 2))
mse = np.sqrt(np.mean(mse, axis=(1, 2)))
au = np.mean(au, axis=(1, 2))
eu = np.mean(eu, axis=(1, 2))

np.savetxt(path + 'au.txt', au)
np.savetxt(path + 'eu.txt', eu)
np.savetxt(path + 'cc.txt', cc)
np.savetxt(path + 'mae.txt', mae)
np.savetxt(path + 'mse.txt', mse)
np.savetxt(path + 'psnr.txt', psnr)
np.savetxt(path + 'ssim.txt', ssim)
np.savetxt(path + 'rfne.txt', rfne)


