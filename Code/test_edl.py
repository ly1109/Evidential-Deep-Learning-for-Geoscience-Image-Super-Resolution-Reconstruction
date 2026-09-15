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

model = build_model(lr_shape, scale, num_filters, num_group, num_block, 'evidential')
model.summary()

initial_lr = train_parameters[data_type]["initial_learning_rate"]

optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr,
                                     beta_1=0.9,
                                     beta_2=0.999,
                                     epsilon=1e-07,
                                     amsgrad=False,
                                     global_clipnorm=1.0)


path = f"checkpoints/{data_type}/EDL_GISRR/{degradation_type}/"
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, path + "model", max_to_keep=15)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

y_pred = model.predict(x_test)
gamma, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
gamma, v, alpha, beta = np.squeeze(gamma), np.squeeze(v), np.squeeze(alpha), np.squeeze(beta)
au = beta / (alpha - 1)
eu = beta / (alpha - 1) / v

psnr = []
for i in range(x_test.shape[0]):
    a = peak_signal_noise_ratio(y_test[i, :, :], gamma[i, :, :], data_range=1.0)
    psnr.append(a)

ssim = []
for i in range(x_test.shape[0]):
    a = structural_similarity(y_test[i, :, :], gamma[i, :, :], data_range=1.0)
    ssim.append(a)

gamma = gamma * (xmax - xmin) + xmin
eu = np.sqrt(eu) * (xmax - xmin)
au = np.sqrt(au) * (xmax - xmin)
y_test = y_test * (xmax - xmin) + xmin
x_test = x_test * (xmax - xmin) + xmin

cc = []
for i in range(x_test.shape[0]):
    r2 = np.corrcoef(y_test[i, :, :].flatten(), gamma[i, :, :].flatten())
    if np.isnan(r2).any():
        cc.append(1.0)
    else:
        cc.append(r2[0, 1])

rfne = []
for i in range(x_test.shape[0]):
    a = np.linalg.norm(y_test[i, :, :]-gamma[i, :, :], 'fro') / np.linalg.norm(y_test[i, :, :], 'fro')
    rfne.append(a)

mae = np.abs(y_test - gamma)
mse = np.square(y_test - gamma)

data = {'y_test': y_test, 'y_pred': gamma, 'x_test': x_test, 'eu': eu, 'au': au}
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
