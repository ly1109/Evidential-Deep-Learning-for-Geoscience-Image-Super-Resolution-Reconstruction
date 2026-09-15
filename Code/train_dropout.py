import random
import time
import numpy as np
import tensorflow as tf

from generate_dataset import build_dataset
from model.rcan import build_model
from loss_functions import dropout_regression
from config import *


random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


data_type = dataset["type_choices"][0]
degradation_type = dataset["degradation_choices"][0]

x_train, y_train, _, _, _, _ = build_dataset(data_type, degradation_type)

lr_shape = train_parameters[data_type]["input_size"]
scale = train_parameters[data_type]["scale"]

num_filters = model_parameters["number_filters"]
num_group = model_parameters["number_groups"]
num_block = model_parameters["number_blocks"]

model = build_model(lr_shape, scale, num_filters, num_group, num_block, 'dropout')
model.summary()

initial_lr = train_parameters[data_type]["initial_learning_rate"]

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4,
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


@tf.function
def metric_function(real, fake):
    fake = fake[:, :, :, 0]
    mae = tf.reduce_mean(tf.abs(tf.cast(real, tf.float32) - fake))
    mse = tf.reduce_mean(tf.square(tf.cast(real, tf.float32) - fake))
    return mae, mse


@tf.function
def train_step(x, y):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x, training=True)
        loss = (dropout_regression(y, y_pred))
        mae, mse = metric_function(y, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, mae, mse


epochs = train_parameters[data_type]["epochs"]
batch_size = train_parameters[data_type]["batch_sizes"]
train_loss = []
for epoch in range(epochs):
    start = time.time()
    train_metric = []
    shuffle_ix = np.random.permutation(np.arange(x_train.shape[0]))
    x_train = x_train[shuffle_ix, :, :]
    y_train = y_train[shuffle_ix, :, :]
    for i in range(int(x_train.shape[0] / batch_size)):
        begin, end = i * batch_size, (i + 1) * batch_size
        x, y = x_train[begin:end, :, :], y_train[begin:end, :, :]
        loss, mae, mse = train_step(x, y)
        train_metric.append([float(loss), float(mae), float(mse)])
        if (i + 1) % 10 == 0:
            print('Step {:03d}, Loss:{:05f}, MAE:{:05f}, MSE:{:06f}'.format(i + 1, *list(train_metric[-1])), end='\n')
    if (x_train.shape[0] % batch_size) != 0:
        i = i + 1
        x, y = x_train[end:, :], y_train[end:, :, :]
        loss, mae, mse = train_step(x, y)
        train_metric.append([float(loss), float(mae), float(mse)])
        if (i + 1) % 10 == 0:
            print('Step {:03d}, Loss:{:05f}, MAE:{:05f}, MSE:{:06f}'.format(i + 1, *list(train_metric[-1])), end='\n')
    train_metric = np.mean(train_metric, axis=0)
    train_loss.append(train_metric)
    if (epoch + 1) % 10 == 0:
        save_path = ckpt_manager.save()
        print(f"Checkpoint Saved: {save_path}")
    if optimizer.lr >= 1e-6:
        optimizer.lr.assign(optimizer.lr * 0.8)

np.savetxt(path + "train_loss.txt", np.array(train_loss))
print("Training Finished!")

