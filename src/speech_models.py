import os

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, MaxPooling2D, Reshape, MaxPool2D, Lambda, AveragePooling2D
from tensorflow.keras.optimizers import Adam


def ctc_loss_lambda_func(y_true, y_pred):
    """Function for computing the CTC loss"""

    if len(y_true.shape) > 2:
        y_true = tf.squeeze(y_true)

    input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
    input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)
    label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")

    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    loss = tf.reduce_mean(loss)

    return loss


def build_baseline_model(input_size, d_model, learning_rate=3e-4):

    input_data = Input(name="input", shape=input_size)

    conv_1 = Conv2D(32, (3,3), activation = 'relu', padding='same')(input_data)
    pool_1 = MaxPool2D(pool_size=(3, 2), strides=2)(conv_1)
    
    conv_2 = Conv2D(64, (3,3), activation = 'relu', padding='same')(pool_1)
    batch_norm_2 = BatchNormalization()(conv_2)
    
    conv_3 = Conv2D(64, (3,3), activation = 'relu', padding='same')(batch_norm_2)
    batch_norm_3 = BatchNormalization()(conv_3)
    pool_3 = MaxPool2D(pool_size=(1, 2))(batch_norm_3)
    
    shape = pool_3.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(pool_3)
    
    blstm = Bidirectional(LSTM(64, return_sequences=True, dropout = 0.5))(blstm)
    blstm = Dropout(rate=0.5)(blstm)
    output_data = Dense(d_model, activation = 'softmax')(blstm)

    #optimizer = RMSprop(learning_rate=learning_rate)
    optimizer = Adam(learning_rate=learning_rate)
    
    model = Model(inputs=input_data, outputs=output_data)
    model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)
    model.summary()
    return model