import os

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense, TimeDistributed
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, MaxPooling2D, Reshape, MaxPool2D, Activation, AveragePooling2D
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


def build_baseline_model(input_size, d_model ,learning_rate=3e-4):

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


def rnn_model(input_size, units, activation = 'relu', output_dim=29, learning_rate=3e-4):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(input_size[0], input_size[1]))
    
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, name='rnn')(input_data)
    
    #Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    
    #Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    
    #Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    #Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # model.output_length = lambda x: x
    
    #compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=ctc_loss_lambda_func)
    model.summary()
    
    return model