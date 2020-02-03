import datetime
import os
import codecs
import argparse
import pickle

import speech_models
import config
import utils

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint  
from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



def get_data_detail(folder_name):
    ''' Get the information of data from data_info.txt
    '''
    ROOT = '../data'

    data_file = os.path.join(ROOT, folder_name, 'data_info.txt')
    with codecs.open(data_file , 'r') as f:
        lines = f.readlines()

    data_detail =  {
        'n_training' : int(lines[0].split(':')[-1]),
        'n_valid' : int(lines[1].split(':')[-1]),
        'n_test' : int(lines[2].split(':')[-1]),
        'max_label_length': int(lines[6].split(':')[-1]),
        'max_input_length': int(lines[5].split(':')[-1]),
        'data_folder' : lines[3].split(':')[-1].strip(),
        'num_features' : 161,
        'num_label' : 29
    }
    return data_detail


def decode_predictions(predictions, MAX_LABEL_LENGTH):
    '''Decode a prediction using tf.ctc_decode for the highest probable character at each
        timestep. Then, simply convert the integer sequence to text
    '''
    x_test = np.array(predictions)
    x_test_len = [MAX_LABEL_LENGTH for _ in range(len(x_test))]
    decode, log = K.ctc_decode(x_test,
                            x_test_len,
                            greedy=True,
                            beam_width=10,
                            top_paths=1)

    #probabilities = [np.exp(x) for x in log]
    predicts = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
    predicts = np.swapaxes(predicts, 0, 1)
    
    predicts = [utils.idx_string(label[0]) for label in predicts]

    return predicts



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-folder", type=str, required=True)
    parser.add_argument("--test-folder", default=None , type=str, required=False)
    parser.add_argument("--crnn", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default = config.training['epochs'])
    parser.add_argument("--model-name", type=str, required=True, help='Name of the model to save')

    args = parser.parse_args()

    # make chackpoints directory, if necessary
    if not os.path.exists('../checkpoints'):
        os.makedirs('../checkpoints')

    checkpoint_path = os.path.join('../checkpoints', args.model_name + '.h5')

    data_detail = get_data_detail(args.train_folder)

    if args.crnn:
        pass
        model = speech_models.deep_speech(input_size = (data_detail['max_input_length'] , data_detail['num_features']), 
                                    units = config.model_architecture['units_rnn'], 
                                    rnn_layers = config.model_architecture['rnn_layers'], 
                                    is_bi = config.model_architecture['is_bi'])
    else:
        model = speech_models.rnn(input_size = (data_detail['max_input_length'] , data_detail['num_features']), 
                                        is_bi = config.model_architecture['is_bi'], 
                                        units = config.model_architecture['units_rnn'], 
                                        layers = config.model_architecture['rnn_layers'])

    if args.train_folder:

        #prepare for training data
        TRAIN_STEPS = int(data_detail['n_training'] / config.training['batch_size'])
        VALID_STEPS = int(data_detail['n_valid'] / config.training['batch_size'])
        
        train_ds, train_labels = utils.get_dataset_from_tfrecords(data_detail,
                                                                 tfrecords_dir=data_detail['data_folder'], 
                                                                 split='train', 
                                                                 batch_size=config.training['batch_size'])
        valid_ds, valid_labels = utils.get_dataset_from_tfrecords(data_detail,
                                                                tfrecords_dir=data_detail['data_folder'], 
                                                                split='valid', 
                                                                batch_size=config.training['batch_size'])

        #load weight to continue training
        if os.path.isfile(checkpoint_path):
            model.load_weights(checkpoint_path)
        
        # add callbacks
        # batch_stats_callback = speech_models.CollectBatchStats()
        callbacks = [
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1),
            EarlyStopping(
                monitor='val_loss',
                min_delta=1e-8,
                patience=15,
                restore_best_weights=True,
                verbose=1),
            ReduceLROnPlateau(
                monitor='val_loss',
                min_delta=1e-8,
                factor=0.2,
                patience=10,
                verbose=1)
        ]

        #train model

        history = model.fit(train_ds, 
                            epochs = args.epochs, 
                            validation_data = valid_ds, 
                            validation_steps = VALID_STEPS , 
                            steps_per_epoch = TRAIN_STEPS, 
                            callbacks= callbacks)

        # loss = {'loss' : batch_stats_callback.batch_losses, 
        # 'val_loss' : batch_stats_callback.batch_val_losses}

        #save the result to compare models after training
        pickle_path = os.path.join('../checkpoints', args.model_name + '.pickle')
        with open(pickle_path, 'wb') as f:
            pickle.dump(history.history, f)

    if args.test_folder:
        
        #get test data detail
        test_detail = get_data_detail(args.test_folder)

        #prepare for testing data
        TEST_STEPS = int(test_detail['n_test'] / config.training['batch_size'])
        
        test_ds, labels = utils.get_dataset_from_tfrecords(test_detail, 
                                                        tfrecords_dir=test_detail['data_folder'], 
                                                        split='test', 
                                                        batch_size=config.training['batch_size'])

        #load model weights
        checkpoint_path = os.path.join('../checkpoints', args.model_name + '.h5')
        try:
            model.load_weights(checkpoint_path)
        except:
            print('There is no checkpoint file in the folder')
        
        start_time = datetime.datetime.now()

        #make predictions
        predictions = model.predict(test_ds, steps = TEST_STEPS)

        total_time = datetime.datetime.now() - start_time

        #decode predictions and save to txt file
        predicts = decode_predictions(predictions, test_detail['max_label_length'])
        
        if not os.path.exists('../results/'):
            os.makedirs('../results/')
        
        #save result to prediction file
        prediction_file = os.path.join('../results/', 'predictions_{}.txt'.format(args.model_name))
        with open(prediction_file, "w") as f:
            for pd, gt in zip(predicts, labels):
                f.write("Y {}\nP {}\n\n".format(gt, pd))

        #calculate metrics to assess the model
        evaluate = utils.calculate_metrics(predicts=predicts,
                                          ground_truth=labels)

        e_corpus = "\n".join([
            "Total test audios:    {}".format(len(labels)),
            "Total time:           {}\n".format(total_time),
            "Metrics:",
            "Character Error Rate: {}".format(evaluate[0]),
            "Word Error Rate:      {}".format(evaluate[1]),
            "Sequence Error Rate:  {}".format(evaluate[2]),
        ])
        
        evaluate_file = os.path.join('../results/', "evaluate_{}.txt".format(args.model_name))
        with open(evaluate_file, "w") as ev_f:
            ev_f.write(e_corpus)
            print(e_corpus)
