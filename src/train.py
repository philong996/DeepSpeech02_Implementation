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

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--train-folder", type=str, required=False)
    parser.add_argument("--decode-valid", action="store_true", default=False)
    parser.add_argument("--crnn", action="store_true", default=False)
    parser.add_argument("--ini-epochs", type=int, default = 0)
    parser.add_argument("--epochs", type=int, default = config.training['epochs'])
    parser.add_argument("--model-name", type=str, required=True, help='Name of the model to save')

    args = parser.parse_args()

    # make chackpoints directory, if necessary
    if not os.path.exists('../checkpoints'):
        os.makedirs('../checkpoints')

    checkpoint_path = os.path.join('../checkpoints', args.model_name + '.h5')

    data_detail = utils.get_data_detail(args.train_folder)

    if args.crnn:
        pass
        model = speech_models.deep_speech(input_size = (data_detail['max_input_length'] , data_detail['num_features']), 
                                    units = config.model_architecture['units_rnn'], 
                                    rnn_layers = config.model_architecture['rnn_layers'], 
                                    is_bi = config.model_architecture['is_bi'])
    else:
        model = speech_models.rnn(input_size = (data_detail['max_input_length'] , data_detail['num_features']), 
                                is_bi = config.model_architecture['is_bi'], 
                                units = 200, 
                                layers = 2)

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
    
    
    
    
    if args.train:

        #load weight to continue training
        if os.path.isfile(checkpoint_path):
            model.load_weights(checkpoint_path)
        
        # add callbacks
        batch_stats_callback = speech_models.CollectBatchStats()
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
                verbose=1),
            batch_stats_callback
        ]

        #train model
        history = model.fit(train_ds, 
                            epochs = args.epochs, 
                            validation_data = valid_ds, 
                            validation_steps = VALID_STEPS , 
                            steps_per_epoch = TRAIN_STEPS, 
                            callbacks= callbacks)

        loss = {'loss' : batch_stats_callback.batch_losses, 
        'val_loss' : batch_stats_callback.batch_val_losses}
        
        if not os.path.exists('../results/'):
            os.makedirs('../results/')
        
        #save the result to compare models after training
        pickle_path = os.path.join('../results', args.model_name + '{}.pickle'.format('_' + str(args.ini_epochs) + '_' + str(args.epochs)))
        with open(pickle_path, 'wb') as f:
            pickle.dump(loss, f)

    if args.decode_valid:

        #make predictions
        predictions = model.predict(valid_ds, steps = VALID_STEPS)

        #decode predictions and save to txt file
        predicts = utils.decode_predictions(predictions, data_detail['max_label_length'])
        
        #save result to prediction file
        prediction_file = os.path.join('../results/', 'predictions_{}.txt'.format(args.model_name+ '_for_valid_set_' + str(args.ini_epochs + args.epochs)))
        
        with open(prediction_file, "w") as f:
            for pd, gt in zip(predicts, valid_labels):
                f.write("Y {}\nP {}\n\n".format(gt, pd))

        #calculate metrics to assess the model
        evaluate = utils.calculate_metrics(predicts=predicts,
                                          ground_truth=valid_labels)

        e_corpus = "\n".join([
            "Total test audios:    {}".format(len(valid_labels)),
            "Metrics:",
            "Character Error Rate: {}".format(evaluate[0]),
            "Word Error Rate:      {}".format(evaluate[1]),
            "Sequence Error Rate:  {}".format(evaluate[2]),
        ])
        
        evaluate_file = os.path.join('../results/', "evaluate_{}.txt".format(args.model_name + '_for_valid_set_' + str(args.ini_epochs + args.epochs)))
        with open(evaluate_file, "w") as ev_f:
            ev_f.write(e_corpus)
            print(e_corpus)