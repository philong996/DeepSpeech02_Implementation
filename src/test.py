import argparse
import datetime
import os
import codecs

import speech_models
import config
import utils

import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-folder", default=None , type=str, required=False)
    parser.add_argument("--crnn", action="store_true", default=False)
    parser.add_argument("--model-name", type=str, required=True, help='Name of the model to save')
    parser.add_argument("--ini-epochs", type=int, required=True)

    args = parser.parse_args()

    if not os.path.exists('../checkpoints'):
        os.makedirs('../checkpoints')

    checkpoint_path = os.path.join('../checkpoints', args.model_name + '.h5')

    #get test data detail
    data_detail = utils.get_data_detail(args.test_folder)

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

    #prepare for testing data
    TEST_STEPS = int(data_detail['n_test'] / config.training['batch_size'])

    test_ds, labels = utils.get_dataset_from_tfrecords(data_detail, 
                                                    tfrecords_dir=data_detail['data_folder'], 
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

    #decode predictions and save to txt file
    predicts = utils.decode_predictions(predictions, data_detail['max_label_length'])

    total_time = datetime.datetime.now() - start_time

    if not os.path.exists('../results/'):
        os.makedirs('../results/')

    #save result to prediction file
    prediction_file = os.path.join('../results/', 'predictions_{}.txt'.format(args.model_name + '_for_test_set_' + str(args.ini_epochs)))
    
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

    evaluate_file = os.path.join('../results/', "evaluate_{}.txt".format(args.model_name + '_for_test_set_' + str(args.ini_epochs)))
    with open(evaluate_file, "w") as ev_f:
        ev_f.write(e_corpus)
        print(e_corpus)



