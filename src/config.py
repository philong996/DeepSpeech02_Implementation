preprocess = {
    'test_size': 0,
    'valid_size': 0.1,
    'max_label_length': 400,
    'max_input_length' : 2560
}

training = {    
    'epochs' : 100,
    'batch_size' : 32
}

model_architecture = {
    'rnn_layers' :  3,
    'units_rnn': 1024,
    'is_bi': True,
}