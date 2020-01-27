import codecs
with codecs.open('../data/toy_final/data_info.txt', 'r') as f:
    lines = f.readlines()

preprocess = {
    'n_train': 5,
    'n_test': 2,
    'n_valid': 2,
    'test_size': 0.1,
    'valid_size': 0.1,
}

data_detail =  {
    'n_training' : int(lines[0].split(':')[-1]),
    'n_valid' : int(lines[1].split(':')[-1]),
    'n_test' : int(lines[2].split(':')[-1]),
    'max_label_length': int(lines[6].split(':')[-1]),
    'max_input_length': int(lines[5].split(':')[-1]),
    'num_features': 40,
    'data_folder' : lines[3].split(':')[-1].strip()
}

training = {    
    'epochs' : 2,
    'batch_size' : 32}