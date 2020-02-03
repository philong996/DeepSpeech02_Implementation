from time import time
import argparse
import os

import utils
import config


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--src", 
                        type=str,
                        required=True, 
                        help="The name of folder containing raw files")
    parser.add_argument("--dst",
                        type=str,
                        required=True, 
                        help="The name of folder containing preprocessed files")
    parser.add_argument("--meta",
                        action="store_true",
                        help='Create metadata from raw folders')
    parser.add_argument("--tfrecord",
                        action="store_true",
                        help='Convert audio file to TFrecords')
    parser.add_argument("--test-size",
                        type=str,
                        required=False,
                        default=config.preprocess['test_size'], 
                        help="Fraction of examples in the testing set")
    parser.add_argument("--valid-size",
                        type=str,
                        required=False,
                        default=config.preprocess['valid_size'], 
                        help="Fraction of examples in the validation set")

    args = parser.parse_args()

    ROOT = '../data'
    SRC = os.path.join(ROOT, args.src)
    DST = os.path.join(ROOT, args.dst)
    
    if args.meta:
        print('INFO: Creating metadata ')
        utils.create_main_metadata(SRC, DST) 

    if args.tfrecord:
        print('INFO: Preprocessing Audios, Save to TFrecords')
        converter = utils.TFRecordsConverter(meta_path = os.path.join(DST,'metadata.csv'), 
                    output_dir = os.path.join(DST,'TFrecords'), 
                    test_size = float(args.test_size), 
                    val_size = float(args.valid_size))

        converter.convert()