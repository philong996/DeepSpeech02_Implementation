from deepspeech import Model
import os
import wave
import numpy as np
import json


def metadata_to_string(metadata):
    return ''.join(item.character for item in metadata.items)


def metadata_json_output(metadata):
    json_result = dict()
    json_result["words"] = words_from_metadata(metadata)
    json_result["confidence"] = metadata.confidence
    return json.dumps(json_result)


def words_from_metadata(metadata):
    word = ""
    word_list = []
    word_start_time = 0
    # Loop through each character
    for i in range(0, metadata.num_items):
        item = metadata.items[i]
        # Append character to word if it's not a space
        if item.character != " ":
            word = word + item.character
        # Word boundary is either a space or the last character in the array
        if item.character == " " or i == metadata.num_items - 1:
            word_duration = item.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            each_word = dict()
            each_word["word"] = word
            each_word["start_time "] = round(word_start_time, 4)
            each_word["duration"] = round(word_duration, 4)

            word_list.append(each_word)
            # Reset
            word = ""
            word_start_time = 0
        else:
            if len(word) == 1:
                # Log the start time of the new word
                word_start_time = item.start_time

    return word_list

def load_model(model_dir):
    BEAM_WIDTH = 500
    DEFAULT_SAMPLE_RATE = 16000
    LM_ALPHA = 0.75
    LM_BETA = 1.85
    
    model_path = os.path.join(model_dir, 'output_graph.pbmm')
    trie_path = os.path.join(model_dir, 'trie')
    lm_path = os.path.join(model_dir, 'lm.binary')

    model = Model(model_path, BEAM_WIDTH)
    model.enableDecoderWithLM(lm_path , trie_path, LM_ALPHA, LM_BETA)
    
    return model

def transcribe(audio_path, model):
    fin = wave.open(audio_path, 'rb')
    fs = fin.getframerate()

    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
    
    fin.close()
    
    result = metadata_to_string(model.sttWithMetadata(audio))
    
    return result


if __name__ == "__main__":
    model_dir = './deepspeech-0.6.1-models'

    model = load_model(model_dir)

    print(transcribe('./test_mic/chunk-03.wav', model))