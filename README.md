# Implemetation of Deepspeech 2 from Baidu Research

### Dataset

- [LibriSpeech ASR corpus.](http://www.openslr.org/12/)

    - LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. 
    - The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned.
    - [LibriSpeech](http://www.openslr.org/11/) language models, vocabulary and G2P models (Language modeling resources to be used in conjunction with the  LibriSpeech ASR corpus.)
### Architecture of Model

- This is an implementation of the [DeepSpeech2](https://arxiv.org/pdf/1512.02595.pdf) model. Current implementation is based on the code from the authors' [DeepSpeech code](https://github.com/PaddlePaddle/DeepSpeech) and [Tensorflow Research](https://github.com/tensorflow/models/tree/master/research/deep_speech) and [Noah Chalifour](https://github.com/noahchalifour/baidu-deepspeech2)

- DeepSpeech2 is an end-to-end deep neural network for automatic speech recognition (ASR). It consists of 2 convolutional layers, 5 bidirectional RNN layers and a fully connected layer. The feature in use is linear spectrogram extracted from audio input. The network uses Connectionist Temporal Classification [CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf) as the loss function.

## Running code

- Prepare data
    - Download data from [OpenSLP](http://www.openslr.org/12/)
    - Unzip the data following the structure to run code
```
├── RAW
│   ├── dev-clean
│   │   ├── 1272
│   │   │   ├── 128104
│   │   │   │   ├── 1272-128104-0000.flac
│   │   │   │   ├── 1272-128104-0001.flac
│   │   │   │   ├── 1272-128104-0002.flac
│   │   │   │   ├── 1272-128104-0003.flac
│   │   │   │   ├── .
│   │   │   │   ├── .
│   │   │   │   ├── .
│   │   │   ├── 135031
│   │   │   │   ├── 1272-135031-0000.flac
│   │   │   │   ├── 1272-135031-0001.flac
│   │   │   │   ├── .
│   │   │   │   ├── .
│   │   │   │   ├── .
│   │   ├── .
│   │   ├── .
```


- Preprocess
    - Run the following script to preprocess the data (This might take a while depending on the amount of data you have)

```bash
python preprocess.py --src dev-clean --dst dev-clean-final
```

- Train
    - Now that you have preprocessed your data, you can train a model. To do this, you can edit the settings in the [config.py](./src/config.py) file if you want. Then run the following command to train the model

```bash
python train.py --train --name dev-clean-final
```