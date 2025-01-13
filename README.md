# LSTM Music Generator 🎵

This project uses an LSTM neural network to generate music in MIDI format. Trained on a dataset of MIDI files, the model predicts sequences of notes and chords, creating new musical compositions.

## Features

* Train an LSTM model with PyTorch to generate MIDI files
* Support for data augmentation and temperature sampling for diverse outputs
* Lightweight and modular codebase

## Directory Structure

```
music_lstm/
├── data/
│   ├── midi_songs/           # Place your MIDI files here
│   ├── preprocess.py         # Preprocess MIDI files
├── model/
│   ├── lstm_model.py         # PyTorch LSTM model
├── train.py                  # Train the model
├── generate.py               # Generate MIDI music
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/notquitethereyet/music-lstm.git
cd music-lstm
```

2. Set up a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

3. Place MIDI files in `data/midi_songs/`

### Usage

1. Preprocess Data:
```bash
python data/preprocess.py
```

2. Train the Model:
```bash
python train.py
```

3. Generate Music:
```bash
python generate.py
```

Generated music is saved as `output_music.mid`.

~~Listen to the latest generated sample:~~


## Model Overview

* Architecture: 3 LSTM layers (512 units), batch normalization, dropout, dense layers, softmax output
* Input: Sequences of notes/chords extracted from MIDI files
* Output: Predicted next notes or chords

## Dataset

Use single-instrument MIDI files. Public datasets:

* [MAESTRO](https://magenta.tensorflow.org/datasets/maestro)
* [MIDIWorld](http://www.midiworld.com/)

Augment your dataset by transposing MIDI files to multiple keys for better generalization.

## Troubleshooting

* Repetitive output:
        * Use temperature sampling (generate.py)
        * Add more training data
        * Reduce learning rate

* Memory errors:
        * Decrease batch size in train.py

## License

This project is licensed under the MIT License.