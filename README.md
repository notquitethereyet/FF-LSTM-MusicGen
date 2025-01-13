# LSTM Music Generator 🎵

This project uses an LSTM neural network to generate music in MIDI format. Trained on a dataset of MIDI files, the model predicts sequences of notes and chords, creating new musical compositions.

## Features

- Train an LSTM model with PyTorch to generate MIDI files
- Support for data augmentation and temperature sampling for diverse outputs
- Lightweight and modular codebase for customization

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
git clone https://github.com/notquitethereyet/FF-LSTM-MusicGen.git
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
The generated music is saved as `output_music.mid`.

## Preprocessing Steps

### Extract Notes and Chords
- MIDI files are parsed using music21 to extract individual notes and chords
- Notes are stored as strings (e.g., "C4"), and chords are represented as dot-separated strings of note integers (e.g., "60.64.67")

### Create Input-Output Sequences
- A sliding window of sequences is created with a fixed length (e.g., 100 notes)
- Each sequence serves as input, and the following note/chord serves as output

### Normalize Data
- Input sequences are mapped to integers and normalized to [0, 1]
- Normalization is done by dividing by the total number of unique notes/chords

### Save Processed Data
- Processed sequences and mappings (note_to_int and int_to_note) are saved for training and generation

## Model Overview

### Architecture

The LSTM model uses PyTorch with:

- **LSTM Layers**: Three layers with 512 hidden units and recurrent dropout
- **Batch Normalization**: For faster convergence and stability
- **Dense Layers**: 256 units with ReLU activation
- **Dropout Layers**: 30% dropout rate
- **Softmax Output Layer**: Probability distribution over possible notes/chords

### Input and Output
- Input: Sequence of 100 notes/chords (normalized integers)
- Output: Next note/chord prediction (one-hot encoded)

## Dataset

### Requirements
- Single-instrument MIDI files (e.g., piano)
- Recommended datasets:
    - MAESTRO
    - MIDIWorld

### Data Augmentation

Example code for transposition:
```python
from music21 import converter

midi = converter.parse("example.mid")
for semitone in range(-5, 6):  # Transpose to nearby keys
        transposed = midi.transpose(semitone)
        transposed.write("midi", fp=f"example_transposed_{semitone}.mid")
```

## Troubleshooting

### Repetitive Output
- Use temperature sampling in generate.py
- Add more training data
- Reduce the learning rate

### Memory Errors
- Decrease the batch size in train.py

## License

This project is licensed under the MIT License.
