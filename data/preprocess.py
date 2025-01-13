import glob
from music21 import converter, instrument, note, chord
import pickle
import numpy as np
import os

def preprocess_midi(input_dir="data/midi_songs", sequence_length=100, save_path="data/processed_data.pkl"):
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    notes = []
    for file in glob.glob(f"{input_dir}/*.mid"):
        try:
            midi = converter.parse(file)
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if not notes:
        print("No notes were extracted. Check the MIDI files in the input directory.")
        return

    # Map notes to integers
    pitchnames = sorted(set(notes))
    note_to_int = {note: num for num, note in enumerate(pitchnames)}

    # Create input-output sequences
    network_input, network_output = [], []
    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[note] for note in sequence_in])
        network_output.append(note_to_int[sequence_out])

    # Normalize inputs
    network_input = np.array(network_input) / float(len(note_to_int))

    # Save data
    data = {
        "input": network_input,
        "output": network_output,
        "note_to_int": note_to_int,
        "int_to_note": {v: k for k, v in note_to_int.items()}
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Preprocessed data saved to {save_path}")

if __name__ == "__main__":
    preprocess_midi()
