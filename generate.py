import torch
from music21 import instrument, note, chord, stream
from model.lstm_model import MusicLSTM
import pickle
import numpy as np
import torch.nn.functional as F

def sample_with_temperature(prediction, temperature=1.0):
    """Sample an index from a probability distribution with temperature scaling."""
    prediction = F.softmax(prediction.squeeze() / temperature, dim=0).cpu().detach().numpy()
    return np.random.choice(len(prediction), p=prediction)

# Load data
with open("data/processed_data.pkl", "rb") as f:
    data = pickle.load(f)

# Load model
model = MusicLSTM(1, 256, len(data["note_to_int"]))
model.load_state_dict(torch.load("best_model.pth", weights_only=True))  # Enable weights_only for safety
model.eval()

# Generate music
start_idx = np.random.randint(0, len(data["input"]) - 1)
pattern = data["input"][start_idx]
int_to_note = data["int_to_note"]
output_notes = []

for _ in range(500):
    input_seq = torch.tensor(np.array([pattern]), dtype=torch.float32).unsqueeze(-1)
    prediction = model(input_seq)
    predicted_idx = sample_with_temperature(prediction, temperature=0.8)
    output_notes.append(int_to_note[predicted_idx])
    
    # Update pattern
    pattern = list(pattern)  # Convert NumPy array to a list
    pattern.append(predicted_idx)  # Append the predicted index
    pattern = pattern[1:]  # Remove the first element to maintain sequence length
    pattern = np.array(pattern)  # Convert back to NumPy array

# Convert to MIDI
offset = 0
notes = []
for item in output_notes:
    if '.' in item or item.isdigit():
        notes_in_chord = item.split('.')
        chord_notes = [note.Note(int(n)) for n in notes_in_chord]
        new_chord = chord.Chord(chord_notes)
        new_chord.offset = offset
        notes.append(new_chord)
    else:
        new_note = note.Note(item)
        new_note.offset = offset
        notes.append(new_note)
    offset += 0.5

midi_stream = stream.Stream(notes)
midi_stream.write('midi', fp='output_music.mid')
print("Generated music saved to 'output_music.mid'.")
