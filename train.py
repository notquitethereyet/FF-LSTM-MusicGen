import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle
from model.lstm_model import MusicLSTM
from torch.optim.lr_scheduler import StepLR

class MusicDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float32), torch.tensor(self.outputs[idx], dtype=torch.long)

# Load data
with open("data/processed_data.pkl", "rb") as f:
    data = pickle.load(f)

# Split data into training and validation sets
split_idx = int(len(data["input"]) * 0.8)
train_dataset = MusicDataset(data["input"][:split_idx], data["output"][:split_idx])
val_dataset = MusicDataset(data["input"][split_idx:], data["output"][split_idx:])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
input_dim = 1
hidden_dim = 256
output_dim = len(data["note_to_int"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MusicLSTM(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
epochs = 200
best_loss = float("inf")
with tqdm(total=epochs, desc="Training Progress") as pbar:
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            batch_x = batch_x.unsqueeze(-1).to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc="Validation", leave=False):
                batch_x = batch_x.unsqueeze(-1).to(device)
                batch_y = batch_y.to(device)
                output = model(batch_x)
                val_loss += criterion(output, batch_y).item()
        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
        scheduler.step()
        
        # Update progress bar
        pbar.set_postfix({'train_loss': f'{train_loss:.4f}', 'val_loss': f'{val_loss:.4f}'})
        pbar.update(1)

print("Training complete. Best model saved as 'best_model.pth'.")
