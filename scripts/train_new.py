import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.fno_3d import FNO3d
from src.physics.loss import CryoPhysicsLoss
from src.utils.data_loader import CryoDataset

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}...")
    
    # 1. Load Data
    data_path = os.path.join("data", "processed")
    if not os.path.exists(data_path) or not os.listdir(data_path):
        print("Error: Data folder empty. Please run 'python scripts/generate_data.py' first.")
        return

    dataset = CryoDataset(data_path)
    # Batch size 8 is good for 32^3 grids on most GPUs/CPUs
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 2. Setup Model
    model = FNO3d(modes=8, width=20, in_channels=3, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # Loss: Weight the physics loss slightly less in the beginning
    loss_fn = CryoPhysicsLoss(saturation_temp=77.0)
    
    # 3. Training Loop
    epochs = 20
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Input format is (Batch, Channels, X, Y, Z)
            # FNO expects (Batch, X, Y, Z, Channels) usually, let's permute
            inputs = inputs.permute(0, 2, 3, 4, 1)
            targets = targets.permute(0, 2, 3, 4, 1)
            
            optimizer.zero_grad()
            pred = model(inputs)
            
            # Calculate Loss
            loss = loss_fn.calculate(pred, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    # 4. Save
    torch.save(model.state_dict(), os.path.join(save_dir, "model_final.pth"))
    print("Training Complete. Model Saved.")

if __name__ == "__main__":
    main()