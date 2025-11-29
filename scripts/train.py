import sys
import os
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.fno_3d import FNO3d
from src.physics.loss import CryoPhysicsLoss

# --- Dummy Data Generator ---
def generate_synthetic_batch(batch_size=4, grid=32):
    "Creates random blobs to simulate tanks for demonstration."
    
    # Inputs: [Batch, Grid, Grid, Grid, 3]
    inputs = torch.randn(batch_size, grid, grid, grid, 3)
    
    # Targets: [Batch, Grid, Grid, Grid, 2] (Alpha, Temp)
    # Synthetic ground truth
    targets = torch.randn(batch_size, grid, grid, grid, 2)
    return inputs, targets

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}...")
    
    # Setup
    model = FNO3d(modes=8, width=20, in_channels=3, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = CryoPhysicsLoss(saturation_temp=77.0)
    
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # Loop
    for epoch in range(5):
        model.train()
        pbar = tqdm(range(20), desc=f"Epoch {epoch+1}")
        epoch_loss = 0
        
        for _ in pbar:
            optimizer.zero_grad()
            
            # 1. Get Data
            inputs, targets = generate_synthetic_batch()
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 2. Forward
            pred = model(inputs)
            
            # 3. Loss
            loss = loss_fn.calculate(pred, targets)
            
            # 4. Update
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/20:.4f}")
        
    # Save
    save_path = os.path.join(save_dir, "model_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
