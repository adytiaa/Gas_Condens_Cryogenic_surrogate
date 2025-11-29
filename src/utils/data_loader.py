import os
import torch
import numpy as np
from torch.utils.data import Dataset

class CryoDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        if len(self.files) == 0:
            raise RuntimeError(f"No data found in {data_dir}. Run scripts/generate_data.py first!")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load npz
        data = np.load(self.files[idx])
        
        # Raw Data
        geo = data['geometry']      # (32, 32, 32)
        wall_temp = float(data['wall_temp'])
        target_temp = data['temperature']
        target_alpha = data['liquid_fraction']
        
        # --- PREPROCESSING INPUTS ---
        # 1. Normalize Temperature Channel (Input)
        # Normalize relative to Saturation (77K)
        # Range approx -1.0 to 0.0
        norm_wall_temp = (wall_temp - 77.0) / 10.0
        
        # Create full field for wall temp
        D, H, W = geo.shape
        temp_input_field = np.full((D, H, W), norm_wall_temp, dtype=np.float32)
        
        # Pressure placeholder (Constant 1 atm for now)
        press_input_field = np.zeros((D, H, W), dtype=np.float32)
        
        # Stack Inputs: [Geo, Wall_Temp, Pressure] -> (3, D, H, W)
        inputs = np.stack([geo, temp_input_field, press_input_field], axis=0)
        
        # --- PREPROCESSING TARGETS ---
        # Stack Targets: [Alpha, Temp] -> (2, D, H, W)
        # We normalize target temp similarly
        norm_target_temp = (target_temp - 77.0) / 10.0
        targets = np.stack([target_alpha, norm_target_temp], axis=0)
        
        return torch.from_numpy(inputs).float(), torch.from_numpy(targets).float()