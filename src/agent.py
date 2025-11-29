import torch
import numpy as np
import logging
import os
from src.models.fno_3d import FNO3d

class CryoAgent:
    def __init__(self, model_path=None, device='cuda'):
        self.logger = self._setup_logger()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Agent running on: {self.device}")
        
        # Initialize Architecture (Matches Config)
        self.model = FNO3d(modes=8, width=20, in_channels=3, out_channels=2).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_brain(model_path)
        else:
            self.logger.warning("No model found/provided. Agent is using RANDOM initialization.")

    def _setup_logger(self):
        logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
        return logging.getLogger("CryoAgent")

    def load_brain(self, path):
        self.logger.info(f"Loading weights from {path}...")
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")

    def simulate(self, geometry_grid, wall_temp_k, pressure_pa):
        "
           Main Inference Function.
        "
        self.logger.info("Starting Simulation...")
        self.model.eval()
        
        # 1. Preprocess Input
        # geometry_grid should be (D, H, W)
        D, H, W = geometry_grid.shape
        
        # Create full tensors for constant conditions
        temp_field = np.full((D, H, W), wall_temp_k, dtype=np.float32)
        press_field = np.full((D, H, W), pressure_pa, dtype=np.float32)
        
        # Stack: Channel 0=Geo, 1=Temp, 2=Press
        input_np = np.stack([geometry_grid, temp_field, press_field], axis=-1)
        
        # To Torch (Batch dim 0)
        input_tensor = torch.from_numpy(input_np).float().unsqueeze(0).to(self.device)
        
        # 2. Inference
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
            
        # 3. Postprocess
        # Output: [1, D, H, W, 2] -> Remove batch
        results = output_tensor.cpu().numpy()[0]
        
        return {
            "liquid_fraction": results[..., 0],
            "temperature_field": results[..., 1]
        }

    def analyze(self, results):
        liq_vol = np.sum(results['liquid_fraction'][results['liquid_fraction'] > 0.1])
        self.logger.info(f"Estimated Liquid Formation Index: {liq_vol:.2f}")
