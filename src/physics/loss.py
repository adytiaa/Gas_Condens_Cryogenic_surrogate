import torch

class CryoPhysicsLoss:
    def __init__(self, saturation_temp=77.0):
        self.t_sat = saturation_temp

    def lee_model_loss(self, pred_alpha, pred_temp):
        "  
            Calculates a physical consistency loss based on the Lee Model.
            Equation: If Temp < Tsat, Condensation should occur (Alpha > 0).
        
        "
        # Mask: Where is the fluid colder than saturation?
        cold_region = (pred_temp < self.t_sat).float()
        
        # In cold regions, we penalize if Liquid Fraction (Alpha) is low.
        # Ideally, Alpha should approach 1.0 (Liquid) when T < Tsat.
        # Loss = Cold_Mask * (1 - Alpha)^2
        loss = torch.mean(cold_region * torch.pow((1.0 - pred_alpha), 2))
        return loss

    def calculate(self, predictions, targets=None):
        # predictions shape: [Batch, X, Y, Z, 2] -> (Alpha, Temp)
        alpha = predictions[..., 0]
        temp = predictions[..., 1]
        
        total_loss = 0.0
        
        # 1. Data Loss (if ground truth available)
        if targets is not None:
            mse_loss = torch.nn.functional.mse_loss(predictions, targets)
            total_loss += mse_loss

        # 2. Physics Constraint (Unsupervised Loss)
        # Penalize non-physical states
        phys_loss = self.lee_model_loss(alpha, temp)
        
        return total_loss + (0.1 * phys_loss)
