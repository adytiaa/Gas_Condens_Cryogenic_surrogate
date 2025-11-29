import os
import numpy as np
import scipy.ndimage
from tqdm import tqdm

# Configuration
DATA_DIR = os.path.join("data", "processed")
NUM_SAMPLES = 100  # Small but enough for a demo
GRID_SIZE = 32     # 32x32x32 voxels
T_SAT = 77.0       # Saturation Temp of Nitrogen (K)

def generate_cylinder_mask(grid_size, radius_ratio):
    """Creates a binary mask of a cylinder."""
    N = grid_size
    x, y, z = np.indices((N, N, N))
    center = N // 2
    radius = (N // 2) * radius_ratio
    
    # Cylinder along Z-axis
    mask = (x - center)**2 + (y - center)**2 < radius**2
    return mask.astype(np.float32)

def generate_physics_fields(geometry, wall_temp):
    """
    Approximates CFD physics using Distance Fields.
    """
    # 1. Calculate Distance from Wall (Inside the tank)
    # Invert geometry (0 inside, 1 outside) for distance transform
    inverted_geo = 1.0 - geometry
    # Calculate distance to nearest wall (where inverted_geo is 1)
    dist_map = scipy.ndimage.distance_transform_edt(geometry)
    
    # Normalize distance (0 at wall, 1 at center)
    max_dist = np.max(dist_map) + 1e-5
    norm_dist = dist_map / max_dist
    
    # 2. Temperature Field Approximation
    # Temp is T_wall at distance 0, decays exponentially to T_sat at center
    # Formula: T = T_sat - (T_sat - T_wall) * exp(-decay * dist)
    decay_rate = 3.0  # Controls boundary layer thickness
    temp_field = T_SAT - (T_SAT - wall_temp) * np.exp(-decay_rate * norm_dist)
    
    # Apply geometry mask (Outside is 300K or 0)
    temp_field = temp_field * geometry
    
    # 3. Liquid Fraction Approximation (Alpha)
    # Liquid forms where Temp < T_sat.
    # We also add a gravity bias (Pool at the bottom).
    
    # A. Condensation at walls
    alpha_wall = np.exp(-decay_rate * norm_dist * 2) # Liquid film is thinner than thermal layer
    
    # B. Gravity Pooling (Linear gradient along Z)
    z_indices = np.linspace(0, 1, geometry.shape[2]) # 0 bottom, 1 top
    # Broadcast Z to 3D
    z_map = np.tile(z_indices[np.newaxis, np.newaxis, :], (geometry.shape[0], geometry.shape[1], 1))
    
    # Pool forms at bottom (low Z)
    pool_height = np.random.uniform(0.1, 0.4) # Random fill level
    alpha_pool = (z_map < pool_height).astype(np.float32)
    
    # Combine (Max of wall film and bottom pool)
    alpha_field = np.maximum(alpha_wall, alpha_pool)
    
    # Noise injection (to make it robust)
    noise = np.random.normal(0, 0.02, alpha_field.shape)
    alpha_field = np.clip(alpha_field + noise, 0.0, 1.0) * geometry
    
    return temp_field, alpha_field

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Generating {NUM_SAMPLES} synthetic CFD samples in {DATA_DIR}...")
    
    for i in tqdm(range(NUM_SAMPLES)):
        # Randomize Simulation Parameters
        # Wall temp between 65K and 76K
        wall_temp = np.random.uniform(65.0, 76.0) 
        
        # Randomize Cylinder Radius
        radius_ratio = np.random.uniform(0.6, 0.9)
        
        # 1. Create Geometry
        geometry = generate_cylinder_mask(GRID_SIZE, radius_ratio)
        
        # 2. Simulate Physics
        temp, alpha = generate_physics_fields(geometry, wall_temp)
        
        # 3. Save as Numpy Dictionary
        save_path = os.path.join(DATA_DIR, f"sample_{i:03d}.npz")
        np.savez(save_path, 
                 geometry=geometry, 
                 wall_temp=np.array(wall_temp), 
                 temperature=temp, 
                 liquid_fraction=alpha)
        
    print("✅ Data generation complete.")

if __name__ == "__main__":
    main()