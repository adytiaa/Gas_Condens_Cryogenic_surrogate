import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agent import CryoAgent

def main():
    model_path = os.path.join("checkpoints", "model_final.pth")
    
    # 1. Initialize Agent
    if not os.path.exists(model_path):
        print("Wait! No trained model found at checkpoints/model_final.pth")
        print("Initializing agent with random weights (Results will be noise).")
        agent = CryoAgent(model_path=None)
    else:
        agent = CryoAgent(model_path=model_path)
        
    # 2. Define a Cylinder Tank Geometry
    # 32x32x32 Grid
    N = 32
    x, y, z = np.indices((N, N, N))
    center = N // 2
    # Equation for cylinder along Z axis
    mask = (x - center)**2 + (y - center)**2 < (N//3)**2
    geometry = mask.astype(np.float32)
    
    print(f"Geometry created: Cylinder (Voxels: {np.sum(geometry)})")
    
    # 3. Run Simulation
    results = agent.simulate(geometry, wall_temp_k=70.0, pressure_pa=101325)
    
    # 4. Analyze
    agent.analyze(results)
    
    # 5. Visualize (Central Slice)
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.title("Liquid Fraction (Slice)")
    plt.imshow(results['liquid_fraction'][center, :, :], cmap="Blues", origin='lower')
    plt.colorbar(label="Volume Fraction")
    
    plt.subplot(1, 2, 2)
    plt.title("Temperature (Slice)")
    plt.imshow(results['temperature_field'][center, :, :], cmap="inferno", origin='lower')
    plt.colorbar(label="Kelvin")
    
    print("Close plot window to finish.")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
