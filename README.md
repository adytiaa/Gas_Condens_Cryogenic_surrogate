# Surrogate Models for Simulation of Gas Condensation in Cryogenic Vessels 

![onnes-condes.jpg]()

**Agent for Cryogenic Gas Condensation (PI-GANO / FNO approach)**

This repository implements a **Physics-Informed Neural Operator (FNO)** to act as a surrogate solver for cryogenic fluid dynamics. It predicts liquid formation (condensation) in 3D vessels instantly, replacing slow CFD simulations (OpenFOAM/Ansys).

## ⚡ Quick Start

1.  **Install Environment**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Agent (Demo)**
    This runs the agent on a random geometry to demonstrate the pipeline. 
    *Note: It uses random weights initially.*
    ```bash
    python scripts/run_agent.py
    ```

3.  **Train the Physics Model**
    To make the agent accurate, train it using the synthetic data generator included:
    ```bash
    python scripts/train.py
    ```

## Architecture
*   **Model:** 3D Fourier Neural Operator (FNO). Learns resolution-independent solution operators.
*   **Physics:** Implements Lee Model constraints for Phase Change (Gas -> Liquid).
*   **Agent:** Automates the loop of Geometry -> Inference -> Analysis.

## Folder Structure
*   `src/models/`: Neural Network architectures.
*   `src/physics/`: Loss functions and thermodynamic equations.
*   `data/`: Storage for VTK/Numpy files.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---------------------------------------------------
