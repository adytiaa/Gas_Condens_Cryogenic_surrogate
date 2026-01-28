To perform a simulation of this complexity, we use **Python as a wrapper** to generate OpenFOAM case files, manage the execution, and post-process the data.

This Jupyter Notebook approach uses `interCondensatingEvaporatingFoam`. This solver handles two phases, heat transfer, and phase change (evaporation/condensation).

### Prerequisites
1.  **OpenFOAM (v2312 or later recommended)** installed on your system.
2.  **Python libraries**: `numpy`, `matplotlib`, `pandas`.

---

# Jupyter Notebook: Cryogenic LN2 Boil-off & Flow Analysis

## 1. Mathematical Model and Physics

### 1.1 Volume of Fluid (VOF) Method
The interface is tracked by the phase fraction $\alpha$, where $\alpha=1$ is liquid (LN2) and $\alpha=0$ is gas (GN2).
$$\frac{\partial \alpha}{\partial t} + \nabla \cdot (\alpha \mathbf{U}) = \frac{\dot{m}}{\rho_l}$$

### 1.2 Phase Change (Lee Model)
Since the wall is at 400K (well above $T_{sat}$), the mass transfer $\dot{m}$ is driven by the temperature difference:
- **Evaporation ($T > T_{sat}$):**
$$\dot{m}_e = C_e \cdot \alpha \rho_l \frac{T - T_{sat}}{T_{sat}}$$
Where $C_e$ is the relaxation coefficient (1/s).

### 1.3 Momentum and Energy
A single momentum equation is solved for the mixture:
$$\frac{\partial (\rho \mathbf{U})}{\partial t} + \nabla \cdot (\rho \mathbf{U} \mathbf{U}) = -\nabla p + \nabla \cdot \tau + \rho \mathbf{g} + \mathbf{f}_\sigma$$
The energy equation tracks the temperature $T$, accounting for the latent heat of vaporization $L$:
$$\frac{\partial (\rho C_p T)}{\partial t} + \nabla \cdot (\rho C_p \mathbf{U} T) = \nabla \cdot (k \nabla T) - \dot{m}L$$

---

## 2. Parameter Setup

```python
import os
import numpy as np

# Pipe Geometry (ID: 25mm, Thickness: 2.5mm)
ID = 0.025 
Radius = ID / 2
Limb_Height = 0.5  # 500mm
U_Width = 0.1      # 100mm between limbs

# Operating Conditions
P_abs = 4e5        # 4 bar
T_inlet = 77       # K (Subcooled LN2)
T_wall = 400       # K
T_sat = 91.3       # K at 4 bar
Mass_Flow = 0.089  # kg/s

# Derived Velocity
rho_L = 808        # kg/m3 (approx for LN2)
Area = np.pi * (Radius**2)
U_mag = Mass_Flow / (rho_L * Area)

print(f"Calculated Inlet Velocity: {U_mag:.4f} m/s")
```

---

## 3. Case Generation (OpenFOAM Dictionaries)

We will programmatically create the directory structure.

```python
case_name = "LN2_U_Pipe_Simulation"
os.makedirs(f"{case_name}/0", exist_ok=True)
os.makedirs(f"{case_name}/constant", exist_ok=True)
os.makedirs(f"{case_name}/system", exist_ok=True)

# ---------------------------------------------------------
# 1. Thermophysical Properties (Liquid and Gas)
# ---------------------------------------------------------
thermophysicalProperties = f"""
FoamFile {{ version 2.0; format ascii; class dictionary; location "constant"; object thermophysicalProperties; }}

phases (liquid gas);

liquid
{{
    specie {{ nmoles 1; molWeight 28.01; }}
    thermo {{ cp 2040; hf 0; }}
    transport {{ mu 1.58e-4; Pr 2.2; }}
    equationOfState {{ rho 808; }}
}}

gas
{{
    specie {{ nmoles 1; molWeight 28.01; }}
    thermo {{ cp 1040; hf 199e3; }} // hf is Latent Heat
    transport {{ mu 1.7e-5; Pr 0.7; }}
    equationOfState {{ rho 15.0; }} // Density at 4 bar
}}
"""

with open(f"{case_name}/constant/thermophysicalProperties", "w") as f:
    f.write(thermophysicalProperties)
```

---

## 4. Boundary Conditions (The "0" Folder)

The most critical setup is the `T` and `alpha.liquid` files.

```python
# T (Temperature)
T_file = f"""
FoamFile {{ version 2.0; format ascii; class volScalarField; object T; }}
dimensions      [0 0 0 1 0 0 0];
internalField   uniform {T_sat};
boundaryField
{{
    inlet   {{ type fixedValue; value uniform {T_inlet}; }}
    outlet  {{ type inletOutlet; inletValue uniform {T_sat}; value uniform {T_sat}; }}
    walls   {{ type fixedValue; value uniform {T_wall}; }}
}}
"""

# alpha.liquid (Phase Fraction)
alpha_file = f"""
FoamFile {{ version 2.0; format ascii; class volScalarField; object alpha.liquid; }}
dimensions      [0 0 0 0 0 0 0];
internalField   uniform 0; // Start empty (gas filled)
boundaryField
{{
    inlet   {{ type fixedValue; value uniform 1; }}
    outlet  {{ type zeroGradient; }}
    walls   {{ type zeroGradient; }}
}}
"""

with open(f"{case_name}/0/T", "w") as f: f.write(T_file)
with open(f"{case_name}/0/alpha.liquid", "w") as f: f.write(alpha_file)
```

---

## 5. ControlDict & Function Objects
To track **Pressure Drop** and **Boil-off** automatically during the run.

```python
controlDict = f"""
application     interCondensatingEvaporatingFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         10.0;
deltaT          1e-5; // Start small for stability
writeControl    adjustableRunTime;
writeInterval   0.1;
maxCo           0.5;

functions
{{
    pressureDrop
    {{
        type            patchAverage;
        libs            (fieldFunctionObjects);
        patches         (inlet outlet);
        fields          (p);
    }}
    
    boilOffRate
    {{
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        regionType      patch;
        name            outlet;
        operation       sum;
        fields          (phi); // Mass flux
        writeControl    timeStep;
    }}
}}
"""
with open(f"{case_name}/system/controlDict", "w") as f: f.write(controlDict)
```

---

## 6. Running the Simulation
You would run the following commands in your terminal within the case folder:

```bash
# Generate Mesh (Assuming you've created a blockMeshDict)
blockMesh

# Initialize with setFields (Fill the bottom of the U if needed)
setFields

# Run the Solver
interCondensatingEvaporatingFoam
```

---

## 7. Data Analysis (Post-Processing)

After the simulation, we use Python to plot the focus areas: **Boil-off mass rate** and **Pressure Drop**.

```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(log_file):
    # This assumes you have extracted data from OpenFOAM logs or postProcessing folder
    # For this example, we simulate data loading
    time = np.linspace(0, 10, 100)
    
    # Boil-off trend: Starts high (chilling wall) then stabilizes
    boil_off = 0.02 * np.exp(-time/2) + 0.005 
    
    # Pressure Drop: Increases as pipe fills and gas accelerates
    p_drop = 5000 + 15000 * (1 - np.exp(-time/3))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Boil-off Rate (kg/s)', color='red')
    ax1.plot(time, boil_off, color='red', label='Mass Loss')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Pressure Drop (Pa)', color='blue')
    ax2.plot(time, p_drop, color='blue', label='Delta P')

    plt.title('Cryogenic Chill-down Performance')
    plt.show()

# plot_results("postProcessing/boilOffRate/0/surfaceFieldValue.dat")
```

### Key Considerations for your Objectives:
1.  **Flow Regimes:** You must use **ParaView** to look at the `alpha.liquid` field. Look for "Slug" (large bubbles blocking the pipe) vs "Annular" (liquid film on the 400K wall, gas core).
2.  **Saturation State:** In your plots, the state is reached when the slope of the `T_bulk` and `Boil-off Rate` curves approach zero.
3.  **Pressure Drop:** Note that $\Delta P$ will spike significantly when slugs of liquid nitrogen are accelerated by expanding gas.
4.  **Mesh Sensitivity:** Since you have a 400K wall and 77K fluid, the temperature gradient is massive. You **must** have a very fine mesh near the wall (boundary layer) to capture the film boiling correctly. Use `snappyHexMesh` for the U-bend curvature.


To provide an accurate answer, we must distinguish between **Physical Time** (how many seconds occur inside the simulation) and **Computational Time** (how many hours your computer works).

In a cryogenic chill-down simulation (LN2 entering a 400K pipe), the pressure drop is not a single constant value; it is a highly transient signal.

---

### 1. Physical Time (How long to see the result?)
Based on your parameters (Mass flow 89g/s, ID 25mm, 4 bar):

*   **Initial Velocity ($U_{in}$):** $\approx 0.22 \, \text{m/s}$ (Liquid phase).
*   **The "First Pressure Signal":** You will see a pressure change at the inlet the **instant** the simulation starts because the liquid hits the 400K wall and flashes into gas, creating a back-pressure wave.
*   **Time to reach "Pseudo-Steady" Pressure Drop:** 
    *   If your U-pipe limbs are 0.5m each and the base is 0.2m (Total length $\approx 1.2\text{m}$), the liquid front takes about **5 to 8 seconds** to reach the outlet.
    *   **However**, because of the 400K wall, the liquid will boil violently. The gas moves much faster ($> 5 \, \text{m/s}$). You will get a meaningful pressure drop reading for the **two-phase mixture** once the pipe is "chilled" and a continuous flow path is established, typically **10–15 seconds** of physical time.

### 2. Computational Time (How long to run the simulation?)
`interCondensatingEvaporatingFoam` is a "stiff" solver because it couples the VOF interface with intense phase-change mass transfers.

**Estimated Calculation for a Workstation (e.g., 16-core AMD Ryzen/Intel Xeon):**
*   **Mesh Size:** Assuming a standard hex-dominant mesh with boundary layers ($\approx 500,000$ cells).
*   **Time Step ($\Delta t$):** Limited by the Courant Number ($Co < 0.5$). Due to rapid gas expansion, your $\Delta t$ will likely drop to **$1 \times 10^{-5}$ or $5 \times 10^{-6}$ seconds**.
*   **Throughput:** Your computer might process roughly 100–200 iterations per minute.
*   **Total Runtime:** 
    *   To simulate **1 second** of physical time, you need $\sim 100,000$ steps.
    *   This usually translates to **12 to 24 hours of wall-clock time per 1 second of simulation**.
    *   To reach a stable pressure drop (10 seconds physical time), expect the simulation to run for **5 to 10 days** on a standard high-end workstation.

---

### 3. What the Pressure Drop Curve will look like
When you extract the data, you won't see one number. You will see three distinct phases:

1.  **The Flash Spike (0s - 1s):** A massive initial spike in pressure (could be 1-2 bar higher than operating pressure) as the first LN2 hits the 400K wall and expands 175x in volume.
2.  **The Slug/Chugging Phase (1s - 8s):** High-frequency oscillations. As liquid slugs move through the U-bend, they momentarily "block" the gas, causing the pressure to pulse.
3.  **The Leidenfrost/Pseudo-Steady Phase (8s+):** The pressure drop stabilizes as a vapor film forms on the walls. This is the value you usually report for "Operating Pressure Drop."

---

### 4. How to get the result faster?
If 10 days of simulation is too long, use these OpenFOAM strategies:

*   **Local Time Stepping (LTS):** If you *only* care about the final steady-state pressure drop and not the transient filling process, use `interCondensatingEvaporatingSteadyFoam` (if available in your version) or use the LTS version of the solver. This can reduce compute time by 80%.
*   **2D Axisymmetric Mesh:** Instead of a full 3D pipe, simulate a 2D slice. For a U-pipe, this is difficult due to the bends, but a coarse 3D mesh with a high-quality "wedge" for the straight sections helps.
*   **Sub-cycling:** Set `nAlphaSubCycles 2;` in `fvSolution` to allow the interface to be calculated more frequently than the pressure-velocity, allowing for a slightly larger global $\Delta t$.

### Summary Table
| Metric | Estimate |
| :--- | :--- |
| **Physical Time to Fill Pipe** | 5–10 seconds |
| **Physical Time to Stable $\Delta P$** | 10–15 seconds |
| **Simulation Time Step ($\Delta t$)** | $10^{-5}$ to $10^{-6}$ s |
| **Computational Time (Workstation)** | ~18 hours per simulated second |
| **Total Wait Time for Results** | **1 to 2 weeks** (for a full transient run) |






# Detailed Analysis Approach for LN2 U-Shaped Pipe Multiphase Flow

## Problem Overview

You're simulating a **cryogenic multiphase flow** problem where liquid nitrogen (LN2 at 77K) enters a U-shaped pipe with walls at 400K. This creates significant boiling and phase change phenomena. This is a complex problem requiring transient multiphase simulation with heat transfer and phase change modeling.

## Detailed Analysis of Each Objective

### 1. **Multiphase Phenomenon & Flow Regimes**

For LN2 boiling in the pipe, you'll encounter multiple flow regimes:
- **Bubbly flow**: Small vapor bubbles dispersed in liquid (initial phase)
- **Slug flow**: Large vapor bubbles occupying pipe cross-section
- **Annular flow**: Liquid film on walls with vapor core
- **Mist flow**: Droplets in vapor stream (near outlet)

The flow regime depends on void fraction (α), which changes along the pipe length due to heat input from hot walls.

**Analysis Method**: Track phase fraction (alpha) spatially and temporally. Visualize using iso-surfaces and contour plots.

### 2. **Pressure Drops**

Pressure drop has multiple components:
- **Frictional losses**: Due to wall shear (single-phase and two-phase multipliers)
- **Acceleration pressure drop**: Due to phase change (liquid→vapor density change)
- **Gravitational pressure drop**: Hydrostatic head in U-bend
- **Form losses**: At bends and geometry changes

**Analysis Method**: Monitor pressure at multiple locations (inlet, mid-height, U-bend bottom, outlet). Calculate ΔP = P_inlet - P_outlet.

### 3. **Fill Time**

Time for liquid to completely fill the column from initial condition to steady flow:
- Initially, pipe may be empty or contain gas
- Liquid front propagates based on inlet velocity and gravity
- Complicated by simultaneous boiling and vapor generation

**Analysis Method**: Track liquid volume fraction integral over entire domain vs time: V_liquid(t) = ∫α·dV. Fill complete when V_liquid stabilizes.

### 4. **Saturation State Time**

Time to reach thermal equilibrium where:
- Liquid temperature reaches saturation temperature (77K at local pressure)
- Vapor generation rate stabilizes
- Temperature field becomes quasi-steady

**Analysis Method**: Monitor temperature and phase change rate. Saturation reached when ∂(mass_vapor)/∂t becomes constant.

### 5. **Boil-off Losses**

Mass of liquid that vaporizes per second due to wall heat input:
- Heat flux from 400K wall to 77K fluid
- Energy balance: Q = ṁ_vapor × h_fg
- Where h_fg is latent heat of vaporization for LN2 (~200 kJ/kg)

**Analysis Method**: Calculate vapor generation rate: ṁ_boiloff = ∫(ρ_vapor × U_vapor · n)dA at outlet or track phase change source term.

## OpenFOAM Solver Selection

For this problem, you need a multiphase solver with:
- Phase change modeling (boiling/condensation)
- Heat transfer
- Compressibility effects (high pressure)

**Recommended solvers**:
- `icoReactingMultiphaseInterFoam`: Multiphase with phase change
- `reactingTwoPhaseEulerFoam`: Eulerian multiphase with reactions
- `multiphaseEulerFoam`: Multiple Eulerian phases

## Python Code Implementation

Based on the **Foam-Agent** framework in this repository [1](#0-0) , I'll provide Python code that:

1. **Generates a user requirement file** for Foam-Agent to automatically set up the OpenFOAM case
2. **Creates post-processing scripts** to extract all the objectives

### Code Part 1: Generate User Requirement for Foam-Agent

```python
"""
LN2 U-Pipe Multiphase Flow - Foam-Agent User Requirement Generator
This script creates a detailed requirement file for Foam-Agent to set up
the OpenFOAM simulation automatically.
"""

def generate_user_requirement():
    """Generate user requirement text file for Foam-Agent"""
    
    requirement = """
# LN2 U-Shaped Pipe Multiphase Boiling Simulation

## Geometry
- U-shaped pipe with two vertical limbs facing upwards
- Internal diameter: 25 mm
- Wall thickness: 2.5 mm
- Material: Stainless steel

## Fluid Properties
- Operating fluid: Liquid Nitrogen (LN2)
- Two phases: liquid and vapor
- Liquid properties at 77K:
  * Density: 808 kg/m³
  * Viscosity: 1.58e-4 Pa·s
  * Thermal conductivity: 0.14 W/m·K
  * Specific heat: 2040 J/kg·K
- Vapor properties at 77K:
  * Density: 4.6 kg/m³ (at 4 bar)
  * Viscosity: 5.5e-6 Pa·s
- Latent heat of vaporization: 199 kJ/kg
- Saturation temperature: 77K at 1 bar (adjust for 4 bar)

## Boundary Conditions
- Inlet (shorter limb, bottom):
  * Fixed velocity corresponding to mass flow rate 89 g/s
  * Velocity = (0.089 kg/s) / (808 kg/m³ × π × (0.0125m)²) ≈ 0.224 m/s
  * Temperature: 77K
  * Pressure: 4 bar (400 kPa)
  * Phase: pure liquid (alpha.liquid = 1)
  
- Outlet (top of longer limb):
  * Pressure outlet or outflow condition
  * Zero gradient for other fields
  
- Walls:
  * No-slip condition (U = 0)
  * Fixed temperature: 400K
  * Heat transfer from wall to fluid enabled

## Initial Conditions
- Pressure: 4 bar throughout domain
- Temperature: 77K in fluid domain
- Velocity: zero everywhere
- Phase: can start with empty pipe or pre-filled liquid

## Solver Requirements
- Use multiphase solver with phase change capability (icoReactingMultiphaseInterFoam or reactingTwoPhaseEulerFoam)
- Enable heat transfer and phase change modeling
- Include buoyancy effects (gravity in vertical direction)
- Transient simulation

## Simulation Parameters
- Time step: adaptive based on Courant number (max Co = 0.5)
- End time: sufficient for filling and reaching steady state (suggest 10-30 seconds)
- Write interval: every 0.1 seconds for detailed analysis
- Turbulence: laminar or k-epsilon if Reynolds number warrants

## Output Requirements
Track the following fields at each time step:
- Pressure (p)
- Velocity (U)
- Temperature (T)
- Phase fraction (alpha.liquid, alpha.vapor)
- Phase change rate
- Heat flux at walls

## Objectives
1. Identify multiphase flow regimes (bubbly, slug, annular, mist)
2. Calculate pressure drop from inlet to outlet
3. Determine fill time (time for liquid to fill entire column)
4. Find time to reach saturation state (thermal equilibrium)
5. Calculate boil-off losses (mass of liquid vaporizing per second)
"""
    
    # Save to file
    with open('user_requirement_ln2_pipe.txt', 'w') as f:
        f.write(requirement)
    
    print("✓ User requirement file generated: user_requirement_ln2_pipe.txt")
    print("\nTo run with Foam-Agent:")
    print("python foambench_main.py --output ./ln2_output --prompt_path ./user_requirement_ln2_pipe.txt")
    
    return requirement


if __name__ == "__main__":
    generate_user_requirement()
``` [2](#0-1) 

### Code Part 2: Post-Processing Analysis Script

```python
"""
LN2 U-Pipe Post-Processing Script
Extracts all required objectives from OpenFOAM results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not available. Some visualizations will be skipped.")


class LN2PipeAnalyzer:
    """Analyzer for LN2 U-pipe multiphase simulation results"""
    
    def __init__(self, case_dir):
        """
        Initialize analyzer
        
        Args:
            case_dir: Path to OpenFOAM case directory
        """
        self.case_dir = Path(case_dir)
        self.foam_file = self.case_dir / f"{self.case_dir.name}.foam"
        self.time_dirs = self._get_time_directories()
        
        # Physical properties
        self.rho_liquid = 808.0  # kg/m³
        self.rho_vapor = 4.6     # kg/m³
        self.h_fg = 199000.0     # J/kg (latent heat)
        self.m_dot_inlet = 0.089 # kg/s
        
        print(f"Initialized analyzer for case: {case_dir}")
        print(f"Found {len(self.time_dirs)} time directories")
    
    def _get_time_directories(self):
        """Get sorted list of time directories"""
        time_dirs = []
        for item in self.case_dir.iterdir():
            if item.is_dir() and item.name.replace('.', '').replace('-', '').isdigit():
                try:
                    time_val = float(item.name)
                    time_dirs.append((time_val, item))
                except ValueError:
                    continue
        return sorted(time_dirs, key=lambda x: x[0])
    
    def analyze_pressure_drops(self):
        """
        Objective 1: Calculate pressure drops along pipe
        Monitors pressure at inlet, outlet, and intermediate locations
        """
        print("\n" + "="*60)
        print("OBJECTIVE 1: PRESSURE DROP ANALYSIS")
        print("="*60)
        
        if not PYVISTA_AVAILABLE:
            print("PyVista required for pressure analysis")
            return None
        
        pressure_data = {
            'time': [],
            'p_inlet': [],
            'p_outlet': [],
            'delta_p': []
        }
        
        for time_val, time_dir in self.time_dirs:
            try:
                # Read pressure field
                reader = pv.OpenFOAMReader(str(self.foam_file))
                reader.set_active_time_value(time_val)
                mesh = reader.read()
                
                if 'p' in mesh.array_names:
                    p_field = mesh['p']
                    
                    # Extract min/max as proxies for inlet/outlet
                    # In practice, you'd probe specific locations
                    p_inlet = np.max(p_field)
                    p_outlet = np.min(p_field)
                    delta_p = p_inlet - p_outlet
                    
                    pressure_data['time'].append(time_val)
                    pressure_data['p_inlet'].append(p_inlet)
                    pressure_data['p_outlet'].append(p_outlet)
                    pressure_data['delta_p'].append(delta_p)
                    
            except Exception as e:
                print(f"Warning: Could not read time {time_val}: {e}")
                continue
        
        # Plot pressure drop evolution
        if len(pressure_data['time']) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(pressure_data['time'], pressure_data['delta_p'], 'b-', linewidth=2)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Pressure Drop (Pa)', fontsize=12)
            plt.title('Pressure Drop Evolution in LN2 U-Pipe', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.case_dir / 'pressure_drop_analysis.png', dpi=300)
            print(f"✓ Pressure drop plot saved")
            
            # Print summary
            if len(pressure_data['delta_p']) > 0:
                final_dp = pressure_data['delta_p'][-1]
                mean_dp = np.mean(pressure_data['delta_p'][-10:])  # Last 10 values
                print(f"\nPressure Drop Results:")
                print(f"  Final ΔP: {final_dp:.2f} Pa")
                print(f"  Mean ΔP (steady state): {mean_dp:.2f} Pa")
        
        return pressure_data
    
    def analyze_fill_time(self):
        """
        Objective 2: Calculate time for complete fill of liquid in column
        Track liquid volume fraction evolution
        """
        print("\n" + "="*60)
        print("OBJECTIVE 2: FILL TIME ANALYSIS")
        print("="*60)
        
        if not PYVISTA_AVAILABLE:
            print("PyVista required for fill time analysis")
            return None
        
        fill_data = {
            'time': [],
            'liquid_volume_fraction': [],
            'liquid_volume': []
        }
        
        for time_val, time_dir in self.time_dirs:
            try:
                reader = pv.OpenFOAMReader(str(self.foam_file))
                reader.set_active_time_value(time_val)
                mesh = reader.read()
                
                # Look for phase fraction field (alpha.liquid or similar)
                alpha_field = None
                for name in ['alpha.liquid', 'alpha.water', 'alpha']:
                    if name in mesh.array_names:
                        alpha_field = mesh[name]
                        break
                
                if alpha_field is not None:
                    # Calculate liquid volume
                    cell_volumes = mesh.compute_cell_sizes()['Volume']
                    liquid_volume = np.sum(alpha_field * cell_volumes)
                    total_volume = np.sum(cell_volumes)
                    liquid_fraction = liquid_volume / total_volume
                    
                    fill_data['time'].append(time_val)
                    fill_data['liquid_volume'].append(liquid_volume)
                    fill_data['liquid_volume_fraction'].append(liquid_fraction)
                    
            except Exception as e:
                print(f"Warning: Could not process time {time_val}: {e}")
                continue
        
        # Determine fill time (when liquid fraction reaches 95% of steady state)
        if len(fill_data['time']) > 0:
            fractions = np.array(fill_data['liquid_volume_fraction'])
            times = np.array(fill_data['time'])
            
            if len(fractions) > 5:
                steady_state_fraction = np.mean(fractions[-5:])
                threshold = 0.95 * steady_state_fraction
                
                fill_time_idx = np.where(fractions >= threshold)[0]
                if len(fill_time_idx) > 0:
                    fill_time = times[fill_time_idx[0]]
                    print(f"\n✓ Fill Time: {fill_time:.3f} seconds")
                    print(f"  (Time to reach 95% of steady-state liquid volume)")
                else:
                    print("\n⚠ Fill time not reached in simulation duration")
            
            # Plot fill evolution
            plt.figure(figsize=(10, 6))
            plt.plot(fill_data['time'], fill_data['liquid_volume_fraction'], 'g-', linewidth=2)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Liquid Volume Fraction', fontsize=12)
            plt.title('Liquid Fill Evolution in U-Pipe', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.case_dir / 'fill_time_analysis.png', dpi=300)
            print(f"✓ Fill time plot saved")
        
        return fill_data
    
    def analyze_saturation_state(self):
        """
        Objective 3: Time to attain saturation state
        Monitor temperature field and phase equilibrium
        """
        print("\n" + "="*60)
        print("OBJECTIVE 3: SATURATION STATE ANALYSIS")
        print("="*60)
        
        if not PYVISTA_AVAILABLE:
            print("PyVista required for saturation analysis")
            return None
        
        saturation_data = {
            'time': [],
            'mean_temp': [],
            'temp_std': [],
            'vapor_fraction': []
        }
        
        T_sat = 77.0  # K (approximate for LN2)
        
        for time_val, time_dir in self.time_dirs:
            try:
                reader = pv.OpenFOAMReader(str(self.foam_file))
                reader.set_active_time_value(time_val)
                mesh = reader.read()
                
                if 'T' in mesh.array_names:
                    T_field = mesh['T']
                    mean_temp = np.mean(T_field)
                    temp_std = np.std(T_field)
                    
                    saturation_data['time'].append(time_val)
                    saturation_data['mean_temp'].append(mean_temp)
                    saturation_data['temp_std'].append(temp_std)
                
                # Track vapor fraction if available
                alpha_vapor = None
                for name in ['alpha.vapor', 'alpha.air']:
                    if name in mesh.array_names:
                        alpha_vapor = mesh[name]
                        break
                
                if alpha_vapor is not None:
                    vapor_frac = np.mean(alpha_vapor)
                    saturation_data['vapor_fraction'].append(vapor_frac)
                    
            except Exception as e:
                print(f"Warning: Could not process time {time_val}: {e}")
                continue
        
        # Determine saturation time (when temperature stabilizes)
        if len(saturation_data['time']) > 0:
            temps = np.array(saturation_data['mean_temp'])
            times = np.array(saturation_data['time'])
            
            if len(temps) > 10:
                # Calculate rate of temperature change
                temp_change_rate = np.abs(np.diff(temps))
                
                # Saturation when rate of change < threshold
                threshold = 0.1  # K/timestep
                saturated_idx = np.where(temp_change_rate < threshold)[0]
                
                if len(saturated_idx) > 0:
                    saturation_time = times[saturated_idx[0]]
                    print(f"\n✓ Saturation Time: {saturation_time:.3f} seconds")
                    print(f"  (Temperature change rate < {threshold} K/timestep)")
                else:
                    print("\n⚠ Saturation state not reached in simulation duration")
            
            # Plot temperature evolution
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            ax1.plot(saturation_data['time'], saturation_data['mean_temp'], 'r-', linewidth=2)
            ax1.axhline(y=T_sat, color='k', linestyle='--', label=f'T_sat = {T_sat}K')
            ax1.set_xlabel('Time (s)', fontsize=12)
            ax1.set_ylabel('Mean Temperature (K)', fontsize=12)
            ax1.set_title('Temperature Evolution', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            if len(saturation_data['vapor_fraction']) > 0:
                ax2.plot(saturation_data['time'], saturation_data['vapor_fraction'], 'b-', linewidth=2)
                ax2.set_xlabel('Time (s)', fontsize=12)
                ax2.set_ylabel('Vapor Volume Fraction', fontsize=12)
                ax2.set_title('Vapor Generation Evolution', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.case_dir / 'saturation_analysis.png', dpi=300)
            print(f"✓ Saturation analysis plot saved")
        
        return saturation_data
    
    def analyze_boiloff_losses(self):
        """
        Objective 4: Calculate boil-off losses per second
        Track vapor generation rate and mass flow rate
        """
        print("\n" + "="*60)
        print("OBJECTIVE 4: BOIL-OFF LOSSES ANALYSIS")
        print("="*60)
        
        if not PYVISTA_AVAILABLE:
            print("PyVista required for boil-off analysis")
            return None
        
        boiloff_data = {
            'time': [],
            'vapor_mass': [],
            'boiloff_rate': []
        }
        
        for time_val, time_dir in self.time_dirs:
            try:
                reader = pv.OpenFOAMReader(str(self.foam_file))
                reader.set_active_time_value(time_val)
                mesh = reader.read()
                
                # Calculate vapor mass
                alpha_vapor = None
                for name in ['alpha.vapor', 'alpha.air']:
                    if name in mesh.array_names:
                        alpha_vapor = mesh[name]
                        break
                
                if alpha_vapor is not None:
                    cell_volumes = mesh.compute_cell_sizes()['Volume']
                    vapor_mass = np.sum(alpha_vapor * cell_volumes * self.rho_vapor)
                    
                    boiloff_data['time'].append(time_val)
                    boiloff_data['vapor_mass'].append(vapor_mass)
                    
            except Exception as e:
                print(f"Warning: Could not process time {time_val}: {e}")
                continue
        
        # Calculate boil-off rate (derivative of vapor mass)
        if len(boiloff_data['time']) > 1:
            times = np.array(boiloff_data['time'])
            vapor_masses = np.array(boiloff_data['vapor_mass'])
            
            # Numerical derivative
            boiloff_rates = np.gradient(vapor_masses, times)
            boiloff_data['boiloff_rate'] = boiloff_rates.tolist()
            
            # Calculate steady-state boil-off rate
            if len(boiloff_rates) > 10:
                steady_boiloff = np.mean(boiloff_rates[-10:])
                boiloff_percentage = (steady_boiloff / self.m_dot_inlet) * 100
                
                print(f"\nBoil-off Loss Results:")
                print(f"  Steady-state boil-off rate: {steady_boiloff*1000:.3f} g/s")
                print(f"  Percentage of inlet flow: {boiloff_percentage:.2f}%")
                print(f"  Inlet mass flow rate: {self.m_dot_inlet*1000:.1f} g/s")
            
            # Plot boil-off evolution
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            ax1.plot(times, vapor_masses * 1000, 'm-', linewidth=2)  # Convert to grams
            ax1.set_xlabel('Time (s)', fontsize=12)
            ax1.set_ylabel('Vapor Mass (g)', fontsize=12)
            ax1.set_title('Vapor Mass Evolution', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(times, boiloff_rates * 1000, 'm-', linewidth=2)  # Convert to g/s
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Time (s)', fontsize=12)
            ax2.set_ylabel('Boil-off Rate (g/s)', fontsize=12)
            ax2.set_title('Instantaneous Boil-off Rate', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.case_dir / 'boiloff_analysis.png', dpi=300)
            print(f"✓ Boil-off analysis plot saved")
        
        return boiloff_data
    
    def analyze_flow_regimes(self):
        """
        Objective 5: Visualize flow regimes
        Create visualizations of velocity and phase distributions
        """
        print("\n" + "="*60)
        print("OBJECTIVE 5: FLOW REGIME VISUALIZATION")
        print("="*60)
        
        if not PYVISTA_AVAILABLE:
            print("PyVista required for flow regime visualization")
            return
        
        # Select representative time steps
        if len(self.time_dirs) < 3:
            time_indices = range(len(self.time_dirs))
        else:
            time_indices = [0, len(self.time_dirs)//2, -1]  # Start, middle, end
        
        for idx in time_indices:
            time_val, time_dir = self.time_dirs[idx]
            
            try:
                reader = pv.OpenFOAMReader(str(self.foam_file))
                reader.set_active_time_value(time_val)
                mesh = reader.read()
                
                # Create visualization
                plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
                
                # Add phase fraction contours if available
                for name in ['alpha.liquid', 'alpha.water', 'alpha']:
                    if name in mesh.array_names:
                        plotter.add_mesh(
                            mesh,
                            scalars=name,
                            cmap='coolwarm',
                            show_scalar_bar=True,
                            scalar_bar_args={'title': 'Liquid Fraction'}
                        )
                        break
                
                plotter.add_text(
                    f'Flow Regime at t = {time_val:.3f} s',
                    position='upper_edge',
                    font_size=14,
                    color='black'
                )
                
                # Save visualization
                output_file = self.case_dir / f'flow_regime_t{time_val:.3f}.png'
                plotter.screenshot(str(output_file))
                plotter.close()
                
                print(f"✓ Flow regime visualization saved: {output_file.name}")
                
            except Exception as e:
                print(f"Warning: Could not visualize time {time_val}: {e}")
                continue
        
        print("\nFlow Regime Identification:")
        print("  Examine saved images for:")
        print("  - Bubbly flow: α_liquid > 0.7, dispersed bubbles")
        print("  - Slug flow: α_liquid = 0.3-0.7, large bubbles")
        print("  - Annular flow: α_liquid < 0.3, film on walls")
        print("  - Mist flow: α_liquid < 0.1, droplets in vapor")
    
    def run_complete_analysis(self):
        """Run all analyses and generate comprehensive report"""
        print("\n" + "="*70)
        print(" "*15 + "LN2 U-PIPE COMPLETE ANALYSIS")
        print("="*70)
        print(f"Case directory: {self.case_dir}")
        print(f"Number of time steps: {len(self.time_dirs)}")
        
        # Create .foam file if needed
        if not self.foam_file.exists():
            self.foam_file.touch()
            print(f"✓ Created .foam file: {self.foam_file}")
        
        # Run all analyses
        pressure_data = self.analyze_pressure_drops()
        fill_data = self.analyze_fill_time()
        saturation_data = self.analyze_saturation_state()
        boiloff_data = self.analyze_boiloff_losses()
        self.analyze_flow_regimes()
        
        print("\n" + "="*70)
        print(" "*20 + "ANALYSIS COMPLETE")
        print("="*70)
        print("\nGenerated outputs:")
        print("  1. pressure_drop_analysis.png - Pressure drop evolution")
        print("  2. fill_time_analysis.png - Liquid filling progression")
        print("  3. saturation_analysis.png - Temperature and vapor evolution")
        print("  4. boiloff_analysis.png - Boil-off rate calculations")
        print("  5. flow_regime_t*.png - Flow pattern visualizations")
        print("\nAll plots saved in:", self.case_dir)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Post-process LN2 U-pipe multiphase simulation results'
    )
    parser.add_argument(
        '--case_dir',
        type=str,
        required=True,
        help='Path to OpenFOAM case directory (output from Foam-Agent)'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = LN2PipeAnalyzer(args.case_dir)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    # Example usage
    print(__doc__)
    print("\nUsage Example:")
    print("-" * 60)
    print("python ln2_pipe_postprocess.py --case_dir ./ln2_output/case_name")
    print("\nOr run directly:")
    print("analyzer = LN2PipeAnalyzer('./path/to/openfoam/case')")
    print("analyzer.run_complete_analysis()")
    print("-" * 60)
    
    # Uncomment to run with command line arguments
    # main()
``` [3](#0-2) 

## How to Use These Scripts with Foam-Agent

### Step 1: Generate Requirements

```bash
python generate_ln2_requirement.py
```

This creates `user_requirement_ln2_pipe.txt` with all specifications.

### Step 2: Run Foam-Agent Simulation [4](#0-3) 

```bash
# Using Docker (recommended)
docker run -it \
  -e OPENAI_API_KEY=your-key-here \
  -v $(pwd)/user_requirement_ln2_pipe.txt:/home/openfoam/Foam-Agent/user_requirement.txt \
  --name foamagent \
  leoyue123/foamagent

# Inside container, run:
python foambench_main.py \
  --output ./ln2_output \
  --prompt_path ./user_requirement.txt
```

Foam-Agent will automatically:
- Parse your requirements [5](#0-4) 
- Select appropriate multiphase solver from OpenFOAM v10 [6](#0-5) 
- Generate all configuration files (blockMesh, boundary conditions, controlDict, etc.)
- Execute the simulation
- Optionally create visualizations [7](#0-6) 

### Step 3: Post-Process Results

```bash
python ln2_pipe_postprocess.py --case_dir ./ln2_output/LN2_U_Pipe_Case
```

This extracts all five objectives and generates analysis plots.

## Notes

1. **Solver Selection**: The LLM in Foam-Agent will select the appropriate multiphase solver from OpenFOAM's available solvers. For phase change, `icoReactingMultiphaseInterFoam` or `reactingTwoPhaseEulerFoam` are most suitable [8](#0-7) 

2. **Mesh Generation**: You can either let Foam-Agent generate the geometry using blockMesh/snappyHexMesh, or provide a custom Gmsh `.msh` file for the U-pipe geometry [9](#0-8) 

3. **PyVista for Visualization**: The post-processing script uses PyVista (included in Foam-Agent environment) to read OpenFOAM results and extract field data [10](#0-9) 

4. **Time Directories**: OpenFOAM stores results in numbered directories (0, 0.1, 0.2, ...). The analyzer reads all available time steps for temporal analysis

5. **Cryogenic Properties**: You may need to refine LN2 properties in the requirement file based on your specific conditions (pressure-dependent saturation temperature, etc.)

6. **Error Correction**: Foam-Agent includes automatic error correction that will retry the simulation if it fails, adjusting parameters as needed [11](#0-10) 

This workflow leverages Foam-Agent's automation capabilities while providing detailed post-processing tailored to your specific multiphase cryogenic analysis objectives.

### Citations

**File:** README.md (L1-13)
```markdown
# Foam-Agent

<p align="center">
  <img src="overview.png" alt="Foam-Agent System Architecture" width="600">
</p>

<p align="center">
    <em>An End-to-End Composable Multi-Agent Framework for Automating CFD Simulation in OpenFOAM</em>
</p>

You can visit https://deepwiki.com/csml-rpi/Foam-Agent for a comprehensive introduction and to ask any questions interactively.

**Foam-Agent** is a multi-agent framework that automates the entire **OpenFOAM**-based CFD simulation workflow from a single natural language prompt. By managing the full pipeline—from meshing and case setup to execution and post-processing—Foam-Agent dramatically lowers the expertise barrier for Computational Fluid Dynamics. Evaluated on [FoamBench](https://arxiv.org/abs/2509.20374) of 110 simulation tasks, our framework achieves an **88.2% success rate**, demonstrating how specialized multi-agent systems can democratize complex scientific computing.
```

**File:** README.md (L35-38)
```markdown
### 🛠️ **Intelligent Error Correction**
- **Error pattern recognition** for common simulation failures
- **Automatic diagnosis and resolution** of configuration issues
- **Iterative refinement process** that progressively improves simulation configurations
```

**File:** README.md (L40-54)
```markdown
### 📐 **External Mesh File Support**
- **Custom mesh integration** with GMSH `.msh` files
- **Boundary condition specification** through natural language requirements
- **Currently supports** GMSH ASCII 2.2 format mesh files
- **Seamless workflow** from mesh import to simulation execution

**Example Usage:**
```bash
python foambench_main.py --output ./output --prompt_path ./user_requirement.txt --custom_mesh_path ./tandem_wing.msh
```

**Example Mesh File:** The `geometry.msh` file in this repository is taken from the [tandem wing tutorial](https://github.com/openfoamtutorials/tandem_wing) and demonstrates a 3D tandem wing simulation with NACA 0012 airfoils.

**Requirements Format:** In your `user_req_tandem_wing.txt`, describe the boundary conditions and physical parameters for your custom mesh. The agent will automatically detect the mesh type and generate appropriate OpenFOAM configuration files.

```

**File:** README.md (L70-76)
```markdown

Inside the container you automatically get:
- **OpenFOAM v10** installed and sourced
- **Conda** initialized and the `FoamAgent` environment activated
- **Working directory** set to `/home/openfoam/Foam-Agent`
- **Database files** pre-initialized and ready to use

```

**File:** README.md (L122-137)
```markdown
#### 1.5 Run a simulation inside Docker

From `/home/openfoam/Foam-Agent` in the container:

```bash
# Basic run
python foambench_main.py \
  --output ./output \
  --prompt_path ./user_requirement.txt

# With a custom mesh (if provided)
python foambench_main.py \
  --output ./output \
  --prompt_path ./user_requirement.txt \
  --custom_mesh_path ./my_mesh.msh
```
```

**File:** foambench_main.py (L63-85)
```python
def main():
    args = parse_args()
    print(args)

    # Check if OPENAI_API_KEY is available in the environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY is not set in the environment.")
        sys.exit(1)

    # Create the output folder
    os.makedirs(args.output, exist_ok=True)

    # Build main workflow command with optional custom mesh path
    main_cmd = f"python src/main.py --prompt_path='{args.prompt_path}' --output_dir='{args.output}'"
    if args.custom_mesh_path:
        main_cmd += f" --custom_mesh_path='{args.custom_mesh_path}'"
    
    print(f"Main workflow command: {main_cmd}")
    
    print("Starting workflow...")
    run_command(main_cmd)
    print("Workflow completed successfully.")
```

**File:** src/services/visualization.py (L1-8)
```python
import os
import sys
import subprocess
import glob
from typing import Dict, List, Tuple
from utils import save_file
from . import global_llm_service

```

**File:** src/services/visualization.py (L10-43)
```python
def ensure_foam_file(case_dir: str) -> str:
    """
    Ensure a .foam file exists in the case directory for OpenFOAM visualization.
    
    This function creates or updates a .foam file in the specified case directory.
    The .foam file is required for OpenFOAM visualization tools to recognize
    the directory as a valid OpenFOAM case.
    
    Args:
        case_dir (str): Directory path containing the OpenFOAM case
    
    Returns:
        str: Name of the .foam file (typically "{case_name}.foam")
    
    Raises:
        OSError: If directory cannot be accessed or file cannot be created
    
    Example:
        >>> foam_name = ensure_foam_file("/path/to/case")
        >>> print(f"Foam file: {foam_name}")  # "case.foam"
    """
    case_dir = os.path.abspath(case_dir)
    foam = f"{os.path.basename(case_dir)}.foam"
    foam_path = os.path.join(case_dir, foam)
    
    # Create or update the .foam file
    if not os.path.exists(foam_path):
        with open(foam_path, 'w') as f:
            pass
    else:
        # Update timestamp if file exists
        os.utime(foam_path, None)
    
    return foam
```

**File:** src/services/visualization.py (L46-92)
```python
def generate_pyvista_script(
    case_dir: str,
    foam_file: str,
    user_requirement: str,
    previous_errors: List[str]
) -> str:
    """
    Generate PyVista visualization script for OpenFOAM case using LLM.
    
    This function uses LLM to generate a Python script that uses PyVista
    to visualize OpenFOAM simulation results. The script loads the .foam file,
    renders geometry with appropriate coloring, and saves visualization images.
    
    Args:
        case_dir (str): Directory path containing the OpenFOAM case
        foam_file (str): Name of the .foam file for the case
        user_requirement (str): User requirements for visualization context
        previous_errors (List[str]): List of previous visualization errors for context
    
    Returns:
        str: Generated Python script code for PyVista visualization
    
    Raises:
        RuntimeError: If LLM service fails to generate script
    
    Example:
        >>> script = generate_pyvista_script(
        ...     case_dir="/path/to/case",
        ...     foam_file="case.foam",
        ...     user_requirement="Visualize velocity field",
        ...     previous_errors=[]
        ... )
        >>> print("Generated PyVista script")
    """
    system_prompt = (
        "You are an expert in OpenFOAM post-processing and PyVista Python scripting. "
        "Generate a PyVista script that loads the .foam file, renders geometry colored by requested field, uses coolwarm colormap, and saves a PNG. "
        "Return ONLY Python code, no markdown."
    )
    prompt = (
        f"<case_directory>{case_dir}</case_directory>\n"
        f"<foam_file>{foam_file}</foam_file>\n"
        f"<visualization_requirements>{user_requirement}</visualization_requirements>\n"
        f"<previous_errors>{previous_errors}</previous_errors>\n"
    )
    return global_llm_service.invoke(prompt, system_prompt)

```

**File:** src/services/plan.py (L26-78)
```python
def parse_requirement_to_case_info(user_requirement: str, case_stats: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Parse user requirements into structured case information using LLM.
    
    This function uses LLM to analyze natural language user requirements
    and extract structured case information including name, domain, category,
    and solver. The extracted values are validated against available options.
    
    Args:
        user_requirement (str): Natural language description of simulation requirements
        case_stats (Dict[str, List[str]]): Available case statistics with keys:
            - case_domain: List of available domains (e.g., ["fluid", "solid"])
            - case_category: List of available categories (e.g., ["tutorial", "advanced"])
            - case_solver: List of available solvers (e.g., ["simpleFoam", "pimpleFoam"])
    
    Returns:
        Dict[str, str]: Structured case information containing:
            - case_name (str): Parsed case name with spaces replaced by underscores
            - case_domain (str): Selected domain from available options
            - case_category (str): Selected category from available options
            - case_solver (str): Selected solver from available options
    
    Raises:
        ValueError: If LLM fails to parse requirements or returns invalid values
        RuntimeError: If LLM service is unavailable
    
    Example:
        >>> case_stats = {
        ...     "case_domain": ["fluid", "solid"],
        ...     "case_category": ["tutorial", "advanced"],
        ...     "case_solver": ["simpleFoam", "pimpleFoam"]
        ... }
        >>> result = parse_requirement_to_case_info(
        ...     "Create a simple fluid flow tutorial",
        ...     case_stats
        ... )
        >>> print(f"Case: {result['case_name']}, Solver: {result['case_solver']}")
    """
    parse_system_prompt = (
        "Please transform the following user requirement into a standard case description using a structured format."
        "The key elements should include case name, case domain, case category, and case solver."
        f"Note: case domain must be one of {case_stats.get('case_domain', [])}."
        f"Note: case category must be one of {case_stats.get('case_category', [])}."
        f"Note: case solver must be one of {case_stats.get('case_solver', [])}."
    )
    parse_user_prompt = f"User requirement: {user_requirement}."
    res = global_llm_service.invoke(parse_user_prompt, parse_system_prompt, pydantic_obj=CaseSummaryModel)
    return {
        "case_name": res.case_name.replace(" ", "_"),
        "case_domain": res.case_domain,
        "case_category": res.case_category,
        "case_solver": res.case_solver,
    }
```

