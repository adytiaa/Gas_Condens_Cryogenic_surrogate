
# AI combined with CFD for BOG/BOR of LH2 storage in Composite Cryogenic Vessels (OpenFOAM)

**AI-based CFD workfow** combining OpenFOAM, LLM reasoning (OpenFOAM-GPT / MetaOpenFOAM 2.0).

## Capabilities
- Autonomous case generation
- Self-healing solver control
- Adaptive timestep & mesh reasoning
- Automatic BOG/BOR extraction & validation
- Experiment orchestration

# The detail aspects of the LH2 and Cryogenic Tanks
Below is a clear, technically detailed explanation you can adapt for a paper or proposal.

---

### Core idea of the multiphysics, multiphase CFD approach

The multiphysics, multiphase thermal CFD with focus on **all physical mechanisms governing boil-off in cryogenic liquid hydrogen (LH₂) storage tanks**, rather than relying on empirical models. The simulation typically couples:

1. **Multiphase flow physics**

   * Liquid LH₂ and gaseous hydrogen phases are explicitly modeled.
   * Interface dynamics, phase change (evaporation/condensation), and free-surface behavior are resolved using VOF or similar interface-capturing methods.
   * Bubble formation and vapor accumulation near the liquid–vapor interface are captured.

2. **Thermal transport**

   * Heat ingress through tank walls due to imperfect insulation (conduction and radiation).
   * Convective heat transfer within the liquid and vapor regions.
   * Local temperature gradients at the liquid–vapor interface that drive evaporation.

3. **Phase-change modeling**

   * Mass and energy transfer across the liquid–vapor interface is computed using thermodynamically consistent evaporation/condensation models.
   * Latent heat effects are included, allowing direct calculation of boil-off gas (BOG) generation.

4. **Fluid dynamics**

   * Natural convection in the liquid due to thermal stratification.
   * Vapor circulation and pressure evolution in the ullage.
   * Coupling between pressure rise and saturation temperature.

5. **Real-fluid thermophysical properties**

   * Temperature- and pressure-dependent LH₂ properties (density, viscosity, thermal conductivity).
   * Accurate hydrogen equations of state near saturation.

By resolving these interacting processes, the CFD model can **directly compute boil-off rate (BOR)** as an emergent quantity rather than prescribing it.

---

### Why this matters for LH₂ boil-off prediction

LH₂ is extremely sensitive to even small heat leaks due to its:

* Very low boiling temperature (~20 K),
* Low latent heat per unit volume,
* Strong density variations near saturation.

Simplified models often underestimate:

* Local hot spots at the tank wall,
* Stratification-driven convection,
* Transient pressure–temperature coupling.

Multiphysics CFD overcomes these limitations and has been shown to reproduce **experimental BOR and pressure rise trends**, validating its predictive capability.

---

### Relationship to “zero boil-off” (ZBO) LH₂ storage

Efforts to achieve **zero boil-off (ZBO)** aim to eliminate or recondense vapor so that no hydrogen is vented. The CFD framework directly supports ZBO development in several ways:

#### 1. Quantifying unavoidable heat ingress

Even with advanced insulation, some heat leak is inevitable. CFD:

* Identifies dominant heat transfer paths,
* Quantifies spatially resolved heat fluxes,
* Determines the *minimum cooling power* required for ZBO.

This is essential for sizing cryocoolers or active refrigeration systems.

#### 2. Optimizing active thermal control systems

For ZBO concepts such as:

* Integrated cryocoolers,
* Subcooled liquid injection,
* Thermodynamic vent systems (TVS),
* Internal heat exchangers,

CFD can simulate:

* Local condensation of BOG,
* Mixing efficiency of recondensed liquid,
* Impact on stratification and pressure stability.

#### 3. Reducing BOR through passive design

CFD helps refine passive strategies that *minimize* boil-off before active control is applied:

* Tank geometry and aspect ratio,
* Placement and thickness of insulation,
* Internal baffles or liners to suppress convection,
* Wall material selection.

Lower baseline BOR directly reduces ZBO system complexity and power demand.

#### 4. Transient and mission-specific analysis

ZBO performance is highly transient:

* Filling and chill-down,
* Long-duration storage,
* Variable environmental heat loads,
* Sloshing or microgravity effects (for space applications).

Multiphysics CFD enables time-resolved analysis of these scenarios, which is critical for **spaceflight LH₂ tanks and long-term terrestrial storage**.

---

### Overall significance

In summary, multiphysics, multiphase thermal CFD:


* Provides a **physics-based, validated tool** for predicting BOG and BOR,
* Bridges the gap between experimental testing and full-scale tank design,
* Enables rational design and optimization of **zero boil-off LH₂ storage systems**.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---------------------------------------------------
