# Douglas AUV - Hydrodynamic System Identification Pipeline

This repository contains a System Identification (SysID) pipeline for the McGill Robotics Douglas AUV. It uses CasADi and IPOPT to solve a non-linear optimization problem to extract 6-DoF hydrodynamic parameters (added mass, linear drag, quadratic drag, and restoring forces) from ROS MCAP logs.

## Installation (uv)

This project uses [uv](https://github.com/astral-sh/uv).

1. **Install uv:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Setup Environment:**
   Because this pipeline interacts with ROS 2 native bindings (`rclpy`), you **must** create the virtual environment with the `--system-site-packages` flag.
   ```bash
   uv venv --system-site-packages
   uv sync
   ```
3. **Run Scripts:**
   Always prefix the script execution with `uv run` to ensure you are using the correct environment:
   ```bash
   uv run 03_hydrodynamic_sysid.py --dataset-dir processed_datasets
   ```

## Dependencies

The `pyproject.toml` automatically installs the following PyPI packages:

## Pipeline Architecture

The pipeline is split into 5 sequential steps:

### 1. `01_inspect_mcap.py`
Inspects `.mcap` files to verify that the required topics (`/odometry/filtered` and `/controller/wrench`) are present and overlapping.
```bash
python3 01_inspect_mcap.py --mcap-dir raw_mcaps
```

### 2. `02_extract_kinematics.py`
Extracts the raw data from the MCAP files, synchronizes the timestamps, calculates velocities and accelerations, applies Savitzky-Golay filtering for smoothness, and exports flat `.csv` files.
```bash
python3 02_extract_kinematics.py --mcap-dir raw_mcaps --output-dir processed_datasets
```

### 3. `03_hydrodynamic_sysid.py`
The core optimization engine. It loads the CSV datasets, aligns the Kinematics and Wrench arrays using automatic Cross-Correlation (to fix network delays), and uses CasADi/IPOPT to fit the Fossen 6-DoF equations of motion.
```bash
python3 03_hydrodynamic_sysid.py --dataset-dir processed_datasets --starts 3
```
*Note: You can use `--single-bag <file.csv>` to train and test strictly on an 80/20 temporal split of a single continuous bag.*

### 4. `04_validate_model.py`
Uses an Explicit Runge-Kutta 4 (ERK4) integrator to perform a forward simulation of the AUV using the identified parameters, validating them against the real dataset.

### 5. `05_estimate_hydrodynamics.py`
A fallback script that generates mathematically sound default parameters for the NMPC based on the AUV's physical mass and dimensions. Use this if SysID data is flawed.

---

## Operational Procedure: Poolside System Identification
**Do not use closed-loop competition mission logs for baseline System Identification.** To extract accurate, uncoupled matrices for Model Predictive Control (NMPC), you must perform dedicated open-loop data collection in a pool:

1. **Tether Management**: The physical drag of the tether will mathematically lump into the AUV's hydrodynamic drag, causing the NMPC to overestimate drag during an untethered competition run. Run the AUV untethered, or have a diver manually feed extreme slack so zero tension is applied during the maneuver.
2. **Dedicated Maneuvers**: Do not run standard waypoint following. Explicitly isolate the 6 axes:
   * **Step Responses**: Command pure open-loop forward thrust (e.g., 40N) for 3 seconds, then cut thrust entirely. Allow the vehicle to coast to a complete stop. This isolates Surge Drag and Surge Added Mass. Repeat for Sway and Heave.
   * **Orthogonal Multisines (Chirps)**: For rotational axes, command sinusoidal torque sweeps (e.g., oscillating Yaw back and forth) to excite rotational inertia.
3. **Handling Positive Buoyancy**: If the AUV is positively buoyant, it will float to the surface during X/Y step responses. Run a closed-loop PID depth-hold (Z-axis only) while sending open-loop commands to the X/Y axes. CasADi will mathematically isolate the Z-axis thrust and attribute it to buoyancy without ruining your Surge/Sway SysID.
4. **The Closed-Loop Problem (Identifiability)**: Data collected during closed-loop waypoint following is inherently flawed for SysID. The PID calculates thrust based on velocity error, making thrust and drag mathematically indistinguishable to the optimizer. Furthermore, PID saturation creates non-physical thrust logs. Always use open-loop commands.
5. **Closed-Loop Coasting Filter (`--coasting-only`)**: If you must extract parameters from closed-loop mission bags, append the `--coasting-only` flag to `03_hydrodynamic_sysid.py`. This drops all cruising data where thrust is saturated, restricting the optimizer to evaluate only the frames where the AUV is actively decelerating with near-zero thrust. By removing the corrupted thrust variable ($F_x \approx 0$), the optimizer can calculate drag purely from the deceleration profile.

## Physical Constants
All mass, volume, and geometry constraints are defined in `03_hydrodynamic_sysid.py`. Update `MASS = 27.9` if the AUV's dry weight changes next year.