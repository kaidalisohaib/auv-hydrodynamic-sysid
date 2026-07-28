#!/usr/bin/env python3
"""
05_estimate_hydrodynamics.py

Generates safe, mathematically sound baseline hydrodynamic parameters
for a ~28kg open-frame AUV. Use this to initialize your NMPC matrices
when system identification data is unavailable or flawed.

Approximations based on Fossen (2011) and BlueROV2 Heavy empirical data.
"""
import json
import os

def estimate_parameters():
    # Known constraints
    mass = 27.9          # Dry mass [kg]
    weight = mass * 9.81 # [N]
    buoyancy = weight + 7.0 # Slight positive buoyancy [N]
    
    # -- Added Mass (Approximations) --
    # Surge: ~0.5x mass. Sway: ~0.8x mass. Heave: ~1.2x mass (due to flat top/bottom profile)
    added_mass_trans = [15.0, 22.0, 33.0]
    
    # Rotational Added Mass: roughly 0.5 - 1.5 kg*m^2 for this size
    added_mass_rot = [1.0, 1.5, 1.5]
    
    # -- Linear Drag --
    # Open-frame AUVs have some linear drag at low speeds
    linear_drag_trans = [5.0, 10.0, 15.0]
    linear_drag_rot = [0.5, 1.0, 1.0]
    
    # -- Quadratic Drag --
    # Formula: D = 0.5 * rho * Cd * Area
    # Assuming rho = 1000, Cd = 1.0 (bluff body)
    # Estimated Frontal Areas: X=0.08m^2, Y=0.15m^2, Z=0.25m^2
    quad_drag_trans = [40.0, 75.0, 125.0]
    quad_drag_rot = [5.0, 10.0, 10.0]
    
    params = {
        "mass": mass,
        "weight": weight,
        "buoyancy": buoyancy,
        "CoM": [0.0, 0.0, -0.01],
        "CoB": [0.0, 0.0, -0.02],
        "added_mass": {
            "translational": added_mass_trans,
            "rotational": added_mass_rot
        },
        "linear_drag": {
            "translational": linear_drag_trans,
            "rotational": linear_drag_rot
        },
        "quadratic_drag": {
            "translational": quad_drag_trans,
            "rotational": quad_drag_rot
        }
    }
    
    os.makedirs('sysid_results', exist_ok=True)
    with open('sysid_results/estimated_parameters_fallback.json', 'w') as f:
        json.dump(params, f, indent=4)
        
    print("==============================================================")
    print("  Douglas AUV - Fallback Hydrodynamic Estimates")
    print("==============================================================")
    print("Run this script to generate safe baseline parameters for your MPC.")
    print(f"Exported to: sysid_results/estimated_parameters_fallback.json")
    print("Use these values as your starting point for poolside tuning.")

if __name__ == "__main__":
    estimate_parameters()
