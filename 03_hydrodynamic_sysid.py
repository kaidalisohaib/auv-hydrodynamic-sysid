#!/usr/bin/env python3
"""
03_hydrodynamic_sysid.py - Hydrodynamic System Identification
McGill Robotics | douglas-sysid pipeline

Identifies hydrodynamic parameters for Douglas AUV using CasADi/IPOPT
nonlinear optimization with Pseudo-Huber robust loss and Weighted Least
Squares normalization.

Physics Model (6-DoF Fossen Equations):
    τ = M·v̇ + C_RB(v)·v + D(v)·v + g(η)

Where:
    M     = M_RB + M_A (rigid body + added mass, including off-diagonal cross-coupling)
    C_RB  = rigid body Coriolis/centripetal matrix
    D(v)  = D_linear·v + D_quadratic·v|v| (skin friction + form drag)
    g(η)  = quaternion-based restoring forces (gravity, buoyancy, tether)

Identified Parameters (32 total):
    - Buoyancy force, tether pull                              (2)
    - Center of mass, center of buoyancy                       (6)
    - Diagonal added mass (6-DoF)                              (6)
    - Diagonal linear drag (6-DoF)                             (6)
    - Diagonal quadratic drag (6-DoF)                          (6)
    - Cross-coupling added mass: X_q̇=M_u̇, Y_ṙ=N_v̇ (symmetric) (2)
    - Cross-coupling linear drag: X_q, Y_r, M_w, N_v          (4)

Features:
    - Singularity-free quaternion hydrostatics (supports 360° rotation/flips)
    - Train/test split validation with per-axis R²/NRMSE/MAE metrics
    - Multi-start optimization to escape IPOPT local minima
    - Residual diagnostic plots + JSON parameter export
    - Backward-compatible with Euler-angle CSV datasets

Usage:
    python 03_hydrodynamic_sysid.py
    python 03_hydrodynamic_sysid.py --test-bag processed_datasets/finals_run_dataset.csv
    python 03_hydrodynamic_sysid.py --dataset-dir processed_datasets --starts 1

Flags:
    --coasting-only: CRITICAL for closed-loop competition datasets. 
                     Applies a boolean mask to drop saturated PID thrust frames.
                     Forces the optimizer to solve for drag purely from deceleration.
    --starts N:      Run N independent multi-start optimization passes (default: 3).
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import casadi as ca
from scipy.signal import savgol_filter, correlate, correlation_lags
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt


# ============================================================================
# PHYSICAL CONSTANTS - Douglas AUV (McGill Robotics)
# ============================================================================
MASS = 27.9              # Dry mass [kg]
WEIGHT = MASS * 9.81     # Gravitational force [N]

# Moments of Inertia from Onshape CAD (converted from lb·in² to kg·m²)
I_XX = 796.517 * 0.00064516    # Roll inertia  [kg·m²]
I_YY = 818.679 * 0.00064516    # Pitch inertia [kg·m²]
I_ZZ = 1250.459 * 0.00064516   # Yaw inertia   [kg·m²]


# ============================================================================
# OPTIMIZATION BOUNDS & SCALING
# ============================================================================
# --- Buoyancy ---
BUOYANCY_MIN_RATIO  = 1.001    # Must be positively buoyant
BUOYANCY_MAX_RATIO  = 1.05     # Max 5% positive buoyancy
BUOYANCY_INIT_RATIO = 1.01     # Initial guess: 1% positive buoyancy

# --- Center of Mass / Buoyancy offsets ---
COM_SHIFT_LIMIT = 0.04         # ±4 cm from body origin [m]
COB_SHIFT_LIMIT = 0.04
COB_INIT_Z      = -0.02        # CoB ~2 cm above CoM in FLU (negative Z = up)

# --- Tether ---
W_TETHER_MAX = 7.0             # Max static tether pull [N] (~700g in water)

# --- Data Conditioning ---
# Wrench smoothing (match velocity Savgol filter from Step 02)
WRENCH_SAVGOL_WINDOW  = 51     # Slightly wider than Step 02 to suppress PID transients
WRENCH_SAVGOL_POLY    = 3      # Must match 02_extract_kinematics.py POLY_ORDER
# Outlier clipping (removes controller saturation artifacts)
MAX_FORCE_CLIP   = 40.0        # [N] clip forces beyond this magnitude
MAX_TORQUE_CLIP  = 5.0         # [N·m] clip torques beyond this magnitude
# Minimum speed filter (remove near-static frames with no drag information)
MIN_SPEED_THRESHOLD = 0.02     # [m/s] total body speed

# --- Scaling factors: optimizer variable → physical units ---
# These keep all optimizer variables near unity for numerical conditioning.
SCALE_MA_TRANS    = MASS * 0.5  # ~13.4 kg
SCALE_MA_ROT      = 0.5         # kg·m²
SCALE_DQUAD_TRANS = 25.0        # N·s²/m²
SCALE_DQUAD_ROT   = 5.0         # N·m·s²/rad²
SCALE_DLIN_TRANS  = 10.0        # N·s/m
SCALE_DLIN_ROT    = 2.0         # N·m·s/rad

# --- Cross-coupling bounds (physical units, allow positive AND negative) ---
MA_CROSS_LIMIT   = 20.0        # Added mass cross-coupling [kg·m]
DRAG_CROSS_LIMIT = 30.0        # Drag cross-coupling [N·s/rad or N·s]

# --- L2 Regularization weights ---
LAMBDA_COM         = 200.0     # Pull CoM toward body origin
LAMBDA_COB         = 200.0     # Pull CoB toward initial guess
LAMBDA_YAW         = 100.0     # Yaw/sway drag ratio (physical prior)
LAMBDA_CROSS_MA    = 50.0      # Drive cross-coupling added mass → 0 unless data demands
LAMBDA_CROSS_DRAG  = 50.0      # Drive cross-coupling drag → 0 unless data demands

# --- Multi-start initial guess profiles (optimizer-scale values) ---
INITIAL_GUESSES = [
    {
        'name': 'Default (balanced)',
        'MA': 1.0, 'Dquad': 1.0, 'Dlin': 1.0,
    },
    {
        'name': 'High drag / Low added mass',
        'MA': 0.3, 'Dquad': 2.0, 'Dlin': 2.0,
    },
    {
        'name': 'Low drag / High added mass',
        'MA': 2.5, 'Dquad': 0.3, 'Dlin': 0.3,
    },
]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_datasets(directory, test_bag=None, single_bag=None, coasting_only=False):
    """Load CSV datasets with train/test split.

    Splitting strategy:
        - If single_bag is specified, load that one file and do an 80/20
          temporal split (first 80% = train, last 20% = test).
        - If test_bag is specified, that file is held out entirely.
        - Otherwise, uses a deterministic bag-level split: hold out 1 flips,
          2 slalom, and 2 controls-testing bags. Falls back to last ~20% of
          bags if category diversity is insufficient.

    Returns:
        (df_train, df_test, train_files, test_files)
    """
    def _align_time_series(df):
        """Automatically fix time sync delays using cross-correlation."""
        dt = 0.02
        best_lag = 0
        best_corr = 0
        
        # Test the 3 primary linear axes for the strongest correlation signal
        for vel, force in [('u', 'Fx'), ('v', 'Fy'), ('w', 'Fz')]:
            x = df[vel].values
            y = df[force].values
            if len(x) == 0 or np.std(x) < 1e-4 or np.std(y) < 1e-4:
                continue
            
            correlation = correlate(x - np.mean(x), y - np.mean(y), mode='full')
            lags = correlation_lags(len(x), len(y), mode='full')
            lag_idx = np.argmax(correlation)
            
            # Normalize to get Pearson correlation coefficient
            corr = correlation[lag_idx] / (np.std(x) * np.std(y) * len(x))
            
            # Use the axis with the absolutely strongest alignment signal
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lags[lag_idx]
                
        if best_lag != 0:
            print(f"    Auto-align: shifting wrench by {best_lag*dt:+.3f}s (max corr: {best_corr:.3f})")
            wrench_cols = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
            for col in wrench_cols:
                if col in df.columns:
                    df[col] = df[col].shift(-best_lag)
            
        return df.dropna().reset_index(drop=True)

    def _extract_coasting_windows(df):
        """Extract only segments where the AUV is actively decelerating."""
        dt = 0.02
        u_dot = np.gradient(df['u'].values, dt)
        v_dot = np.gradient(df['v'].values, dt)
        w_dot = np.gradient(df['w'].values, dt)
        
        # Surge coasting: velocity > 0.1 and opposing acceleration > 0.02, thrust < 5N
        cond_u_fwd = (df['u'] > 0.1) & (u_dot < -0.02) & (df['Fx'].abs() < 5.0)
        cond_u_bwd = (df['u'] < -0.1) & (u_dot > 0.02) & (df['Fx'].abs() < 5.0)
        
        # Sway coasting
        cond_v_pos = (df['v'] > 0.1) & (v_dot < -0.02) & (df['Fy'].abs() < 5.0)
        cond_v_neg = (df['v'] < -0.1) & (v_dot > 0.02) & (df['Fy'].abs() < 5.0)
        
        # Heave coasting
        cond_w_pos = (df['w'] > 0.1) & (w_dot < -0.02) & (df['Fz'].abs() < 5.0)
        cond_w_neg = (df['w'] < -0.1) & (w_dot > 0.02) & (df['Fz'].abs() < 5.0)
        
        mask = cond_u_fwd | cond_u_bwd | cond_v_pos | cond_v_neg | cond_w_pos | cond_w_neg
        
        df_coasting = df[mask].reset_index(drop=True)
        print(f"    Coasting Mask: Filtered {len(df)} frames down to {len(df_coasting)} coasting frames.")
        return df_coasting

    # -- Single-bag temporal split mode --
    if single_bag:
        if not os.path.exists(single_bag):
            raise ValueError(f"Single bag '{single_bag}' not found.")

        print(f"Single-bag mode: {os.path.basename(single_bag)}")
        df = pd.read_csv(single_bag, engine='pyarrow')
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        df = _align_time_series(df)
        df = _condition_data(df)
        if coasting_only:
            df = _extract_coasting_windows(df)

        if len(df) == 0:
            raise ValueError("No frames left after conditioning/coasting filter!")

        split_idx = int(len(df) * 0.8)
        df_train = df.iloc[:split_idx].reset_index(drop=True)
        df_test  = df.iloc[split_idx:].reset_index(drop=True)

        duration = df['time'].iloc[-1] - df['time'].iloc[0]
        train_dur = df_train['time'].iloc[-1] - df_train['time'].iloc[0]
        test_dur  = df_test['time'].iloc[-1] - df_test['time'].iloc[0]

        print(f"  Total: {len(df):,} frames ({duration:.0f}s)")
        print(f"  Train: {len(df_train):,} frames ({train_dur:.0f}s, first 80%)")
        print(f"  Test:  {len(df_test):,} frames ({test_dur:.0f}s, last 20%)")
        print(f"\nTraining frames: {len(df_train):,}  |  Test frames: {len(df_test):,}")
        return df_train, df_test, [single_bag], [single_bag]

    csv_files = sorted(glob.glob(os.path.join(directory, "*_dataset.csv")))
    csv_files = [f for f in csv_files if "combined_master" not in os.path.basename(f)]

    if not csv_files:
        raise ValueError(f"No *_dataset.csv files found in '{directory}'")

    print(f"Found {len(csv_files)} dataset files in '{directory}'.")

    if test_bag:
        # -- User-specified test bag --
        test_bag_abs = os.path.abspath(test_bag)
        test_files = [f for f in csv_files if os.path.abspath(f) == test_bag_abs]
        if not test_files:
            # Try basename match
            test_files = [f for f in csv_files
                          if os.path.basename(f) == os.path.basename(test_bag)]
        if not test_files:
            # Try loading directly (might not be in the dataset dir)
            if os.path.exists(test_bag):
                test_files = [test_bag]
            else:
                raise ValueError(f"Test bag '{test_bag}' not found.")
        train_files = [f for f in csv_files if f not in test_files]
    else:
        # -- Automatic bag-level 80/20 split --
        # Categorize by bag name prefix for stratified splitting
        flips    = [f for f in csv_files if 'flips' in os.path.basename(f)]
        slalom   = [f for f in csv_files if 'slalom' in os.path.basename(f)]
        controls = [f for f in csv_files if 'controls-testing' in os.path.basename(f)]

        test_files = []
        if len(flips) >= 2:
            test_files.append(flips[-1])       # Hold out last flips bag
        if len(slalom) >= 3:
            test_files.extend(slalom[-2:])     # Hold out last 2 slalom bags
        if len(controls) >= 3:
            test_files.extend(controls[-2:])   # Hold out last 2 controls bags

        # Fallback: if not enough diversity, take last ~20%
        if len(test_files) < 2:
            n_test = max(1, len(csv_files) // 5)
            test_files = csv_files[-n_test:]

        train_files = [f for f in csv_files if f not in test_files]

    if not train_files:
        raise ValueError("No training files remain after split.")

    # -- Load and concatenate (condition per-bag to avoid boundary artifacts) --
    def _load_concat(file_list, label):
        dfs = []
        for f in file_list:
            try:
                df = pd.read_csv(f, engine='pyarrow')
                df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
                df = _align_time_series(df)
                if len(df) == 0:
                    continue
                df = _condition_data(df)
                if coasting_only:
                    df = _extract_coasting_windows(df)
                if len(df) > 0:
                    dfs.append(df)
            except Exception as e:
                print(f"  Warning: Could not load {os.path.basename(f)}: {e}")
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        return combined

    print(f"\nTrain set ({len(train_files)} bags):")
    for f in train_files:
        print(f"  • {os.path.basename(f)}")
    print(f"\nTest set ({len(test_files)} bags):")
    for f in test_files:
        print(f"  • {os.path.basename(f)}")

    df_train = _load_concat(train_files, "train")
    df_test = _load_concat(test_files, "test")

    # Downsample training set if necessary to prevent CasADi memory explosion
    max_frames = 250_000
    if len(df_train) > max_frames:
        step = int(np.ceil(len(df_train) / max_frames))
        print(f"\nDownsampling training set: {len(df_train)} → ~{len(df_train)//step} "
              f"frames (factor {step})")
        df_train = df_train.iloc[::step].reset_index(drop=True)

    print(f"\nTraining frames: {len(df_train):,}  |  Test frames: {len(df_test):,}")
    return df_train, df_test, train_files, test_files


def _ensure_quaternion_columns(df):
    """Ensure quaternion columns exist; convert from Euler angles if needed."""
    quat_cols = ['qx', 'qy', 'qz', 'qw']

    if all(c in df.columns for c in quat_cols):
        return df

    # Backward compatibility: convert from roll/pitch (assuming yaw=0)
    if 'roll' in df.columns and 'pitch' in df.columns:
        print("  ! Converting Euler angles → quaternions (yaw=0 assumed).")
        print("    Re-run 02_extract_kinematics.py for native quaternion output.")
        phi = df['roll'].values
        theta = df['pitch'].values
        cp, sp = np.cos(phi / 2), np.sin(phi / 2)
        ct, st = np.cos(theta / 2), np.sin(theta / 2)
        df['qx'] = sp * ct
        df['qy'] = cp * st
        df['qz'] = -sp * st
        df['qw'] = cp * ct
        return df

    raise ValueError(
        "Dataset missing quaternion columns (qx, qy, qz, qw) and Euler angles "
        "(roll, pitch). Re-run 02_extract_kinematics.py with the updated script."
    )


def _condition_data(df):
    """Apply data conditioning: smooth wrench, clip outliers, filter static frames.

    1. Applies Savitzky-Golay filter to wrench to match velocity phase lag.
    2. Clips controller saturation outliers.
    3. Removes near-static frames (speed < MIN_SPEED_THRESHOLD).
    """
    n_before = len(df)

    # 1. Smooth wrench to match velocity filtering
    wrench_cols = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    if len(df) > WRENCH_SAVGOL_WINDOW:
        for col in wrench_cols:
            if col in df.columns:
                df[col] = savgol_filter(df[col], WRENCH_SAVGOL_WINDOW,
                                        WRENCH_SAVGOL_POLY)

    # 2. Clip extreme wrench values (controller saturation artifacts)
    for col in ['Fx', 'Fy', 'Fz']:
        if col in df.columns:
            df[col] = df[col].clip(-MAX_FORCE_CLIP, MAX_FORCE_CLIP)
    for col in ['Tx', 'Ty', 'Tz']:
        if col in df.columns:
            df[col] = df[col].clip(-MAX_TORQUE_CLIP, MAX_TORQUE_CLIP)

    # 3. Remove near-static frames (no drag/inertia information)
    if all(c in df.columns for c in ['u', 'v', 'w']):
        speed = np.sqrt(df['u']**2 + df['v']**2 + df['w']**2)
        df = df[speed > MIN_SPEED_THRESHOLD].reset_index(drop=True)

    n_after = len(df)
    if n_before != n_after:
        print(f"    Data conditioning: {n_before} → {n_after} frames "
              f"({n_before - n_after} static frames removed)")

    return df


def extract_data_arrays(df):
    """Extract numpy arrays from DataFrame for CasADi optimization."""
    df = _ensure_quaternion_columns(df)
    return {
        'u': df['u'].values,     'v': df['v'].values,     'w': df['w'].values,
        'p': df['p'].values,     'q': df['q'].values,     'r': df['r'].values,
        'u_dot': df['u_dot'].values, 'v_dot': df['v_dot'].values, 'w_dot': df['w_dot'].values,
        'p_dot': df['p_dot'].values, 'q_dot': df['q_dot'].values, 'r_dot': df['r_dot'].values,
        'Fx': df['Fx'].values, 'Fy': df['Fy'].values, 'Fz': df['Fz'].values,
        'Tx': df['Tx'].values, 'Ty': df['Ty'].values, 'Tz': df['Tz'].values,
        'qx': df['qx'].values, 'qy': df['qy'].values,
        'qz': df['qz'].values, 'qw': df['qw'].values,
    }


# ============================================================================
# OPTIMIZATION CORE
# ============================================================================

def build_and_solve(data, initial_guess, verbose=True):
    """Build CasADi Opti problem and solve with given initial guess.

    The problem is built from scratch each call to guarantee independence
    between multi-start runs (no warm-start contamination of dual variables).

    Args:
        data: dict of numpy arrays from extract_data_arrays()
        initial_guess: dict with keys 'MA', 'Dquad', 'Dlin' (optimizer-scale values)
        verbose: if True, print IPOPT iteration log

    Returns:
        dict with all identified parameter values and solver cost, or None if failed.
    """
    d = data

    # WLS normalization: scale each axis error by its peak magnitude
    max_F = [np.nanmax(np.abs(d[k])) + 1e-3 for k in ('Fx', 'Fy', 'Fz')]
    max_T = [np.nanmax(np.abs(d[k])) + 1e-3 for k in ('Tx', 'Ty', 'Tz')]

    opti = ca.Opti()

    # ------------------------------------------------------
    # DECISION VARIABLES (32 total)
    # ------------------------------------------------------

    # 1. Buoyancy & Tether (2 scalars)
    Buoyancy = opti.variable()
    W_tether = opti.variable()
    opti.subject_to(opti.bounded(WEIGHT * BUOYANCY_MIN_RATIO, Buoyancy,
                                  WEIGHT * BUOYANCY_MAX_RATIO))
    opti.subject_to(opti.bounded(0.0, W_tether, W_TETHER_MAX))
    opti.set_initial(Buoyancy, WEIGHT * BUOYANCY_INIT_RATIO)
    opti.set_initial(W_tether, 2.0)

    # 2. Centers of Mass & Buoyancy (6 components)
    CoM = opti.variable(3)
    CoB = opti.variable(3)
    opti.subject_to(opti.bounded(-COM_SHIFT_LIMIT, CoM, COM_SHIFT_LIMIT))
    opti.subject_to(opti.bounded(-COB_SHIFT_LIMIT, CoB, COB_SHIFT_LIMIT))
    opti.set_initial(CoM, [0, 0, 0])
    opti.set_initial(CoB, [0, 0, COB_INIT_Z])

    # 3. Diagonal Added Mass (6, scaled)
    MA_opt = opti.variable(6)
    opti.subject_to(MA_opt >= 0.01)
    opti.set_initial(MA_opt, initial_guess['MA'])
    MA_phys = ca.vertcat(
        MA_opt[:3] * SCALE_MA_TRANS,
        MA_opt[3:] * SCALE_MA_ROT
    )

    # 4. Diagonal Quadratic Drag (6, scaled)
    Dquad_opt = opti.variable(6)
    opti.subject_to(Dquad_opt >= 0.01)
    opti.set_initial(Dquad_opt, initial_guess['Dquad'])
    Dquad_phys = ca.vertcat(
        Dquad_opt[:3] * SCALE_DQUAD_TRANS,
        Dquad_opt[3:] * SCALE_DQUAD_ROT
    )

    # 5. Diagonal Linear Drag (6, scaled)
    Dlin_opt = opti.variable(6)
    opti.subject_to(Dlin_opt >= 0.01)
    opti.set_initial(Dlin_opt, initial_guess['Dlin'])
    Dlin_phys = ca.vertcat(
        Dlin_opt[:3] * SCALE_DLIN_TRANS,
        Dlin_opt[3:] * SCALE_DLIN_ROT
    )

    # 6. Cross-Coupling Added Mass (2 variables, symmetric M_A)
    #    X_q̇ = M_u̇  (surge ↔ pitch)
    #    Y_ṙ = N_v̇  (sway  ↔ yaw)
    ma_surge_pitch = opti.variable()
    ma_sway_yaw    = opti.variable()
    opti.subject_to(opti.bounded(-MA_CROSS_LIMIT, ma_surge_pitch, MA_CROSS_LIMIT))
    opti.subject_to(opti.bounded(-MA_CROSS_LIMIT, ma_sway_yaw, MA_CROSS_LIMIT))
    opti.set_initial(ma_surge_pitch, 0.0)
    opti.set_initial(ma_sway_yaw, 0.0)

    # 7. Cross-Coupling Linear Drag (4 variables, NOT symmetric)
    cc_Xq = opti.variable()   # Surge force ← pitch rate     [N·s/rad]
    cc_Yr = opti.variable()   # Sway force  ← yaw rate       [N·s/rad]
    cc_Mw = opti.variable()   # Pitch torque ← heave velocity [N·m·s/m]
    cc_Nv = opti.variable()   # Yaw torque   ← sway velocity  [N·m·s/m]
    for var in [cc_Xq, cc_Yr, cc_Mw, cc_Nv]:
        opti.subject_to(opti.bounded(-DRAG_CROSS_LIMIT, var, DRAG_CROSS_LIMIT))
        opti.set_initial(var, 0.0)

    # ------------------------------------------------------
    # QUATERNION-BASED RESTORING FORCES (singularity-free)
    # ------------------------------------------------------
    qx, qy, qz, qw = d['qx'], d['qy'], d['qz'], d['qw']

    # World UP vector [0, 0, 1] rotated into body frame via quaternion
    Z_body_x = 2 * (qx * qz - qw * qy)
    Z_body_y = 2 * (qy * qz + qw * qx)
    Z_body_z = qw**2 - qx**2 - qy**2 + qz**2

    # Gravity force in body frame (points along -Z_world_in_body)
    Fg_x = -WEIGHT * Z_body_x
    Fg_y = -WEIGHT * Z_body_y
    Fg_z = -WEIGHT * Z_body_z

    # Buoyancy force in body frame (points along +Z_world_in_body)
    Fb_x = Buoyancy * Z_body_x
    Fb_y = Buoyancy * Z_body_y
    Fb_z = Buoyancy * Z_body_z

    # Tether pull (assumed vertical downward in world → along -Z_world_in_body)
    Ft_x = -W_tether * Z_body_x
    Ft_y = -W_tether * Z_body_y
    Ft_z = -W_tether * Z_body_z

    # Net restoring force in body frame
    Rest_Fx = Fg_x + Fb_x + Ft_x
    Rest_Fy = Fg_y + Fb_y + Ft_y
    Rest_Fz = Fg_z + Fb_z + Ft_z

    # Restoring moments: τ = r × F (component-wise for CasADi broadcasting)
    # CoM × F_gravity
    Mom_gx = CoM[1] * Fg_z - CoM[2] * Fg_y
    Mom_gy = CoM[2] * Fg_x - CoM[0] * Fg_z
    Mom_gz = CoM[0] * Fg_y - CoM[1] * Fg_x

    # CoB × F_buoyancy
    Mom_bx = CoB[1] * Fb_z - CoB[2] * Fb_y
    Mom_by = CoB[2] * Fb_x - CoB[0] * Fb_z
    Mom_bz = CoB[0] * Fb_y - CoB[1] * Fb_x

    Rest_Tx = Mom_gx + Mom_bx
    Rest_Ty = Mom_gy + Mom_by
    Rest_Tz = Mom_gz + Mom_bz

    # ------------------------------------------------------
    # 6-DoF FOSSEN EQUATIONS OF MOTION
    # τ = M·v̇ + C_RB(v)·v + D(v)·v + g(η)
    # ------------------------------------------------------
    u, v, w = d['u'], d['v'], d['w']
    p, q, r = d['p'], d['q'], d['r']
    u_dot, v_dot, w_dot = d['u_dot'], d['v_dot'], d['w_dot']
    p_dot, q_dot, r_dot = d['p_dot'], d['q_dot'], d['r_dot']

    # Surge: cross-coupling from pitch (added mass M_A[0,4] and drag X_q)
    Fx_pred = ((MASS + MA_phys[0]) * u_dot + ma_surge_pitch * q_dot
               - MASS * (v * r - w * q)
               + Dquad_phys[0] * u * ca.fabs(u) + Dlin_phys[0] * u
               + cc_Xq * q
               - Rest_Fx)

    # Sway: cross-coupling from yaw (added mass M_A[1,5] and drag Y_r)
    Fy_pred = ((MASS + MA_phys[1]) * v_dot + ma_sway_yaw * r_dot
               - MASS * (w * p - u * r)
               + Dquad_phys[1] * v * ca.fabs(v) + Dlin_phys[1] * v
               + cc_Yr * r
               - Rest_Fy)

    # Heave: no cross-coupling
    Fz_pred = ((MASS + MA_phys[2]) * w_dot
               - MASS * (u * q - v * p)
               + Dquad_phys[2] * w * ca.fabs(w) + Dlin_phys[2] * w
               - Rest_Fz)

    # Roll: no cross-coupling (passive metacentric stability)
    Tx_pred = ((I_XX + MA_phys[3]) * p_dot
               - (I_YY - I_ZZ) * q * r
               + Dquad_phys[3] * p * ca.fabs(p) + Dlin_phys[3] * p
               - Rest_Tx)

    # Pitch: cross-coupling from surge (symmetric M_A[4,0] = M_A[0,4]) and heave drag M_w
    Ty_pred = (ma_surge_pitch * u_dot + (I_YY + MA_phys[4]) * q_dot
               - (I_ZZ - I_XX) * r * p
               + Dquad_phys[4] * q * ca.fabs(q) + Dlin_phys[4] * q
               + cc_Mw * w
               - Rest_Ty)

    # Yaw: cross-coupling from sway (symmetric M_A[5,1] = M_A[1,5]) and sway drag N_v
    Tz_pred = (ma_sway_yaw * v_dot + (I_ZZ + MA_phys[5]) * r_dot
               - (I_XX - I_YY) * p * q
               + Dquad_phys[5] * r * ca.fabs(r) + Dlin_phys[5] * r
               + cc_Nv * v
               - Rest_Tz)

    # ------------------------------------------------------
    # COST FUNCTION
    # ------------------------------------------------------
    delta = 1.0

    def huber(err):
        """Pseudo-Huber loss: smooth approximation to L1 near zero, L2 far."""
        return (delta**2) * (ca.sqrt(1.0 + (err / delta)**2) - 1.0)

    # Weighted least squares: normalize each axis by its peak amplitude
    wls_error = (ca.sum1(huber((d['Fx'] - Fx_pred) / max_F[0])) +
                 ca.sum1(huber((d['Fy'] - Fy_pred) / max_F[1])) +
                 ca.sum1(huber((d['Fz'] - Fz_pred) / max_F[2])) +
                 ca.sum1(huber((d['Tx'] - Tx_pred) / max_T[0])) +
                 ca.sum1(huber((d['Ty'] - Ty_pred) / max_T[1])) +
                 ca.sum1(huber((d['Tz'] - Tz_pred) / max_T[2])))

    # L2 regularization penalties
    penalty_com  = LAMBDA_COM * ca.sumsqr(CoM)
    penalty_cob  = LAMBDA_COB * ca.sumsqr(CoB - ca.DM([0, 0, COB_INIT_Z]))
    penalty_yaw  = LAMBDA_YAW * (Dquad_phys[5] - 0.3 * Dquad_phys[1])**2
    penalty_cross = (LAMBDA_CROSS_MA * (ma_surge_pitch**2 + ma_sway_yaw**2) +
                     LAMBDA_CROSS_DRAG * (cc_Xq**2 + cc_Yr**2 + cc_Mw**2 + cc_Nv**2))

    total_cost = wls_error + penalty_com + penalty_cob + penalty_yaw + penalty_cross
    opti.minimize(total_cost)

    # ------------------------------------------------------
    # SOLVER CONFIGURATION
    # ------------------------------------------------------
    p_opts = {"expand": True}
    s_opts = {
        "max_iter": 3000,
        "tol": 1e-6,
        "print_level": 5 if verbose else 0,
        "sb": "no" if verbose else "yes",
    }
    opti.solver("ipopt", p_opts, s_opts)

    # ------------------------------------------------------
    # SOLVE
    # ------------------------------------------------------
    try:
        sol = opti.solve()
        get_val = sol.value
        converged = True
    except Exception as e:
        if verbose:
            print(f"  Solver stopped early: {e}")
        get_val = opti.debug.value
        converged = False

    # Extract all parameter values into a portable dict
    result = {
        'converged': converged,
        'cost': float(get_val(total_cost)),
        'wls_error': float(get_val(wls_error)),
        'buoyancy': float(get_val(Buoyancy)),
        'tether_pull': float(get_val(W_tether)),
        'center_of_mass': np.array(get_val(CoM)).flatten().tolist(),
        'center_of_buoyancy': np.array(get_val(CoB)).flatten().tolist(),
        'added_mass_diagonal': np.array(get_val(MA_phys)).flatten().tolist(),
        'linear_drag_diagonal': np.array(get_val(Dlin_phys)).flatten().tolist(),
        'quadratic_drag_diagonal': np.array(get_val(Dquad_phys)).flatten().tolist(),
        'cross_coupling_added_mass': {
            'surge_pitch': float(get_val(ma_surge_pitch)),  # X_q̇ = M_u̇
            'sway_yaw': float(get_val(ma_sway_yaw)),        # Y_ṙ = N_v̇
        },
        'cross_coupling_linear_drag': {
            'X_q': float(get_val(cc_Xq)),
            'Y_r': float(get_val(cc_Yr)),
            'M_w': float(get_val(cc_Mw)),
            'N_v': float(get_val(cc_Nv)),
        },
    }

    return result


# ============================================================================
# NUMPY-BASED PREDICTION (for diagnostics - mirrors CasADi equations exactly)
# ============================================================================

def predict_forces(params, data):
    """Compute predicted forces/torques using identified parameters.

    This re-implements the Fossen equations in pure numpy for fast evaluation
    on train/test sets after the optimization is complete.
    """
    d = data
    MA = np.array(params['added_mass_diagonal'])
    Dq = np.array(params['quadratic_drag_diagonal'])
    Dl = np.array(params['linear_drag_diagonal'])
    CoM = np.array(params['center_of_mass'])
    CoB = np.array(params['center_of_buoyancy'])
    B   = params['buoyancy']
    W_t = params['tether_pull']
    ma_sp = params['cross_coupling_added_mass']['surge_pitch']
    ma_sy = params['cross_coupling_added_mass']['sway_yaw']
    Xq = params['cross_coupling_linear_drag']['X_q']
    Yr = params['cross_coupling_linear_drag']['Y_r']
    Mw = params['cross_coupling_linear_drag']['M_w']
    Nv = params['cross_coupling_linear_drag']['N_v']

    u, v, w = d['u'], d['v'], d['w']
    p, q, r = d['p'], d['q'], d['r']
    u_dot, v_dot, w_dot = d['u_dot'], d['v_dot'], d['w_dot']
    p_dot, q_dot, r_dot = d['p_dot'], d['q_dot'], d['r_dot']
    qx, qy, qz, qw = d['qx'], d['qy'], d['qz'], d['qw']

    # Quaternion → body-frame gravity direction
    Zx = 2 * (qx * qz - qw * qy)
    Zy = 2 * (qy * qz + qw * qx)
    Zz = qw**2 - qx**2 - qy**2 + qz**2

    # Restoring forces
    Fg = np.column_stack([-WEIGHT * Zx, -WEIGHT * Zy, -WEIGHT * Zz])
    Fb = np.column_stack([B * Zx, B * Zy, B * Zz])
    Ft = np.column_stack([-W_t * Zx, -W_t * Zy, -W_t * Zz])
    Rest_F = Fg + Fb + Ft

    # Restoring moments: r × F
    Mom_g = np.cross(CoM, Fg)
    Mom_b = np.cross(CoB, Fb)
    Rest_M = Mom_g + Mom_b

    # 6-DoF predicted forces/torques
    Fx_p = ((MASS + MA[0]) * u_dot + ma_sp * q_dot
            - MASS * (v * r - w * q)
            + Dq[0] * u * np.abs(u) + Dl[0] * u + Xq * q
            - Rest_F[:, 0])

    Fy_p = ((MASS + MA[1]) * v_dot + ma_sy * r_dot
            - MASS * (w * p - u * r)
            + Dq[1] * v * np.abs(v) + Dl[1] * v + Yr * r
            - Rest_F[:, 1])

    Fz_p = ((MASS + MA[2]) * w_dot
            - MASS * (u * q - v * p)
            + Dq[2] * w * np.abs(w) + Dl[2] * w
            - Rest_F[:, 2])

    Tx_p = ((I_XX + MA[3]) * p_dot
            - (I_YY - I_ZZ) * q * r
            + Dq[3] * p * np.abs(p) + Dl[3] * p
            - Rest_M[:, 0])

    Ty_p = (ma_sp * u_dot + (I_YY + MA[4]) * q_dot
            - (I_ZZ - I_XX) * r * p
            + Dq[4] * q * np.abs(q) + Dl[4] * q + Mw * w
            - Rest_M[:, 1])

    Tz_p = (ma_sy * v_dot + (I_ZZ + MA[5]) * r_dot
            - (I_XX - I_YY) * p * q
            + Dq[5] * r * np.abs(r) + Dl[5] * r + Nv * v
            - Rest_M[:, 2])

    return {'Fx': Fx_p, 'Fy': Fy_p, 'Fz': Fz_p,
            'Tx': Tx_p, 'Ty': Ty_p, 'Tz': Tz_p}


# ============================================================================
# DIAGNOSTICS & REPORTING
# ============================================================================

def compute_axis_metrics(actual, predicted):
    """Compute R², NRMSE, and MAE for a single force/torque axis."""
    residual = actual - predicted
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)
    nrmse = np.sqrt(np.mean(residual**2)) / (np.ptp(actual) + 1e-10)
    mae = np.mean(np.abs(residual))
    return {'R2': round(float(r2), 6),
            'NRMSE': round(float(nrmse), 6),
            'MAE': round(float(mae), 6)}


def compute_all_metrics(params, data):
    """Compute per-axis metrics on a dataset."""
    pred = predict_forces(params, data)
    return {ax: compute_axis_metrics(data[ax], pred[ax])
            for ax in ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']}


def print_metrics_table(train_metrics, test_metrics=None):
    """Print formatted metrics comparison table."""
    axes = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    units = ['N', 'N', 'N', 'N·m', 'N·m', 'N·m']

    if test_metrics:
        header = (f"{'Axis':<6} {'Train R²':>10} {'Test R²':>10} "
                  f"{'Train NRMSE':>12} {'Test NRMSE':>12} "
                  f"{'Train MAE':>10} {'Test MAE':>10}")
        print(f"\n{header}")
        print("─" * len(header))
        for ax, unit in zip(axes, units):
            tm, em = train_metrics[ax], test_metrics[ax]
            print(f"{ax:<6} {tm['R2']:>10.4f} {em['R2']:>10.4f} "
                  f"{tm['NRMSE']:>12.4f} {em['NRMSE']:>12.4f} "
                  f"{tm['MAE']:>8.4f} {unit:>2s} {em['MAE']:>8.4f} {unit:>2s}")

        # Flag potential overfitting
        for ax in axes:
            r2_drop = train_metrics[ax]['R2'] - test_metrics[ax]['R2']
            if r2_drop > 0.15:
                print(f"\n  [!] Overfitting Alert: {ax} shows R² drop of {r2_drop:.3f} "
                      f"(train->test).")
    else:
        header = f"{'Axis':<6} {'R²':>10} {'NRMSE':>12} {'MAE':>12}"
        print(f"\n{header}")
        print("─" * len(header))
        for ax, unit in zip(axes, units):
            m = train_metrics[ax]
            print(f"{ax:<6} {m['R2']:>10.4f} {m['NRMSE']:>12.4f} {m['MAE']:>10.4f} {unit}")


def plot_residuals(params, data, output_dir, prefix="train"):
    """Generate 6-panel overlay + histogram plots for residual diagnostics."""
    pred = predict_forces(params, data)
    axes = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    labels = ['Surge Force [N]', 'Sway Force [N]', 'Heave Force [N]',
              'Roll Torque [N·m]', 'Pitch Torque [N·m]', 'Yaw Torque [N·m]']

    # -- Predicted vs Measured overlay --
    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Force/Torque: Predicted vs Measured ({prefix.replace("_", " ").title()})',
                 fontsize=14, fontweight='bold')

    n_pts = len(data['Fx'])
    t = np.arange(n_pts) * 0.02  # 50 Hz

    for i, (ax_name, label) in enumerate(zip(axes, labels)):
        ax = axs[i // 2, i % 2]
        ax.plot(t, data[ax_name], alpha=0.6, lw=0.5, label='Measured', color='#4A90D9')
        ax.plot(t, pred[ax_name], alpha=0.6, lw=0.5, label='Predicted', color='#E85D3A')
        ax.set_ylabel(label, fontsize=9)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.25)
        if i >= 4:
            ax.set_xlabel('Time [s]', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"residuals_{prefix}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # -- Residual histograms --
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'Residual Distribution ({prefix.replace("_", " ").title()})',
                 fontsize=14, fontweight='bold')

    for i, (ax_name, label) in enumerate(zip(axes, labels)):
        ax = axs[i // 2, i % 2]
        residual = data[ax_name] - pred[ax_name]
        ax.hist(residual, bins=80, alpha=0.7, color='#4A90D9',
                edgecolor='white', linewidth=0.3)
        ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(np.mean(residual), color='orange', linestyle='-', linewidth=1.5,
                   label=f'μ = {np.mean(residual):.3f}')
        ax.set_xlabel(label.split('[')[0].strip() + ' Residual', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"residual_hist_{prefix}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# PARAMETER EXPORT
# ============================================================================

def export_parameters(params, output_path, train_metrics, test_metrics,
                      train_files, test_files):
    """Export identified parameters to a structured JSON file.

    This JSON is consumed by:
        - 04_validate_model.py  (forward simulation validation)
        - Future acados NMPC pipeline (parameter loading)
        - Unity simulation  (via manual copy-paste or automated loader)
    """
    export = {
        'metadata': {
            'description': 'Douglas AUV Hydrodynamic Parameters - CasADi/IPOPT System Identification',
            'generated_at': datetime.now().isoformat(),
            'vehicle': 'Douglas (McGill Robotics)',
            'convention': 'FLU body frame (ROS 2), quaternion orientation',
        },
        'physical_constants': {
            'dry_mass_kg': MASS,
            'weight_N': WEIGHT,
            'inertia_kgm2': [float(I_XX), float(I_YY), float(I_ZZ)],
        },
        'identified_parameters': {
            'buoyancy_N': params['buoyancy'],
            'tether_pull_N': params['tether_pull'],
            'center_of_mass_m': params['center_of_mass'],
            'center_of_buoyancy_m': params['center_of_buoyancy'],
            'added_mass_diagonal': {
                'translational_kg': params['added_mass_diagonal'][:3],
                'rotational_kgm2': params['added_mass_diagonal'][3:],
            },
            'added_mass_cross_coupling': params['cross_coupling_added_mass'],
            'linear_drag_diagonal': {
                'translational_Ns_m': params['linear_drag_diagonal'][:3],
                'rotational_Nms_rad': params['linear_drag_diagonal'][3:],
            },
            'quadratic_drag_diagonal': {
                'translational_Ns2_m2': params['quadratic_drag_diagonal'][:3],
                'rotational_Nms2_rad2': params['quadratic_drag_diagonal'][3:],
            },
            'linear_drag_cross_coupling': params['cross_coupling_linear_drag'],
        },
        'optimization': {
            'converged': params['converged'],
            'total_cost': params['cost'],
            'wls_error': params['wls_error'],
        },
        'validation': {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_bags': [os.path.basename(f) for f in train_files],
            'test_bags': [os.path.basename(f) for f in test_files],
        },
    }

    with open(output_path, 'w') as f:
        json.dump(export, f, indent=2)
    print(f"\nParameters exported to: {output_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_sysid(args):
    """Orchestrate the full SysID pipeline."""

    print("=" * 62)
    print("  Douglas AUV - Hydrodynamic System Identification")
    print("  McGill Robotics | CasADi + IPOPT | Fossen 6-DoF")
    print("=" * 62)

    # -- Load data --
    df_train, df_test, train_files, test_files = load_datasets(
        args.dataset_dir, args.test_bag, args.single_bag, args.coasting_only)

    train_data = extract_data_arrays(df_train)
    test_data = extract_data_arrays(df_test) if len(df_test) > 0 else None

    # -- Multi-start optimization --
    n_starts = min(args.starts, len(INITIAL_GUESSES))
    guesses = INITIAL_GUESSES[:n_starts]

    print(f"\n{'=' * 62}")
    print(f"  Running {n_starts}-Start Optimization")
    print(f"{'=' * 62}")

    best_result = None
    best_test_cost = float('inf')
    best_start_idx = -1

    for i, guess in enumerate(guesses):
        print(f"\n{'─' * 50}")
        print(f"  Start {i + 1}/{n_starts}: {guess['name']}")
        print(f"{'─' * 50}")

        result = build_and_solve(train_data, guess, verbose=(i == 0))

        if result is None:
            print("  [-] Failed to solve.")
            continue

        # Evaluate on test set (or train set if no test data) to select best
        if test_data is not None:
            test_pred = predict_forces(result, test_data)
            test_cost = sum(
                np.sum((test_data[k] - test_pred[k])**2)
                for k in ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
            )
        else:
            test_cost = result['cost']

        status = "[+] Converged" if result['converged'] else "! Partial convergence"
        print(f"  {status} | Train cost: {result['cost']:.2f} | "
              f"{'Test' if test_data else 'Train'} MSE: {test_cost:.2f}")

        if test_cost < best_test_cost:
            best_test_cost = test_cost
            best_result = result
            best_start_idx = i

    if best_result is None:
        print("\n[-] ERROR: All optimization starts failed. Check data quality.")
        sys.exit(1)

    # -- Report results --
    print(f"\n{'=' * 62}")
    print(f"  Best: Start {best_start_idx + 1} - {guesses[best_start_idx]['name']}")
    print(f"{'=' * 62}")

    params = best_result
    MA = params['added_mass_diagonal']
    Dl = params['linear_drag_diagonal']
    Dq = params['quadratic_drag_diagonal']
    cc_ma = params['cross_coupling_added_mass']
    cc_drag = params['cross_coupling_linear_drag']

    print(f"\nWeight:            {WEIGHT:.2f} N")
    print(f"Buoyancy:          {params['buoyancy']:.2f} N "
          f"({params['buoyancy'] - WEIGHT:+.2f} N net)")
    print(f"Tether Pull:       {params['tether_pull']:.2f} N")
    print(f"CoM [x, y, z]:     [{params['center_of_mass'][0]:.4f}, "
          f"{params['center_of_mass'][1]:.4f}, {params['center_of_mass'][2]:.4f}] m")
    print(f"CoB [x, y, z]:     [{params['center_of_buoyancy'][0]:.4f}, "
          f"{params['center_of_buoyancy'][1]:.4f}, {params['center_of_buoyancy'][2]:.4f}] m")
    print(f"CoB−CoM (z):       {params['center_of_buoyancy'][2] - params['center_of_mass'][2]:.4f} m")

    print(f"\n{'─'*40} Diagonal Added Mass {'─'*40}")
    print(f"  Translational:   [{MA[0]:.2f}, {MA[1]:.2f}, {MA[2]:.2f}] kg")
    print(f"  Rotational:      [{MA[3]:.4f}, {MA[4]:.4f}, {MA[5]:.4f}] kg·m²")

    print(f"\n{'─'*37} Cross-Coupling Added Mass {'─'*37}")
    print(f"  Surge↔Pitch (X_q̇ = M_u̇): {cc_ma['surge_pitch']:.4f} kg·m")
    print(f"  Sway↔Yaw   (Y_ṙ = N_v̇):  {cc_ma['sway_yaw']:.4f} kg·m")

    print(f"\n{'─'*40} Diagonal Linear Drag {'─'*39}")
    print(f"  Translational:   [{Dl[0]:.2f}, {Dl[1]:.2f}, {Dl[2]:.2f}] N·s/m")
    print(f"  Rotational:      [{Dl[3]:.3f}, {Dl[4]:.3f}, {Dl[5]:.3f}] N·m·s/rad")

    print(f"\n{'─'*38} Diagonal Quadratic Drag {'─'*38}")
    print(f"  Translational:   [{Dq[0]:.2f}, {Dq[1]:.2f}, {Dq[2]:.2f}] N·s²/m²")
    print(f"  Rotational:      [{Dq[3]:.3f}, {Dq[4]:.3f}, {Dq[5]:.3f}] N·m·s²/rad²")

    print(f"\n{'─'*37} Cross-Coupling Linear Drag {'─'*36}")
    print(f"  X_q (Surge ← Pitch rate):     {cc_drag['X_q']:+.4f} N·s/rad")
    print(f"  Y_r (Sway  ← Yaw rate):       {cc_drag['Y_r']:+.4f} N·s/rad")
    print(f"  M_w (Pitch ← Heave velocity): {cc_drag['M_w']:+.4f} N·m·s/m")
    print(f"  N_v (Yaw   ← Sway velocity):  {cc_drag['N_v']:+.4f} N·m·s/m")

    # -- Diagnostics --
    os.makedirs(args.output_dir, exist_ok=True)

    train_metrics = compute_all_metrics(params, train_data)
    test_metrics = compute_all_metrics(params, test_data) if test_data is not None else None

    print_metrics_table(train_metrics, test_metrics)

    print("\nGenerating diagnostic plots...")
    plot_residuals(params, train_data, args.output_dir, prefix="train")
    if test_data is not None:
        plot_residuals(params, test_data, args.output_dir, prefix="test")

    # -- Export --
    json_path = os.path.join(args.output_dir, "identified_parameters.json")
    export_parameters(params, json_path, train_metrics,
                      test_metrics if test_metrics else {},
                      train_files, test_files)

    # -- Unity copy-paste --
    CoM_v = params['center_of_mass']
    CoB_v = params['center_of_buoyancy']
    print(f"\n{'─'*30} Unity Copy-Paste (ROS FLU → Unity) {'─'*30}")
    print("// Buoyancy.cs")
    print(f"manualBuoyancyForce       = {params['buoyancy']:.2f}f;")
    print(f"centerOfMass              = new Vector3({-CoM_v[1]:.4f}f, {CoM_v[2]:.4f}f, {CoM_v[0]:.4f}f);")
    print(f"centerOfBuoyancy          = new Vector3({-CoB_v[1]:.4f}f, {CoB_v[2]:.4f}f, {CoB_v[0]:.4f}f);")
    print("\n// Drag.cs")
    print(f"addedMassTranslational    = new Vector3({MA[1]:.4f}f, {MA[2]:.4f}f, {MA[0]:.4f}f);")
    print(f"addedMassRotational       = new Vector3({MA[4]:.4f}f, {MA[5]:.4f}f, {MA[3]:.4f}f);")
    print(f"linearDragTranslational   = new Vector3({Dl[1]:.4f}f, {Dl[2]:.4f}f, {Dl[0]:.4f}f);")
    print(f"linearDragRotational      = new Vector3({Dl[4]:.4f}f, {Dl[5]:.4f}f, {Dl[3]:.4f}f);")
    print(f"quadDragTranslational     = new Vector3({Dq[1]:.4f}f, {Dq[2]:.4f}f, {Dq[0]:.4f}f);")
    print(f"quadDragRotational        = new Vector3({Dq[4]:.4f}f, {Dq[5]:.4f}f, {Dq[3]:.4f}f);")

    print(f"\n{'=' * 62}")
    print(f"  System Identification Complete!")
    print(f"  Results saved to: {args.output_dir}/")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hydrodynamic System Identification for Douglas AUV (McGill Robotics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: auto train/test split, 3 multi-starts
  python 03_hydrodynamic_sysid.py

  # Use a specific bag as the held-out test set
  python 03_hydrodynamic_sysid.py --test-bag processed_datasets/finals_run_dataset.csv

  # Quick single-start run for debugging
  python 03_hydrodynamic_sysid.py --starts 1
        """)
    parser.add_argument("--dataset-dir", default="processed_datasets",
                        help="Directory containing *_dataset.csv files from Step 02 "
                             "(default: processed_datasets)")
    parser.add_argument("--test-bag", default=None,
                        help="Specific CSV file to hold out as test set. "
                             "If omitted, uses automatic bag-level 80/20 split.")
    parser.add_argument("--output-dir", default="sysid_results",
                        help="Output directory for plots, JSON, and diagnostics "
                             "(default: sysid_results)")
    parser.add_argument("--single-bag", default=None,
                        help="Load a single CSV and use 80/20 temporal split "
                             "(first 80%% train, last 20%% test). Best for "
                             "long continuous runs like the finals mission.")
    parser.add_argument("--starts", type=int, default=3,
                        help="Number of multi-start optimization runs (default: 3)")
    parser.add_argument("--coasting-only", action="store_true",
                        help="Filter dataset to only include deceleration/coasting segments "
                             "to isolate drag parameters from saturated PID outputs.")

    args = parser.parse_args()
    run_sysid(args)