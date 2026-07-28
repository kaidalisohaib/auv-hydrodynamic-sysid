#!/usr/bin/env python3
"""
04_validate_model.py - Forward Simulation Validation (ERK4)
McGill Robotics | douglas-sysid pipeline

Validates the identified hydrodynamic model by running a forward simulation
using recorded thruster inputs (wrench) and comparing predicted velocities
against actual sensor measurements.

Uses ERK4 (Explicit Runge-Kutta 4th Order) integration - the same scheme
that acados uses for NMPC - to guarantee that model behavior in this
validation matches the deployed controller exactly.

Two validation modes:
    1. Full open-loop:  Propagate state from initial conditions only.
                        Tests long-horizon model stability and drift.
    2. Windowed:        Reset state every N seconds from measurements.
                        Tests MPC-relevant prediction accuracy over the
                        controller's preview horizon (default: 1.5s).

Usage:
    python 04_validate_model.py \\
        --params sysid_results/identified_parameters.json \\
        --data processed_datasets/finals_run_dataset.csv

    python 04_validate_model.py \\
        --params sysid_results/identified_parameters.json \\
        --data processed_datasets/controls-testing_dataset.csv \\
        --mpc-horizon 2.0
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# PHYSICAL CONSTANTS (must match 03_hydrodynamic_sysid.py exactly)
# ============================================================================
MASS = 26.8
WEIGHT = MASS * 9.81
I_XX = 796.517 * 0.00064516
I_YY = 818.679 * 0.00064516
I_ZZ = 1250.459 * 0.00064516


# ============================================================================
# PARAMETER LOADING
# ============================================================================

def load_parameters(json_path):
    """Load identified parameters from the JSON export of Step 03.

    Returns a flat dict with all parameters needed for dynamics evaluation.
    """
    with open(json_path) as f:
        data = json.load(f)

    p = data['identified_parameters']
    ma_diag = p['added_mass_diagonal']
    dl_diag = p['linear_drag_diagonal']
    dq_diag = p['quadratic_drag_diagonal']

    return {
        'buoyancy': p['buoyancy_N'],
        'tether_pull': p['tether_pull_N'],
        'CoM': np.array(p['center_of_mass_m']),
        'CoB': np.array(p['center_of_buoyancy_m']),
        'MA': np.array(ma_diag['translational_kg'] + ma_diag['rotational_kgm2']),
        'Dl': np.array(dl_diag['translational_Ns_m'] + dl_diag['rotational_Nms_rad']),
        'Dq': np.array(dq_diag['translational_Ns2_m2'] + dq_diag['rotational_Nms2_rad2']),
        'ma_sp': p['added_mass_cross_coupling']['surge_pitch'],
        'ma_sy': p['added_mass_cross_coupling']['sway_yaw'],
        'Xq': p['linear_drag_cross_coupling']['X_q'],
        'Yr': p['linear_drag_cross_coupling']['Y_r'],
        'Mw': p['linear_drag_cross_coupling']['M_w'],
        'Nv': p['linear_drag_cross_coupling']['N_v'],
    }


def load_test_data(csv_path):
    """Load and validate a test dataset CSV."""
    df = pd.read_csv(csv_path)

    required = ['time', 'u', 'v', 'w', 'p', 'q', 'r',
                'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Quaternion columns (with Euler fallback)
    quat_cols = ['qx', 'qy', 'qz', 'qw']
    if not all(c in df.columns for c in quat_cols):
        if 'roll' in df.columns and 'pitch' in df.columns:
            print("  ! Converting Euler angles → quaternions (yaw=0 assumed).")
            phi, theta = df['roll'].values, df['pitch'].values
            cp, sp = np.cos(phi / 2), np.sin(phi / 2)
            ct, st = np.cos(theta / 2), np.sin(theta / 2)
            df['qx'] = sp * ct
            df['qy'] = cp * st
            df['qz'] = -sp * st
            df['qw'] = cp * ct
        else:
            raise ValueError("Dataset missing quaternion columns (qx, qy, qz, qw) "
                             "and Euler angles (roll, pitch).")

    # Drop any rows with NaN
    df = df.dropna(subset=required + quat_cols).reset_index(drop=True)
    return df


# ============================================================================
# MASS MATRIX
# ============================================================================

def build_mass_matrix(params):
    """Build the 6×6 mass/inertia matrix including cross-coupling terms.

    Structure (with off-diagonal added mass):
        [m+Xa,   0,    0,    0,  ma_sp,  0   ]
        [  0,  m+Ya,   0,    0,    0,  ma_sy ]
        [  0,    0,  m+Za,   0,    0,    0   ]
        [  0,    0,    0,  Ix+Kp,  0,    0   ]
        [ma_sp,  0,    0,    0,  Iy+Mq,  0   ]
        [  0,  ma_sy,  0,    0,    0,  Iz+Nr ]
    """
    MA = params['MA']
    M = np.diag([
        MASS + MA[0], MASS + MA[1], MASS + MA[2],
        I_XX + MA[3], I_YY + MA[4], I_ZZ + MA[5]
    ])
    # Symmetric cross-coupling: M_A[0,4] = M_A[4,0], M_A[1,5] = M_A[5,1]
    M[0, 4] = M[4, 0] = params['ma_sp']
    M[1, 5] = M[5, 1] = params['ma_sy']
    return M


# ============================================================================
# DYNAMICS FUNCTION
# ============================================================================

def dynamics(state, tau, params):
    """Compute the state derivative for the 10-state system.

    State vector: [u, v, w, p, q, r, qx, qy, qz, qw]
    Input vector: [Fx, Fy, Fz, Tx, Ty, Tz]  (recorded wrench)

    The equations mirror 03_hydrodynamic_sysid.py exactly:
        τ = M·v̇ + C_RB(v)·v + D(v)·v - g(η)
        → M·v̇ = τ - C_RB(v)·v - D(v)·v + g(η)

    Returns: state_dot [10]
    """
    u, v, w, p, q, r = state[:6]
    qx, qy, qz, qw = state[6:10]

    MA = params['MA']
    Dq = params['Dq']
    Dl = params['Dl']
    CoM = params['CoM']
    CoB = params['CoB']
    B   = params['buoyancy']
    W_t = params['tether_pull']

    # -- Quaternion-based restoring forces --
    Zx = 2 * (qx * qz - qw * qy)
    Zy = 2 * (qy * qz + qw * qx)
    Zz = qw**2 - qx**2 - qy**2 + qz**2

    Fg = np.array([-WEIGHT * Zx, -WEIGHT * Zy, -WEIGHT * Zz])
    Fb = np.array([B * Zx, B * Zy, B * Zz])
    Ft = np.array([-W_t * Zx, -W_t * Zy, -W_t * Zz])
    Rest_F = Fg + Fb + Ft

    Mom_g = np.cross(CoM, Fg)
    Mom_b = np.cross(CoB, Fb)
    Rest_M = Mom_g + Mom_b

    restoring = np.concatenate([Rest_F, Rest_M])

    # -- Rigid-body Coriolis (as they appear on LHS of SysID equation) --
    coriolis = np.array([
        -MASS * (v * r - w * q),
        -MASS * (w * p - u * r),
        -MASS * (u * q - v * p),
        -(I_YY - I_ZZ) * q * r,
        -(I_ZZ - I_XX) * r * p,
        -(I_XX - I_YY) * p * q,
    ])

    # -- Hydrodynamic drag (diagonal + cross-coupling) --
    vel = np.array([u, v, w, p, q, r])
    drag = Dq * vel * np.abs(vel) + Dl * vel

    # Cross-coupling drag terms
    drag[0] += params['Xq'] * q      # Surge ← pitch rate
    drag[1] += params['Yr'] * r      # Sway  ← yaw rate
    drag[4] += params['Mw'] * w      # Pitch ← heave velocity
    drag[5] += params['Nv'] * v      # Yaw   ← sway velocity

    # -- Solve M·v̇ = τ − coriolis − drag + restoring --
    rhs = tau - coriolis - drag + restoring
    v_dot = params['_M_inv'] @ rhs

    # -- Quaternion kinematics: q̇ = 0.5 · Ω(ω) · q --
    qx_dot = 0.5 * ( qw * p - qz * q + qy * r)
    qy_dot = 0.5 * ( qz * p + qw * q - qx * r)
    qz_dot = 0.5 * (-qy * p + qx * q + qw * r)
    qw_dot = 0.5 * (-qx * p - qy * q - qz * r)

    return np.array([*v_dot, qx_dot, qy_dot, qz_dot, qw_dot])


# ============================================================================
# ERK4 INTEGRATOR
# ============================================================================

def erk4_step(state, tau, params, dt):
    """Single ERK4 (Explicit Runge-Kutta 4th Order) integration step.

    This is the same integration scheme used by acados with
    solver_options.integrator_type = 'ERK', ensuring that validation
    results directly predict NMPC behavior.
    """
    k1 = dynamics(state, tau, params)
    k2 = dynamics(state + 0.5 * dt * k1, tau, params)
    k3 = dynamics(state + 0.5 * dt * k2, tau, params)
    k4 = dynamics(state + dt * k3, tau, params)

    new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Renormalize quaternion to prevent numerical drift from unit sphere
    q_norm = np.linalg.norm(new_state[6:10])
    if q_norm > 1e-10:
        new_state[6:10] /= q_norm

    return new_state


# ============================================================================
# FORWARD SIMULATION
# ============================================================================

def forward_simulate(df, params, window_seconds=None):
    """Run forward simulation over a dataset.

    Args:
        df: DataFrame with measured data (time, velocities, wrench, quaternion)
        params: Identified parameter dict (with pre-computed '_M_inv')
        window_seconds: If set, reset state from measurements every N seconds.
                        If None, run full open-loop from initial state.

    Returns:
        dict with time, predicted velocities, measured velocities,
        predicted quaternions, measured quaternions
    """
    times = df['time'].values
    dt_arr = np.diff(times)
    N = len(df)

    # Recorded inputs (external wrench from thrusters)
    tau_log = np.column_stack([
        df['Fx'].values, df['Fy'].values, df['Fz'].values,
        df['Tx'].values, df['Ty'].values, df['Tz'].values
    ])

    # Measured state (for comparison and window resets)
    meas_vel = np.column_stack([
        df['u'].values, df['v'].values, df['w'].values,
        df['p'].values, df['q'].values, df['r'].values
    ])
    meas_quat = np.column_stack([
        df['qx'].values, df['qy'].values, df['qz'].values, df['qw'].values
    ])

    # Output buffers
    pred_vel  = np.zeros((N, 6))
    pred_quat = np.zeros((N, 4))

    # Initialize from first measurement
    state = np.concatenate([meas_vel[0], meas_quat[0]])
    pred_vel[0]  = state[:6]
    pred_quat[0] = state[6:]

    # Compute reset interval (in frames)
    median_dt = np.median(dt_arr[dt_arr > 0])
    reset_interval = int(window_seconds / median_dt) if window_seconds else N + 1

    for k in range(N - 1):
        # Window reset: re-initialize from measurements periodically
        if window_seconds and (k % reset_interval == 0) and k > 0:
            state = np.concatenate([meas_vel[k], meas_quat[k]])

        dt = dt_arr[k]
        if dt <= 0 or dt > 1.0:
            # Invalid timestep (duplicate or gap) - snap to measurement
            state = np.concatenate([meas_vel[k + 1], meas_quat[k + 1]])
        else:
            state = erk4_step(state, tau_log[k], params, dt)

        pred_vel[k + 1]  = state[:6]
        pred_quat[k + 1] = state[6:]

    return {
        'time': times - times[0],
        'pred_vel': pred_vel,
        'meas_vel': meas_vel,
        'pred_quat': pred_quat,
        'meas_quat': meas_quat,
    }


# ============================================================================
# METRICS & PLOTTING
# ============================================================================

VEL_LABELS = ['u (Surge) [m/s]', 'v (Sway) [m/s]', 'w (Heave) [m/s]',
              'p (Roll) [rad/s]', 'q (Pitch) [rad/s]', 'r (Yaw) [rad/s]']


def compute_velocity_metrics(meas, pred):
    """Compute R² and NRMSE per velocity axis."""
    metrics = {}
    for i, label in enumerate(VEL_LABELS):
        residual = meas[:, i] - pred[:, i]
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((meas[:, i] - np.mean(meas[:, i]))**2) + 1e-10
        r2 = 1.0 - ss_res / ss_tot
        nrmse = np.sqrt(np.mean(residual**2)) / (np.ptp(meas[:, i]) + 1e-10)
        metrics[label] = {'R2': round(r2, 4), 'NRMSE': round(nrmse, 4)}
    return metrics


def print_velocity_metrics(metrics, title):
    """Print formatted velocity metrics table."""
    print(f"\n  {title}")
    print(f"  {'Velocity':<22} {'R²':>8} {'NRMSE':>8}")
    print(f"  {'─' * 40}")
    for label, m in metrics.items():
        r2_str = f"{m['R2']:>8.4f}"
        nrmse_str = f"{m['NRMSE']:>8.4f}"
        # Color-code R² quality
        if m['R2'] > 0.9:
            status = "[+]"
        elif m['R2'] > 0.7:
            status = "~"
        else:
            status = "[-]"
        print(f"  {label:<22} {r2_str} {nrmse_str}  {status}")


def plot_simulation(result, output_dir, prefix="full", mpc_horizon=None):
    """Plot measured vs predicted velocity overlay (6 panels)."""
    t = result['time']
    meas = result['meas_vel']
    pred = result['pred_vel']

    if "window" in prefix or mpc_horizon:
        sim_type = f"Windowed ({mpc_horizon}s resets)"
    else:
        sim_type = "Full Open-Loop"

    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Forward Simulation Validation - {sim_type}',
                 fontsize=14, fontweight='bold')

    for i, label in enumerate(VEL_LABELS):
        ax = axs[i // 2, i % 2]
        ax.plot(t, meas[:, i], lw=0.8, alpha=0.7,
                label='Measured (DVL/IMU)', color='#4A90D9')
        ax.plot(t, pred[:, i], lw=0.8, alpha=0.7,
                label='Predicted (ERK4)', color='#E85D3A')
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.25)
        if i >= 4:
            ax.set_xlabel('Time [s]', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"forward_sim_{prefix}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_velocity_error(result, output_dir, prefix="full"):
    """Plot velocity prediction error over time (6 panels)."""
    t = result['time']
    error = result['meas_vel'] - result['pred_vel']

    fig, axs = plt.subplots(3, 2, figsize=(16, 10))
    fig.suptitle(f'Velocity Prediction Error ({prefix.replace("_", " ").title()})',
                 fontsize=14, fontweight='bold')

    for i, label in enumerate(VEL_LABELS):
        ax = axs[i // 2, i % 2]
        ax.plot(t, error[:, i], lw=0.5, alpha=0.8, color='#8B4513')
        ax.axhline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.fill_between(t, error[:, i], 0, alpha=0.15, color='#8B4513')
        ax.set_ylabel(f'{label.split("[")[0].strip()} Error', fontsize=9)
        ax.grid(True, alpha=0.25)
        if i >= 4:
            ax.set_xlabel('Time [s]', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, f"velocity_error_{prefix}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Forward simulation validation for Douglas AUV hydrodynamic model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs two validation modes:
  1. Full open-loop: propagates from initial state using only recorded wrench
     inputs. Tests whether the model is stable over long horizons.
  2. Windowed: resets to measured state every --mpc-horizon seconds. Tests
     whether the model is accurate over the NMPC preview window.

If the windowed R² values are high (>0.85) but open-loop diverges, the model
is still suitable for NMPC - the controller only needs 1-3 second accuracy.
        """)
    parser.add_argument("--params", required=True,
                        help="Path to identified_parameters.json from Step 03")
    parser.add_argument("--data", required=True,
                        help="Path to test dataset CSV")
    parser.add_argument("--output-dir", default="sysid_results",
                        help="Output directory for validation plots (default: sysid_results)")
    parser.add_argument("--mpc-horizon", type=float, default=1.5,
                        help="MPC prediction horizon in seconds for windowed validation "
                             "(default: 1.5)")

    args = parser.parse_args()

    print("=" * 62)
    print("  Douglas AUV - Forward Simulation Validation (ERK4)")
    print("  McGill Robotics | douglas-sysid pipeline")
    print("=" * 62)

    # -- Load --
    params = load_parameters(args.params)
    df = load_test_data(args.data)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n  Parameters: {args.params}")
    print(f"  Test data:  {args.data} ({len(df):,} frames)")

    # Pre-compute mass matrix inverse (constant, used at every timestep)
    M = build_mass_matrix(params)
    cond = np.linalg.cond(M)
    params['_M_inv'] = np.linalg.inv(M)

    print(f"\n  Mass matrix condition number: {cond:.1f}", end="")
    if cond > 1000:
        print(" ! HIGH - may cause numerical instability!")
    else:
        print(" [+]")

    duration = df['time'].iloc[-1] - df['time'].iloc[0]
    print(f"  Dataset duration: {duration:.1f}s")

    # -- Mode 1: Full open-loop simulation --
    print(f"\n{'─' * 50}")
    print(f"  Mode 1: Full Open-Loop Forward Simulation")
    print(f"{'─' * 50}")

    result_full = forward_simulate(df, params, window_seconds=None)
    metrics_full = compute_velocity_metrics(result_full['meas_vel'],
                                            result_full['pred_vel'])
    print_velocity_metrics(metrics_full, "Full Open-Loop Results")
    plot_simulation(result_full, args.output_dir, prefix="full_openloop")
    plot_velocity_error(result_full, args.output_dir, prefix="full_openloop")

    # -- Mode 2: Windowed simulation (MPC-horizon resets) --
    print(f"\n{'─' * 50}")
    print(f"  Mode 2: Windowed Simulation ({args.mpc_horizon}s resets)")
    print(f"{'─' * 50}")

    result_window = forward_simulate(df, params, window_seconds=args.mpc_horizon)
    metrics_window = compute_velocity_metrics(result_window['meas_vel'],
                                              result_window['pred_vel'])
    print_velocity_metrics(metrics_window,
                           f"Windowed Results ({args.mpc_horizon}s horizon)")
    window_prefix = f"windowed_{args.mpc_horizon}s"
    plot_simulation(result_window, args.output_dir,
                    prefix=window_prefix, mpc_horizon=args.mpc_horizon)
    plot_velocity_error(result_window, args.output_dir, prefix=window_prefix)

    # -- Summary --
    print(f"\n{'=' * 62}")
    print(f"  Validation Complete!")
    print(f"{'=' * 62}")
    print(f"\n  Open-loop tests long-horizon model stability.")
    print(f"  Windowed ({args.mpc_horizon}s) tests MPC-relevant prediction accuracy.")
    print(f"\n  If windowed R² > 0.85 for translational axes, the model")
    print(f"  is ready for acados NMPC deployment on the Jetson Orin.")
    print(f"\n  Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
