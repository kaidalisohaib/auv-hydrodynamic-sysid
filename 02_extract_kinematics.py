#!/usr/bin/env python3
"""
02_extract_kinematics.py - Kinematic Data Extraction & Preprocessing
McGill Robotics | douglas-sysid pipeline

Reads ROS 2 bag files, extracts DVL velocity, IMU orientation (quaternion),
commanded wrench, and depth data. Resamples to a uniform 50 Hz grid,
applies Savitzky-Golay smoothing, computes numerical accelerations, and
exports clean CSV datasets for downstream system identification.

CRITICAL: Outputs raw quaternion components (qx, qy, qz, qw) instead of
Euler angles, enabling singularity-free hydrostatic modeling in Step 03.
Quaternions are re-normalized after filtering to preserve unit length.

Usage:
    python 02_extract_kinematics.py --input ../rosbags --output processed_datasets
"""

import os
import argparse
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores
import matplotlib.pyplot as plt

# Initialize typestore for ROS 2 message deserialization
typestore = get_typestore(Stores.LATEST)
deserialize_cdr = typestore.deserialize_cdr

# ==========================================
# CONFIGURATION
# ==========================================
MIN_DURATION_SECONDS = 5.0       # Skip bags shorter than this
RESAMPLE_PERIOD_MS = 20          # 50 Hz uniform grid
MIN_SUBMERGED_DEPTH = 0.3        # Depth filter threshold [m]
COASTING_BUFFER_FRAMES = 200     # 4s at 50 Hz - capture drag deceleration after thrusters off

# Savitzky-Golay filter parameters
WINDOW_LENGTH = 31               # Must be odd; increase for smoother acceleration
POLY_ORDER = 3                   # Polynomial order for local fit

# Anomaly rejection thresholds
MAX_LINEAR_ACCEL = 10.0          # m/s² (~1G, rejects collisions/snags)
MAX_ANGULAR_ACCEL = 20.0         # rad/s² (~1140 deg/s², rejects IMU glitches)

# Plotting
SAVE_PLOTS = False
SHOW_PLOTS = False

# ROS 2 Topic names (McGill Robotics AUV conventions)
TOPIC_LIN_VEL = "/auv_frame/dvl/velocity"   # geometry_msgs/msg/TwistStamped (FLU body frame)
TOPIC_ANG_VEL = "/auv_frame/imu"            # sensor_msgs/msg/Imu (quaternion + angular velocity)
TOPIC_WRENCH  = "/controls/total_effort"    # geometry_msgs/msg/Wrench (commanded forces/torques)
TOPIC_DEPTH   = "/auv_frame/depth"          # std_msgs/msg/Float64 (pressure-derived depth)


def find_rosbag_directories(root_path):
    """Recursively scan for ROS 2 bag directories containing metadata.yaml."""
    bag_dirs = []
    if not os.path.exists(root_path):
        print(f"Error: Directory '{root_path}' does not exist.")
        return bag_dirs

    for dirpath, dirnames, filenames in os.walk(root_path):
        if 'metadata.yaml' in filenames:
            bag_dirs.append(dirpath)
            dirnames.clear()  # Don't recurse into valid bag directories

    return sorted(bag_dirs)


def parse_mcgill_mcap(file_path):
    """Read a ROS 2 bag and extract kinematic data as a time-synchronized DataFrame.

    Returns a DataFrame with columns:
        time, u, v, w, p, q, r, qx, qy, qz, qw, Fx, Fy, Fz, Tx, Ty, Tz, depth
    Resampled to a uniform 50 Hz grid with NaN rows dropped.
    """
    print(f"  Reading bag: {file_path}")
    data_list = []

    with Reader(file_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            t_sec = timestamp * 1e-9  # Nanoseconds → seconds

            if connection.topic == TOPIC_LIN_VEL:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                data_list.append({
                    'time': t_sec,
                    'u': msg.twist.linear.x,
                    'v': msg.twist.linear.y,
                    'w': msg.twist.linear.z
                })

            elif connection.topic == TOPIC_ANG_VEL:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                data_list.append({
                    'time': t_sec,
                    'p': msg.angular_velocity.x,
                    'q': msg.angular_velocity.y,
                    'r': msg.angular_velocity.z,
                    'qx': msg.orientation.x,
                    'qy': msg.orientation.y,
                    'qz': msg.orientation.z,
                    'qw': msg.orientation.w,
                })

            elif connection.topic == TOPIC_WRENCH:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                data_list.append({
                    'time': t_sec,
                    'Fx': msg.force.x,  'Fy': msg.force.y,  'Fz': msg.force.z,
                    'Tx': msg.torque.x, 'Ty': msg.torque.y, 'Tz': msg.torque.z,
                })

            elif connection.topic == TOPIC_DEPTH:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                data_list.append({'time': t_sec, 'depth': msg.data})

    if not data_list:
        print("  Warning: No messages extracted.")
        return pd.DataFrame()

    print(f"  Extracted {len(data_list)} messages. Building timeline...")

    # Merge all topics into a single time-indexed DataFrame
    df = pd.DataFrame(data_list).sort_values('time')
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('datetime').groupby(level=0).mean()

    # Resample to a uniform grid so dt is constant for numerical differentiation
    df = df.resample(f'{RESAMPLE_PERIOD_MS}ms').mean().interpolate(method='time').dropna()
    df['time'] = df.index.astype(np.int64) / 1e9

    # ─── Data Sanitization ───
    initial_len = len(df)

    # 1. Depth filter: ensure AUV is submerged
    if 'depth' in df.columns:
        df = df[df['depth'] >= MIN_SUBMERGED_DEPTH]
    else:
        print("  Warning: No depth topic found. Skipping depth filter.")

    # 2. Activity filter: isolate active thruster periods + coasting tail
    if len(df) > 0 and all(c in df.columns for c in ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']):
        effort_mag = np.sqrt(df['Fx']**2 + df['Fy']**2 + df['Fz']**2 +
                             df['Tx']**2 + df['Ty']**2 + df['Tz']**2)
        active_indices = np.where(effort_mag > 0.01)[0]

        if len(active_indices) > 0:
            first_idx = active_indices[0]
            end_idx = min(active_indices[-1] + COASTING_BUFFER_FRAMES, len(df))
            df = df.iloc[first_idx:end_idx]
        else:
            print("  Warning: No active thruster commands found. Skipping bag.")
            return pd.DataFrame()

    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} inactive/surface frames.")

    return df.reset_index(drop=True)


def apply_kinematic_filters(df):
    """Smooth velocities, compute accelerations, filter quaternions, and reject anomalies."""
    # ─── Velocity Smoothing & Differentiation ───
    vel_cols = ['u', 'v', 'w', 'p', 'q', 'r']
    accel_cols = ['u_dot', 'v_dot', 'w_dot', 'p_dot', 'q_dot', 'r_dot']

    for v_col, a_col in zip(vel_cols, accel_cols):
        df[v_col] = savgol_filter(df[v_col], WINDOW_LENGTH, POLY_ORDER)
        raw_accel = np.gradient(df[v_col], df['time'])
        df[a_col] = savgol_filter(raw_accel, WINDOW_LENGTH, POLY_ORDER)

    # ─── Quaternion Smoothing & Renormalization ───
    quat_cols = ['qx', 'qy', 'qz', 'qw']
    if all(q in df.columns for q in quat_cols):
        for q_col in quat_cols:
            df[q_col] = savgol_filter(df[q_col], WINDOW_LENGTH, POLY_ORDER)

        # CRITICAL: Savitzky-Golay filtering distorts quaternion magnitude.
        # Renormalize to ensure ||q|| = 1, preventing downstream CasADi errors.
        q_norm = np.sqrt(df['qx']**2 + df['qy']**2 + df['qz']**2 + df['qw']**2)
        for q_col in quat_cols:
            df[q_col] /= q_norm
    else:
        print("  Warning: Quaternion columns not found in data.")

    # ─── Anomaly Rejection ───
    initial_len = len(df)

    df = df[
        (df['u_dot'].abs() < MAX_LINEAR_ACCEL) &
        (df['v_dot'].abs() < MAX_LINEAR_ACCEL) &
        (df['w_dot'].abs() < MAX_LINEAR_ACCEL)
    ]
    df = df[
        (df['p_dot'].abs() < MAX_ANGULAR_ACCEL) &
        (df['q_dot'].abs() < MAX_ANGULAR_ACCEL) &
        (df['r_dot'].abs() < MAX_ANGULAR_ACCEL)
    ]

    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} anomaly frames (exceeded acceleration limits).")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract kinematic data from ROS 2 bags for hydrodynamic SysID.")
    parser.add_argument("--input", default="../rosbags",
                        help="Root directory containing ROS 2 bag folders (default: ../rosbags)")
    parser.add_argument("--output", default="processed_datasets",
                        help="Output directory for CSV datasets (default: processed_datasets)")
    args = parser.parse_args()

    bag_dirs = find_rosbag_directories(args.input)

    if not bag_dirs:
        print(f"Error: No bag directories found in '{args.input}'.")
        exit(1)

    print(f"Found {len(bag_dirs)} bag(s). Starting processing...\n")
    os.makedirs(args.output, exist_ok=True)

    all_clean_dfs = []
    generated_csv_files = []

    for idx, bag_path in enumerate(bag_dirs, 1):
        bag_name = os.path.basename(os.path.normpath(bag_path))
        print(f"[{idx}/{len(bag_dirs)}] {bag_name}")

        try:
            df_raw = parse_mcgill_mcap(bag_path)
        except Exception as e:
            print(f"  Error reading bag '{bag_name}': {e}. Skipping...\n")
            continue

        if len(df_raw) == 0:
            print(f"  Warning: No valid data. Skipping...\n")
            continue

        duration_sec = df_raw['time'].iloc[-1] - df_raw['time'].iloc[0]
        if duration_sec < MIN_DURATION_SECONDS:
            print(f"  Warning: Too short ({duration_sec:.1f}s < {MIN_DURATION_SECONDS}s). Skipping...\n")
            continue

        df_clean = apply_kinematic_filters(df_raw)

        csv_filename = f"{bag_name}_dataset.csv"
        export_path = os.path.join(args.output, csv_filename)
        df_clean.to_csv(export_path, index=False)

        all_clean_dfs.append(df_clean)
        generated_csv_files.append(export_path)
        print(f"  Saved {len(df_clean)} frames to {export_path}\n")

        # Optional diagnostic plots
        if SAVE_PLOTS or SHOW_PLOTS:
            t_zeroed = df_clean['time'] - df_clean['time'].iloc[0]

            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(t_zeroed, df_clean['u'], label='u [m/s]', color='blue')
            if 'Fx' in df_clean.columns:
                plt.plot(t_zeroed, df_clean['Fx'] / 50.0, label='Fx [N/50]', color='green', alpha=0.6)
            plt.title(f"{bag_name} - Velocity vs Commanded Force")
            plt.ylabel("m/s or scaled N")
            plt.legend()
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.plot(t_zeroed, df_clean['u_dot'], label='u_dot [m/s²]', color='red')
            plt.xlabel("Time (s)")
            plt.ylabel("m/s²")
            plt.legend()
            plt.grid()

            plt.tight_layout()

            if SAVE_PLOTS:
                plot_path = os.path.join(args.output, f"{bag_name}_plot.png")
                plt.savefig(plot_path, dpi=150)

            if SHOW_PLOTS:
                plt.show()
            else:
                plt.close()

    if all_clean_dfs:
        combined_df = pd.concat(all_clean_dfs, ignore_index=True)
        master_path = os.path.join(args.output, "combined_master_dataset.csv")
        combined_df.to_csv(master_path, index=False)

        print("=" * 50)
        print("Processing complete.")
        print(f"Total frames across {len(all_clean_dfs)} run(s): {len(combined_df)}")
        print(f"Master dataset saved to: {master_path}")
    else:
        print("\nNo valid datasets were produced.")