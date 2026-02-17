#!/usr/bin/env python3
# coding: utf-8
"""
Main script to launch multiple sbatch jobs for sweeping main_size values
for each r_candidate. Follows the project code style.
"""

import os
import sys
import subprocess
import time
import numpy as np

# Constants
R_CANDIDATES = [28.0001, 28.01, 28.1, 27.9, 27.99, 30, 38]
TOTAL_SIZE = 10000
N_POINTS = 24                 # number of main_size values per r_candidate
PRED_LEN = 10
GROUP_NAME = "size_sweep"     # ignored by submain.py, kept for consistency
ACCOUNT = "proj_1716"
SUBBASH_SCRIPT = "./subbash"  # assumes script is in the same directory
TEMP_R_DIR = "/home/ikvasilev/PaTHoP/assets/temp_r_files"
MAIN_R = 28.0                 # fixed target r


def generate_main_sizes(total, n):
    """
    Generate n integer main_size values evenly distributed from 0 to total.
    Uses np.linspace with rounding to keep values balanced.
    """
    sizes = np.linspace(0, total, num=n, endpoint=True)
    sizes = np.round(sizes).astype(int)
    return sizes.tolist()


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_r_file(r_value, main_size, directory):
    """
    Create a temporary file containing the single r_value.
    Returns the path to the created file.
    """
    # Replace decimal point with underscore for filesystem safety
    r_str = f"{r_value:.12f}".replace('.', '_')
    filename = f"r_{r_str}_size_{main_size}.txt"
    filepath = os.path.join(directory, filename)

    with open(filepath, 'w') as f:
        # Use the same precision as in submain.py when reading with np.loadtxt
        f.write(f"{r_value:.12f}\n")
    return filepath


def submit_job(main_r, main_size, r_file_path, pred_len, group_name):
    """
    Submit a single sbatch job using the provided parameters.
    The third argument (cand_size) is a placeholder and is ignored by submain.py.
    """
    cmd = (f"sbatch -A {ACCOUNT} {SUBBASH_SCRIPT} "
           f"{main_r} {main_size} 0 {group_name} {pred_len} {r_file_path}")
    print(f"Submitting: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                                capture_output=True, text=True)
        print(f"  Submitted: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"  Error submitting job: {e.stderr}", file=sys.stderr)


def main():
    # Ensure temporary directory exists
    ensure_dir(TEMP_R_DIR)

    # Generate main_size values
    main_sizes = generate_main_sizes(TOTAL_SIZE, N_POINTS)
    print(f"Generated {len(main_sizes)} main_size values: {main_sizes}")

    total_jobs = 0
    # Iterate over each r_candidate and each main_size
    for r_cand in R_CANDIDATES:
        for main_size in main_sizes:
            # Create a file with this single r_candidate
            r_file = create_r_file(r_cand, main_size, TEMP_R_DIR)

            # Submit the job
            submit_job(MAIN_R, main_size, r_file, PRED_LEN, GROUP_NAME)

            total_jobs += 1

            # Small delay to avoid overwhelming the job scheduler
            time.sleep(0.1)

    print(f"Total jobs submitted: {total_jobs}")


if __name__ == "__main__":
    main()