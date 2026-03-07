#!/usr/bin/python3
# calculate-spo2.py - Computes SpO2 from Red and Green signals (camera PPG)

import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt, savgol_filter

SAMPLING_RATE = 30


def bandpass_filter(signal, fs=30):
    nyquist = 0.5 * fs
    low = 0.7 / nyquist
    high = 3.0 / nyquist
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, signal)


def get_ac_dc(signal):

    signal = np.array(signal)

    # DC component
    dc = np.mean(signal)

    # remove DC drift
    signal = signal - dc

    # bandpass filter
    filtered = bandpass_filter(signal, SAMPLING_RATE)

    # smooth signal
    filtered = savgol_filter(filtered, 11, 3)

    # AC amplitude using peak-to-peak
    ac = (np.max(filtered) - np.min(filtered)) / 2.0

    return ac, dc, filtered


def signal_quality(signal):

    if len(signal) < 200:
        return False

    if np.std(signal) < 0.2:
        return False

    return True


def calculate_spo2(R):

    # Empirical calibration (typical camera-PPG mapping)
    A = 104
    B = 17

    spo2 = A - (B * R)

    return np.clip(spo2, 70, 100)


def main():

    path = "../data/ppg-measurements.csv"

    if not os.path.exists(path):
        print("CSV file not found.")
        return

    df = pd.read_csv(path)

    print(f"\n{'ID':<5} | {'R Ratio':<8} | {'SpO2':<6} | Status")
    print("-" * 40)

    for idx, row in df.iterrows():

        r_data = [row[c] for c in row.index if c.startswith("rx")]
        g_data = [row[c] for c in row.index if c.startswith("gx")]

        if not r_data or not g_data:
            continue

        r_data = np.array(r_data)
        g_data = np.array(g_data)

        if not signal_quality(r_data) or not signal_quality(g_data):
            print(f"{row['id']:<5} | ----     | ----  | Low signal quality")
            continue

        ac_r, dc_r, r_filtered = get_ac_dc(r_data)
        ac_g, dc_g, g_filtered = get_ac_dc(g_data)

        if dc_r == 0 or dc_g == 0:
            print(f"{row['id']:<5} | ----     | ----  | Invalid signal")
            continue

        # Ratio of Ratios
        R = (ac_r / dc_r) / (ac_g / dc_g)

        if R <= 0 or R > 3:
            print(f"{row['id']:<5} | {R:<8.3f} | ----  | Unstable R")
            continue

        spo2 = calculate_spo2(R)

        print(f"{row['id']:<5} | {R:<8.3f} | {spo2:.1f}% | OK")


if __name__ == "__main__":
    main()
    