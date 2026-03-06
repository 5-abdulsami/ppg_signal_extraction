#!/usr/bin/python3
# calculate-hr.py
# Advanced Heart Rate Estimation from Camera PPG

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter, welch

SAMPLING_RATE = 30


# -------------------------------------------------------
# Load CSV
# -------------------------------------------------------
def load_measurements(filename="../data/ppg-measurements.csv"):

    if not os.path.exists(filename):
        print(f"Error: File {filename} not found")
        return None

    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} measurements")

    return df


# -------------------------------------------------------
# Signal preprocessing
# -------------------------------------------------------
def preprocess_signal(signal, sampling_rate=30):

    signal = np.array(signal)
    signal = signal[~np.isnan(signal)]

    if len(signal) < 60:
        return None

    # detrend
    x = np.arange(len(signal))
    coeff = np.polyfit(x, signal, 1)
    trend = np.polyval(coeff, x)
    detrended = signal - trend

    # normalize
    normalized = (detrended - np.mean(detrended)) / np.std(detrended)

    # bandpass filter
    nyquist = sampling_rate / 2
    low = 0.7 / nyquist
    high = 3.0 / nyquist

    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, normalized)

    # smoothing (important)
    filtered = savgol_filter(filtered, 11, 3)

    return filtered


# -------------------------------------------------------
# Signal quality check
# -------------------------------------------------------
def signal_quality(signal):

    variance = np.std(signal)

    if variance < 0.3:
        return False

    return True


# -------------------------------------------------------
# Time-domain HR estimation
# -------------------------------------------------------
def hr_time_domain(signal, sampling_rate=30):

    prominence = np.std(signal) * 0.5

    peaks, _ = find_peaks(signal, distance=sampling_rate * 0.4, prominence=prominence)

    if len(peaks) < 2:
        return None, peaks

    intervals = np.diff(peaks) / sampling_rate

    hr = 60 / np.mean(intervals)

    return hr, peaks


# -------------------------------------------------------
# Frequency-domain HR (Welch PSD)
# -------------------------------------------------------
def hr_frequency_domain(signal, sampling_rate=30):

    freqs, power = welch(signal, fs=sampling_rate, nperseg=256)

    mask = (freqs >= 0.7) & (freqs <= 3)

    freqs = freqs[mask]
    power = power[mask]

    if len(freqs) == 0:
        return None

    dominant = freqs[np.argmax(power)]

    return dominant * 60


# -------------------------------------------------------
# WINDOWED HR ESTIMATION
# (major accuracy improvement)
# -------------------------------------------------------
def windowed_hr(signal, sampling_rate=30):

    window_size = 8 * sampling_rate
    stride = 2 * sampling_rate

    hrs = []

    for start in range(0, len(signal) - window_size, stride):
        window = signal[start : start + window_size]

        hr_t, _ = hr_time_domain(window, sampling_rate)
        hr_f = hr_frequency_domain(window, sampling_rate)

        if hr_t and hr_f:
            # fusion of both methods
            hr = (hr_t + hr_f) / 2

            if 40 < hr < 200:
                hrs.append(hr)

    if len(hrs) == 0:
        return None

    return np.median(hrs)


# -------------------------------------------------------
# Plot PPG signal
# -------------------------------------------------------
def plot_signal(signal, peaks, hr, measurement_id):

    t = np.arange(len(signal)) / SAMPLING_RATE

    plt.figure(figsize=(12, 6))

    plt.plot(t, signal, label="Filtered PPG")

    if peaks is not None:
        plt.plot(peaks / SAMPLING_RATE, signal[peaks], "ro")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.title(f"PPG Signal | HR {hr:.1f} BPM | ID {measurement_id}")

    plt.grid(True)
    plt.legend()

    os.makedirs("../data/output", exist_ok=True)

    file = f"../data/output/ppg_hr_{measurement_id}.png"

    plt.savefig(file)
    plt.show()

    print("Saved:", file)


# -------------------------------------------------------
# Analyze measurements
# -------------------------------------------------------
def analyze_measurement(df, measurement_id=None):

    if measurement_id is not None:
        mask = df["id"] == str(measurement_id)
        measurements = df[mask]

    else:
        measurements = df

    results = []

    for idx, row in measurements.iterrows():
        # USE GREEN CHANNEL
        signal_cols = [c for c in row.index if c.startswith("gx")]

        if len(signal_cols) == 0:
            print("No signal columns found")
            continue

        signal = row[signal_cols].values.astype(float)

        pid = row.get("id", idx)
        ref_hr = row.get("hr", None)

        print("\n=================================")
        print("Measurement ID:", pid)

        signal = preprocess_signal(signal)

        if signal is None:
            print("Signal preprocessing failed")
            continue

        if not signal_quality(signal):
            print("Low signal quality")
            continue

        hr = windowed_hr(signal)

        if hr is None:
            print("HR estimation failed")
            continue

        hr_peaks, peaks = hr_time_domain(signal)

        print("Estimated HR:", round(hr, 1), "BPM")

        if ref_hr not in [None, "N/A"]:
            try:
                ref = float(ref_hr)
                err = abs(ref - hr)
                print("Reference:", ref, "Error:", round(err, 1))
            except:
                pass

        if measurement_id is not None:
            plot_signal(signal, peaks, hr, pid)

        results.append(
            {
                "id": pid,
                "estimated_hr": hr,
                "reference_hr": ref_hr,
                "signal_length": len(signal),
                "num_peaks": len(peaks) if peaks is not None else 0,
            }
        )

    return results


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():

    measurement_id = None

    if len(sys.argv) > 1:
        measurement_id = sys.argv[1]

    df = load_measurements()

    if df is None:
        return

    results = analyze_measurement(df, measurement_id)

    if results:
        os.makedirs("../data/output", exist_ok=True)

        out = "../data/output/hr_results.csv"

        pd.DataFrame(results).to_csv(out, index=False)

        print("\nResults saved:", out)


if __name__ == "__main__":
    main()
