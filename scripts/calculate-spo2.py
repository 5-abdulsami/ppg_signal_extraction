#!/usr/bin/python3
# calculate-spo2.py - Computes SpO2 from Red and Blue signals

import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt

def get_ac_dc(signal, sampling_rate=30):
    # DC component (average)
    dc = np.mean(signal)
    
    # AC component (pulsatile)
    nyquist = 0.5 * sampling_rate
    b, a = butter(4, [0.5/nyquist, 4.0/nyquist], btype='band')
    filtered = filtfilt(b, a, signal - dc)
    
    # Use standard deviation as a proxy for AC amplitude
    ac = np.std(filtered)
    return ac, dc

def main():
    path = '../data/df-ac-measurements-2.csv'
    if not os.path.exists(path):
        print("CSV file not found.")
        return

    df = pd.read_csv(path)
    
    print(f"\n{'ID':<5} | {'R-Ratio':<10} | {'SpO2 Status'}")
    print("-" * 35)

    for idx, row in df.iterrows():
        # Filter columns for Red (rx) and Blue (bx)
        r_data = [row[c] for c in row.index if c.startswith('rx')]
        b_data = [row[c] for c in row.index if c.startswith('bx')]
        
        if not r_data or not b_data:
            continue

        ac_r, dc_r = get_ac_dc(np.array(r_data))
        ac_b, dc_b = get_ac_dc(np.array(b_data))
        
        # Ratio of Ratios formula
        # R = (AC_red / DC_red) / (AC_blue / DC_blue)
        R = (ac_r / dc_r) / (ac_b / dc_b)
        
        # Linear approximation: SpO2 = 105 - 25*R
        A = 105.0 
        B = 25.0

        spo2 = A - (B * R)
        spo2 = np.clip(spo2, 70, 100)

        print(f"{row['id']:<5} | {R:<10.3f} | {spo2:.1f}%")

if __name__ == "__main__":
    main()