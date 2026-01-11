#!/usr/bin/python3
# record-ppg.py - Combined HR and SpO2 signal extraction

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import os, csv

def get_image_rgb(image_path):
    image = Image.open(image_path)
    red, green, blue = image.split()
    r_mean = np.mean(np.array(red))
    b_mean = np.mean(np.array(blue))
    return r_mean, b_mean

def get_signals():
    images = os.listdir('../data/frames/')
    # Ensures frames are processed in numerical order (out1, out2, etc.)
    numbers = sorted([int(img[3:][:-4]) for img in images if img.endswith('.png')])
    
    r_signal, b_signal = [], []
    for n in numbers:
        image_path = f'../data/frames/out{n}.png'
        print(f'Reading frame: {n}')
        r, b = get_image_rgb(image_path)
        r_signal.append(r)
        b_signal.append(b)
    return r_signal, b_signal

def save_plots(r_sig, b_sig, pid):
    output_dir = '../data/output/waveforms'
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Standard PPG (Red Channel) for Heart Rate Peaks
    plt.figure(figsize=(13, 6))
    plt.plot(r_sig, color='red', linewidth=1.5)
    plt.title(f'PPG Red Channel (HR Analysis) - ID: {pid}')
    plt.ylabel('Mean Intensity')
    plt.xlabel('Frame Number')
    plt.savefig(f'{output_dir}/ppg_hr_peaks_{pid}.png')
    plt.close() # Close to save memory

    # Plot 2: Dual Channel (Red vs Blue) for SpO2 Analysis
    plt.figure(figsize=(13, 6))
    plt.plot(r_sig, color='red', label='Red Channel')
    plt.plot(b_sig, color='blue', label='Blue Channel')
    plt.title(f'Dual Channel PPG (SpO2 Analysis) - ID: {pid}')
    plt.legend()
    plt.savefig(f'{output_dir}/ppg_spo2_dual_{pid}.png')
    plt.show() # Shows the dual plot to the user

if __name__ == "__main__":
    r_sig, b_sig = get_signals()
    
    # Show a quick preview of the Red channel for verification
    plt.plot(r_sig, color='red')
    plt.title("Signal Preview (Close to continue)")
    plt.show()

    if input('Save measurement? (y/n): ').lower() == 'y':
        pid = input('Enter ID: ')
        save_plots(r_sig, b_sig, pid)

        filename = '../data/ppg-measurements.csv'
        header = ['id', 'date', 'sys', 'dia', 'hr']
        header += [f'rx{i}' for i in range(len(r_sig))]
        header += [f'bx{i}' for i in range(len(b_sig))]

        if not os.path.exists(filename) or os.stat(filename).st_size == 0:
            with open(filename, 'w') as f:
                csv.writer(f).writerow(header)

        timestr = time.strftime("%Y-%m-%d-%H:%M:%S")
        sys, dia, ref_hr = input('SYS: '), input('DIA: '), input('Ref HR: ')

        row = [pid, timestr, sys, dia, ref_hr] + r_sig + b_sig
        with open(filename, 'a') as f:
            csv.writer(f).writerow(row)
        print(f"Success: ID {pid} saved to {filename}")