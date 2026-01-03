#!/usr/bin/python3
# record-ppg-2.py - Extracts Red and Blue PPG signals and stores them in CSV

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import os, csv

def get_rgb_means(image_path):
    '''
    Return mean intensity of Red and Blue channels
    '''
    image = Image.open(image_path)
    red, green, blue = image.split()
    # Get mean of the Red and Blue channels
    r_mean = np.mean(np.array(red))
    b_mean = np.mean(np.array(blue))
    return r_mean, b_mean

def get_signals():
    '''
    Return Red and Blue signals from image sequence
    '''
    images = os.listdir('../data/phone/')
    numbers = sorted([int(img[3:][:-4]) for img in images if img.endswith('.png')])
    
    r_signal = []
    b_signal = []
    
    for n in numbers:
        image_path = f'../data/phone/out{n}.png'
        print(f'Reading image: {image_path}')
        r, b = get_rgb_means(image_path)
        r_signal.append(r)
        b_signal.append(b)
        
    return r_signal, b_signal, numbers

if __name__ == "__main__":
    r_sig, b_sig, nums = get_signals()
    
    # Plotting Red signal for preview
    plt.figure(figsize=(12, 5))
    plt.plot(r_sig, color='red', label='Red Channel')
    plt.plot(b_sig, color='blue', label='Blue Channel')
    plt.legend()
    plt.show()

    if input('Save measurement? (y/n): ') == 'y':
        filename = '../data/df-ac-measurements-2.csv'
        
        # Determine column headers (rx0, rx1... and bx0, bx1...)
        header = ['id', 'date', 'sys', 'dia', 'hr']
        header += [f'rx{i}' for i in range(len(r_sig))]
        header += [f'bx{i}' for i in range(len(b_sig))]

        # File check / write header
        if not os.path.exists(filename) or os.stat(filename).st_size == 0:
            with open(filename, 'w') as f:
                csv.writer(f).writerow(header)

        pid = input('ID: ')
        timestr = time.strftime("%Y-%m-%d-%H:%M:%S")
        sys = input('SYS: ')
        dia = input('DIA: ')
        hr = input('HR: ')

        row = [pid, timestr, sys, dia, hr] + r_sig + b_sig
        with open(filename, 'a') as f:
            csv.writer(f).writerow(row)
        print(f"Data saved to {filename}")