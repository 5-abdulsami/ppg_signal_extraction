#!/usr/bin/python3
# record-ppg.py
# Extracts RGB PPG signals from fingertip video frames
# Stores signals + metadata for HR / SpO2 / BP estimation

import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

FRAME_DIR = "../data/frames/"
OUTPUT_DIR = "../data/output/waveforms"
CSV_FILE = "../data/ppg-measurements.csv"


def get_image_rgb(image_path):
    image = Image.open(image_path)

    red, green, blue = image.split()

    r_mean = np.mean(np.array(red))
    g_mean = np.mean(np.array(green))
    b_mean = np.mean(np.array(blue))

    return r_mean, g_mean, b_mean


def is_valid_frame(current, previous, threshold=15):

    if previous is None:
        return True

    diff = abs(current - previous)

    if diff > threshold:
        return False

    return True


def get_signals():

    images = [img for img in os.listdir(FRAME_DIR) if img.endswith(".png")]

    images = sorted(images, key=lambda x: int(x[3:-4]))

    r_signal = []
    g_signal = []
    b_signal = []

    prev_brightness = None

    for img in images:

        image_path = os.path.join(FRAME_DIR, img)

        r, g, b = get_image_rgb(image_path)

        brightness = (r + g + b) / 3

        if not is_valid_frame(brightness, prev_brightness):
            print(f"Skipping noisy frame: {img}")
            continue

        prev_brightness = brightness

        r_signal.append(r)
        g_signal.append(g)
        b_signal.append(b)

        print(f"Processed frame: {img}")

    return r_signal, g_signal, b_signal


def compute_signal_features(signal):

    signal = np.array(signal)

    ac = np.std(signal)
    dc = np.mean(signal)

    ratio = ac / dc if dc != 0 else 0

    return ac, dc, ratio


def save_plots(r_sig, g_sig, b_sig, pid):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # RED channel PPG (experiment)
    plt.figure(figsize=(13, 6))
    plt.plot(r_sig, color="red", linewidth=1.5)

    plt.title(f"PPG Signal (Red Channel) - ID: {pid}")
    plt.ylabel("Mean Intensity")
    plt.xlabel("Frame Number")

    plt.savefig(f"{OUTPUT_DIR}/ppg_red_{pid}.png")

    plt.close()

    # RGB comparison plot
    plt.figure(figsize=(13, 6))

    plt.plot(r_sig, color="red", label="Red")
    plt.plot(g_sig, color="green", label="Green")
    plt.plot(b_sig, color="blue", label="Blue")

    plt.title(f"RGB PPG Channels - ID: {pid}")

    plt.legend()

    plt.savefig(f"{OUTPUT_DIR}/ppg_rgb_{pid}.png")

    plt.show()


def signal_quality_check(signal):

    signal = np.array(signal)

    variance = np.std(signal)

    if variance < 0.5:
        print("Warning: Low signal variance. Possible poor finger contact.")

    return variance


if __name__ == "__main__":

    print("\nExtracting PPG signals from frames...\n")

    r_sig, g_sig, b_sig = get_signals()

    print("\nFrames processed:", len(r_sig))

    if len(r_sig) < 100:
        print("Warning: Signal too short for reliable analysis.")

    # Preview RED PPG
    plt.plot(r_sig, color="red")
    plt.title("PPG Signal Preview (Red Channel)")
    plt.show()

    signal_quality_check(r_sig)

    if input("\nSave measurement? (y/n): ").lower() != "y":
        exit()

    pid = input("Enter ID: ")

    save_plots(r_sig, g_sig, b_sig, pid)

    ac_r, dc_r, ratio_r = compute_signal_features(r_sig)
    ac_g, dc_g, ratio_g = compute_signal_features(g_sig)

    print("\nSignal Features:")
    print("Red AC/DC:", ratio_r)
    print("Green AC/DC:", ratio_g)

    header = ["id", "date", "sys", "dia", "hr"]

    header += ["ac_red", "dc_red", "ac_green", "dc_green"]

    header += [f"rx{i}" for i in range(len(r_sig))]
    header += [f"gx{i}" for i in range(len(g_sig))]
    header += [f"bx{i}" for i in range(len(b_sig))]

    if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
        with open(CSV_FILE, "w") as f:
            csv.writer(f).writerow(header)

    timestr = time.strftime("%Y-%m-%d-%H:%M:%S")

    sys = input("SYS BP: ")
    dia = input("DIA BP: ")
    ref_hr = input("Reference HR: ")

    row = (
        [
            pid,
            timestr,
            sys,
            dia,
            ref_hr,
            ac_r,
            dc_r,
            ac_g,
            dc_g,
        ]
        + r_sig
        + g_sig
        + b_sig
    )

    with open(CSV_FILE, "a") as f:
        csv.writer(f).writerow(row)

    print(f"\nSuccess: Measurement {pid} saved to {CSV_FILE}")