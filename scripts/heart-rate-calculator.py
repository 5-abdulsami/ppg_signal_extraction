#!/usr/bin/python3
# heart-rate-calculator.py - Calculates heart rate from saved PPG measurements
# Usage: python3 heart-rate-calculator.py [measurement_id]
#   If measurement_id is provided, calculates HR for that specific measurement
#   Otherwise, calculates HR for all measurements in the CSV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft, fftfreq
import sys
import os

def load_measurements(filename='../data/df-ac-measurements.csv'):
    """
    Load PPG measurements from CSV file
    """
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found")
        return None
    
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} measurements from {filename}")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def preprocess_signal(signal, sampling_rate=30):
    """
    Preprocess the PPG signal: detrend, normalize, and filter
    """
    # Convert to numpy array
    signal = np.array(signal)
    
    # Remove NaN values
    signal = signal[~np.isnan(signal)]
    
    if len(signal) < 10:  # Too short for processing
        return None
    
    # 1. Detrend (remove DC component)
    x = np.arange(len(signal))
    coeff = np.polyfit(x, signal, 1)
    trend = np.polyval(coeff, x)
    detrended = signal - trend
    
    # 2. Normalize
    normalized = (detrended - np.mean(detrended)) / np.std(detrended)
    
    # 3. Bandpass filter (0.5 Hz to 4 Hz for heart rate)
    nyquist = sampling_rate / 2
    low = 0.5 / nyquist
    high = 4.0 / nyquist
    
    # Butterworth filter
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, normalized)
    
    return filtered

def calculate_hr_time_domain(signal, sampling_rate=30, min_hr=40, max_hr=200):
    """
    Calculate heart rate using time-domain peak detection
    """
    # Find peaks in the filtered signal
    peaks, properties = find_peaks(signal, 
                                   distance=sampling_rate * 60 / max_hr,  # Minimum distance between peaks
                                   prominence=0.5)
    
    if len(peaks) < 2:
        print("Warning: Too few peaks detected")
        return None
    
    # Calculate peak-to-peak intervals in seconds
    peak_times = peaks / sampling_rate
    intervals = np.diff(peak_times)
    
    # Filter out unrealistic intervals (corresponding to < min_hr or > max_hr)
    valid_intervals = intervals[(intervals > 60/max_hr) & (intervals < 60/min_hr)]
    
    if len(valid_intervals) == 0:
        return None
    
    # Calculate average heart rate
    avg_interval = np.mean(valid_intervals)
    hr_bpm = 60 / avg_interval
    
    return hr_bpm, peaks

def calculate_hr_frequency_domain(signal, sampling_rate=30, min_hr=40, max_hr=200):
    """
    Calculate heart rate using frequency-domain (FFT) analysis
    """
    n = len(signal)
    
    # Apply FFT
    yf = fft(signal)
    xf = fftfreq(n, 1/sampling_rate)
    
    # Get only positive frequencies
    positive_freq_mask = xf >= 0
    xf = xf[positive_freq_mask]
    yf = np.abs(yf[positive_freq_mask])
    
    # Filter to heart rate range (convert BPM to Hz: BPM/60 = Hz)
    hr_mask = (xf >= min_hr/60) & (xf <= max_hr/60)
    
    if not np.any(hr_mask):
        return None
    
    xf_hr = xf[hr_mask]
    yf_hr = yf[hr_mask]
    
    # Find dominant frequency
    dominant_idx = np.argmax(yf_hr)
    dominant_freq = xf_hr[dominant_idx]
    
    # Convert frequency to BPM
    hr_bpm = dominant_freq * 60
    
    return hr_bpm, (xf, yf, dominant_freq)

def plot_signal_with_peaks(signal, peaks, sampling_rate=30, hr_bpm=None, measurement_id=None):
    """
    Plot the processed signal with detected peaks
    """
    time_axis = np.arange(len(signal)) / sampling_rate
    
    plt.figure(figsize=(12, 6))
    
    # Plot the signal
    plt.plot(time_axis, signal, 'b-', alpha=0.7, label='Filtered PPG')
    
    # Plot peaks
    if len(peaks) > 0:
        peak_times = peaks / sampling_rate
        plt.plot(peak_times, signal[peaks], 'r.', markersize=10, label='Detected Peaks')
    
    # Add labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Amplitude')
    
    title = 'PPG Signal with Detected Heartbeats'
    if hr_bpm:
        title += f' | HR: {hr_bpm:.1f} BPM'
    if measurement_id is not None:
        title += f' | Measurement ID: {measurement_id}'
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    output_dir = '../data/output'
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f'{output_dir}/ppg_with_peaks'
    if measurement_id is not None:
        filename += f'_id{measurement_id}'
    filename += '.png'
    
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {filename}")

def plot_frequency_analysis(xf, yf, dominant_freq, hr_bpm, measurement_id=None):
    """
    Plot frequency spectrum with dominant frequency highlighted
    """
    plt.figure(figsize=(12, 6))
    
    # Convert x-axis to BPM for easier interpretation
    xf_bpm = xf * 60
    
    plt.plot(xf_bpm, yf, 'b-', alpha=0.7, label='Frequency Spectrum')
    plt.axvline(x=hr_bpm, color='r', linestyle='--', 
                label=f'Dominant Frequency: {hr_bpm:.1f} BPM')
    
    plt.xlabel('Frequency (BPM)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Analysis of PPG Signal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    output_dir = '../data/output'
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f'{output_dir}/frequency_analysis'
    if measurement_id is not None:
        filename += f'_id{measurement_id}'
    filename += '.png'
    
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"Frequency plot saved to {filename}")

def analyze_measurement(df, measurement_id=None):
    """
    Analyze a specific measurement or all measurements
    """
    if measurement_id is not None:
        # Analyze specific measurement
        if 'id' in df.columns:
            mask = df['id'] == str(measurement_id)
            measurements = df[mask]
        else:
            print("Warning: 'id' column not found in CSV")
            measurements = df.iloc[[measurement_id]] if measurement_id < len(df) else None
        
        if len(measurements) == 0:
            print(f"Error: Measurement ID {measurement_id} not found")
            return
    else:
        # Analyze all measurements
        measurements = df
    
    results = []
    
    for idx, row in measurements.iterrows():
        # Extract signal data (columns starting with 'x')
        signal_columns = [col for col in row.index if col.startswith('x')]
        
        if not signal_columns:
            print(f"Warning: No signal data found for measurement at index {idx}")
            continue
        
        # Get signal values
        signal = row[signal_columns].values.astype(float)
        
        # Get metadata
        measurement_id_val = row.get('id', idx)
        device = row.get('device', 'unknown')
        sys_bp = row.get('sys', 'N/A')
        dia_bp = row.get('dia', 'N/A')
        ref_hr = row.get('hr', 'N/A')
        
        print(f"\n{'='*60}")
        print(f"Analyzing Measurement ID: {measurement_id_val}")
        print(f"Device: {device}, SYS: {sys_bp}, DIA: {dia_bp}, Reference HR: {ref_hr}")
        
        # Preprocess signal
        processed_signal = preprocess_signal(signal, sampling_rate=30)
        
        if processed_signal is None:
            print("Error: Signal preprocessing failed")
            continue
        
        # Calculate HR using both methods
        hr_time, peaks = calculate_hr_time_domain(processed_signal, sampling_rate=30)
        hr_freq, freq_data = calculate_hr_frequency_domain(processed_signal, sampling_rate=30)
        
        if hr_time and hr_freq:
            print(f"Time-domain HR: {hr_time:.1f} BPM")
            print(f"Frequency-domain HR: {hr_freq:.1f} BPM")
            
            # Use average of both methods
            hr_avg = (hr_time + hr_freq) / 2
            print(f"Average HR: {hr_avg:.1f} BPM")
            
            # Compare with reference if available
            if ref_hr != 'N/A':
                try:
                    ref_hr_float = float(ref_hr)
                    error = abs(hr_avg - ref_hr_float)
                    print(f"Reference HR: {ref_hr_float} BPM, Error: {error:.1f} BPM")
                except:
                    pass
            
            # Plot results for individual measurements
            if measurement_id is not None:
                plot_signal_with_peaks(processed_signal, peaks, sampling_rate=30, 
                                     hr_bpm=hr_avg, measurement_id=measurement_id_val)
                
                if freq_data:
                    xf, yf, dominant_freq = freq_data
                    plot_frequency_analysis(xf, yf, dominant_freq, hr_freq, 
                                          measurement_id=measurement_id_val)
            
            # Store results
            results.append({
                'id': measurement_id_val,
                'device': device,
                'time_domain_hr': hr_time,
                'freq_domain_hr': hr_freq,
                'average_hr': hr_avg,
                'reference_hr': ref_hr if ref_hr != 'N/A' else None,
                'num_peaks': len(peaks),
                'signal_length': len(signal)
            })
        else:
            print("Error: Could not calculate heart rate")
    
    # Print summary if analyzing all measurements
    if measurement_id is None and results:
        print(f"\n{'='*60}")
        print("SUMMARY OF ALL MEASUREMENTS")
        print(f"{'='*60}")
        print(f"{'ID':<5} {'Device':<15} {'Avg HR':<10} {'Ref HR':<10} {'Peaks':<10}")
        print(f"{'-'*60}")
        
        for res in results:
            ref_str = f"{res['reference_hr']:.1f}" if res['reference_hr'] else 'N/A'
            print(f"{res['id']:<5} {res['device'][:15]:<15} {res['average_hr']:<10.1f} "
                  f"{ref_str:<10} {res['num_peaks']:<10}")
    
    return results

def main():
    """
    Main function to run heart rate calculation
    """
    # Check if measurement ID is provided as command line argument
    measurement_id = None
    if len(sys.argv) > 1:
        try:
            measurement_id = sys.argv[1]
            # Try to convert to int if it's numeric
            if measurement_id.isdigit():
                measurement_id = int(measurement_id)
        except:
            print(f"Warning: Could not parse measurement ID '{sys.argv[1]}'")
            measurement_id = sys.argv[1]
    
    # Load measurements
    df = load_measurements()
    if df is None:
        return
    
    # Analyze measurements
    results = analyze_measurement(df, measurement_id)
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        output_file = '../data/output/heart_rate_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()