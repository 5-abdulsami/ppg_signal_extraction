# PPG based Vital Health Monitoring

## Purpose
Extract photoplethysmogram (PPG) signals from camera recordings, save measurements to CSV, and compute heart rate and SpO2 estimates.

## Steps
1. Install Python dependencies and system `ffmpeg`
2. Extract frames from video (or provide frames)
3. Run recording script to produce CSV measurement(s)
4. Run heart-rate and SpO2 analyzers

## Project Structure
- **Scripts**: `calculate-hr.py`, `calculate-spo2.py`, `extract-frames.sh`, `record-ppg.py`
- **Data**: `data` — contains recordings, extracted frames, measurements and outputs
- **Outputs**: `output` — plots and `heart_rate_results.csv`

## Dependencies

### Python Packages
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
scipy>=1.7
pillow>=8.0
scikit-learn>=1.0
```

### System
```bash
# Ubuntu example
sudo apt update
sudo apt install ffmpeg
```

## Files & What They Do

### `extract-frames.sh`
- Uses `ffmpeg` to convert video into image frames (30 fps default)
- Default target paths relative to repo

### `record-ppg.py`
- Reads PNG frames (expected naming: `out<n>.png`)
- Computes mean red-channel intensity per frame for PPG waveform
- Saves measurement to `../data/df-ac-measurements.csv`
- Produces preview plot at `../data/output/ppg_waveform.png`

### `calculate-hr.py`
- Loads measurements from `../data/df-ac-measurements.csv`
- Computes heart rate via:
  - Time-domain peak detection
  - Frequency-domain analysis (FFT)
- Preprocessing: detrending, normalization, Butterworth bandpass (0.5–4 Hz)
- Saves results to `../data/output/heart_rate_results.csv`

### `calculate-spo2.py`
- Expects CSV with Red and Blue channel pairs
- Computes AC/DC ratio, estimates SpO2 using linear approximation
- Default input: `../data/df-ac-measurements-2.csv`

## Data Formats

### Single-channel PPG CSV
Header: `id,date,sys,dia,hr,x0,x1,...,xN`
- Each `x*` = per-frame mean intensity
- Example: `ppg-measurements.csv`

### Dual-channel AC CSV (SpO2)
Header: `id,date,sys,dia,hr,rx0,rx1,...,bx0,bx1,...`
- `rx*` = red-channel samples
- `bx*` = blue-channel samples

### Output CSV
`heart_rate_results.csv` contains:
- `time_domain_hr`, `freq_domain_hr`, `average_hr`, `reference_hr`, `num_peaks`, `signal_length`

## Usage Examples

### 1. Extract frames from video
```bash
bash scripts/extract-frames.sh
# or custom command
ffmpeg -i data/recordings/input.mp4 -vf fps=30 data/frames/out%d.png
```

### 2. Create/save measurement from frames
```bash
python3 scripts/record-ppg.py
```

### 3. Compute heart rate (all measurements)
```bash
python3 scripts/calculate-hr.py
```

### 4. Compute heart rate for specific measurement
```bash
python3 scripts/calculate-hr.py 0
```

### 5. Compute SpO2
```bash
python3 scripts/calculate-spo2.py
```

## Configuration Notes
- **Sampling rate**: Default 30 Hz. Update in scripts if different.
- **File paths**: Scripts use relative paths. Run from `scripts` folder.
- **Headless systems**: Set matplotlib backend to `Agg` or skip `show()`.

## Troubleshooting
- **Missing ffmpeg**: Install system package
- **ImportError**: Run `pip install -r requirements.txt`
- **Few peaks**: Check signal quality and exposure settings
- **CSV mismatches**: Verify header matches script expectations

## Outputs & Diagnostics
- Waveform preview: `data/output/ppg_waveform.png`
- HR results: `heart_rate_results.csv`
- Frequency and signal plots saved per-measurement in `output/`

## Contributing
- Video format support: Update `extract-frames.sh`
- Improved algorithms: Modify `calculate-hr.py` and `calculate-spo2.py`