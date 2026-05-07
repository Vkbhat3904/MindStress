# Test Samples — Stress Predictor API

This folder contains three pre-built JSON signal samples for testing the Stress Predictor API without needing live EEG/ECG hardware.

## Files

| File | Stress Level | HR | SDNN | LF/HF |
|------|-------------|-----|------|-------|
| `low_stress_sample.json` | 🟢 Low | 68 bpm | 62 ms | 0.85 |
| `mild_stress_sample.json` | 🟡 Mild | 82 bpm | 45 ms | 1.40 |
| `high_stress_sample.json` | 🔴 High | 102 bpm | 28 ms | 2.20 |

## Format

Each file contains a JSON object with a `features` array of **101 floats**:
- Indices **0–31**: EEG Theta band power (32 channels)
- Indices **32–63**: EEG Alpha band power (32 channels)
- Indices **64–95**: EEG Beta band power (32 channels)
- Index **96**: Frontal Alpha Asymmetry (F4 – F3)
- Index **97**: Heart Rate Mean (bpm)
- Index **98**: SDNN (ms)
- Index **99**: RMSSD (ms)
- Index **100**: LF/HF Ratio

## Usage

### From the Dashboard
Open `http://127.0.0.1:8000` in your browser, then click **"Load & Predict"** on any sample card.

### Via curl
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @low_stress_sample.json
```

### Via Python
```python
import json, requests
with open("low_stress_sample.json") as f:
    data = json.load(f)
r = requests.post("http://127.0.0.1:8000/predict", json=data)
print(r.json())
```
