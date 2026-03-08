# WiFi Human Presence Detection System

A software-only system that detects human presence by analysing
variations in WiFi RSSI (Received Signal Strength Indicator) signals
using signal processing and machine learning.

---

## How It Works

```
WiFi APs → RSSI Samples → Sliding Window → Feature Extraction → ML Classifier → Presence / No Presence
```

When a person moves in a room, their body absorbs and reflects 2.4/5 GHz
radio waves, causing measurable fluctuations in RSSI. The system learns
to distinguish these fluctuation patterns from the ambient baseline.

### Features Extracted (per AP, per window)
| Category    | Features |
|-------------|----------|
| Statistical | mean, std, min, max, range, median, IQR |
| Shape       | skewness, kurtosis |
| Dynamics    | mean-absolute-diff (MAD), zero-crossing rate, RMS |
| Spectral    | dominant-frequency power (FFT) |
| Aggregate   | mean_std_all, max_range_all |

---

## Project Structure

```
wifi_presence/
├── main.py          # CLI entry point
├── config.py        # All tunable parameters
├── scanner.py       # Live WiFi scanning + RSSISimulator
├── collector.py     # Labeled data collection
├── features.py      # Feature extraction pipeline
├── trainer.py       # Model training & evaluation
├── monitor.py       # Real-time inference loop
├── requirements.txt
├── data/            # Saved CSV sessions
├── models/          # Serialized model + scaler
└── logs/            # Training plots (PNG)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Full demo (simulated WiFi — no hardware needed)
```bash
python main.py demo
```

### 3. Step-by-step (simulated)
```bash
# Collect data (2 classes × 3 sessions × 30 s each)
python main.py collect --sessions 3 --duration 30

# Train Random Forest
python main.py train --model rf

# Monitor for 60 s (simulate a person present)
python main.py monitor --duration 60 --person
```

### 4. Real hardware (Linux / macOS with WiFi)
```bash
# Requires root / iwlist access on Linux
python main.py collect --live --sessions 5 --duration 60
python main.py train
python main.py monitor --live
```

---

## CLI Reference

| Command | Options | Description |
|---------|---------|-------------|
| `collect` | `--sessions N`, `--duration SEC`, `--live` | Record labeled sessions |
| `train`   | `--model rf\|gb\|svm`, `--no-plot` | Train & evaluate model |
| `monitor` | `--duration SEC`, `--live`, `--person` | Live inference |
| `demo`    | `--duration SEC` | End-to-end simulated demo |

---

## Configuration (`config.py`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `SCAN_INTERVAL_SEC` | 0.5 | Seconds between RSSI samples |
| `COLLECTION_DURATION` | 30 | Seconds per labeled session |
| `WINDOW_SIZE` | 20 | Samples per feature window |
| `WINDOW_STEP` | 10 | Sliding-window stride (50 % overlap) |

---

## Models Supported

| Key | Algorithm | Notes |
|-----|-----------|-------|
| `rf` | Random Forest (default) | Fast, interpretable, feature importance |
| `gb` | Gradient Boosting | Higher accuracy, slower training |
| `svm` | SVM (RBF kernel) | Good on small datasets |

---

## Extending the System

- **More APs** — place additional routers / extenders; more APs = richer features.
- **CSI instead of RSSI** — with compatible hardware (Intel 5300, ESP32), replace
  `scanner.py` with a CSI reader for far greater sensitivity.
- **Deep learning** — swap the sklearn pipeline in `trainer.py` for an LSTM/CNN
  that operates directly on raw RSSI sequences.
- **Alerts** — extend `monitor.py` to send push notifications or MQTT messages.