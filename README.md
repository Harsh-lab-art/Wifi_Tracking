<div align="center">

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

# 📡 WiFiPresence

### Software-Based Human Presence Detection via WiFi Signal Analysis

> *Detect whether a person is in a room — using only the WiFi router already on your desk.*

**95%+ accuracy · No extra hardware · Works with any router · Real-time inference**

</div>

---

## 🧠 The Idea

When a person walks into a room, their body absorbs and reflects 2.4 GHz and 5 GHz radio waves. This causes tiny but measurable fluctuations in the **Received Signal Strength Indicator (RSSI)** of nearby WiFi access points.

WiFiPresence captures these fluctuations, extracts statistical and spectral features from a sliding time window, and feeds them into a trained machine learning classifier — distinguishing *"someone is here"* from *"the room is empty"* in real time.

```
No cameras. No microphones. No special sensors.
Just the WiFi infrastructure you already have.
```

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📶 **Multi-AP support** | Reads RSSI from all visible access points simultaneously |
| 🪟 **Sliding window pipeline** | 20-sample windows with 50% overlap for temporal context |
| 📊 **13 features per AP** | Statistical, shape, dynamics, and spectral (FFT) features |
| 🤖 **3 model choices** | Random Forest, Gradient Boosting, or SVM |
| 🎯 **95%+ CV accuracy** | On simulated data with realistic physics-based noise |
| 🔴 **Real-time monitor** | Live confidence bar with majority-vote smoothing |
| 🧪 **Built-in simulator** | Physics-based RSSI generator — no WiFi hardware needed to train |
| 📈 **Auto-plots** | Confusion matrix, ROC curve, feature importances saved to `logs/` |

---

## 🗂️ Project Structure

```
wifi_presence/
│
├── main.py           ← CLI entry point  (collect / train / monitor / demo)
├── config.py         ← All tunable parameters in one place
│
├── scanner.py        ← Live WiFi RSSI reader + RSSISimulator
├── collector.py      ← Records labeled sessions to CSV
├── features.py       ← Sliding-window feature extraction
├── trainer.py        ← Model training, cross-validation & evaluation plots
├── monitor.py        ← Real-time inference loop
│
├── data/             ← Saved CSV sessions (auto-created)
├── models/           ← Serialized model (auto-created)
├── logs/             ← Training plots PNG (auto-created)
│
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full demo (no hardware needed)

```bash
python main.py demo
```

This runs the complete pipeline automatically:
1. **Collects** simulated RSSI data for both classes
2. **Trains** a Random Forest classifier
3. **Monitors** in real time (simulated person present)

Expected output:

```
══════════════════════════════════════════════════════════
   WiFi Human Presence Monitor — LIVE
══════════════════════════════════════════════════════════

  🔴  Person Present   Conf: [████████████████████░░░░░░░░░]  82.4%   APs: 4
```

---

## 🛠️ Step-by-Step Usage

### Collect labeled data

```bash
# Simulated (no hardware) — 3 sessions × 30s per class
python main.py collect --sessions 3 --duration 30

# Real WiFi hardware (Linux — requires iwlist/root)
python main.py collect --live --sessions 5 --duration 60
```

During real collection you will be prompted to set up the scene before each session (person present / no person).

### Train the model

```bash
# Random Forest (default, recommended)
python main.py train --model rf

# Gradient Boosting (higher accuracy, slower)
python main.py train --model gb

# SVM (great for small datasets)
python main.py train --model svm

# Skip plots
python main.py train --no-plot
```

Training generates three plots in `logs/`:

| File | Contents |
|---|---|
| `confusion_matrix.png` | True vs predicted labels |
| `feature_importance.png` | Top 20 most predictive features |
| `roc_curve.png` | AUC-ROC curve |

### Monitor in real time

```bash
# Simulated — person present
python main.py monitor --duration 60 --person

# Simulated — empty room
python main.py monitor --duration 60

# Live WiFi hardware
python main.py monitor --live --duration 120
```

---

## ⚙️ Configuration

All parameters live in `config.py` — change them to tune performance:

```python
SCAN_INTERVAL_SEC   = 0.5   # seconds between RSSI samples
COLLECTION_DURATION = 30    # seconds per labeled session
WINDOW_SIZE         = 20    # samples per feature window
WINDOW_STEP         = 10    # sliding-window stride (50% overlap)
CV_FOLDS            = 5     # cross-validation folds
TEST_SIZE           = 0.2   # train/test split ratio
```

**Tuning tips:**
- More data → more sessions (`--sessions 5+`)
- Noisier environment → increase `WINDOW_SIZE`
- Faster response → decrease `WINDOW_SIZE` and `SCAN_INTERVAL_SEC`

---

## 🔬 How Features Are Extracted

For each AP and each window of `WINDOW_SIZE` samples:

```
Signal window [s₁, s₂, ..., s₂₀]
        │
        ├── Statistical  →  mean, std, min, max, range, median, IQR
        ├── Shape        →  skewness, kurtosis
        ├── Dynamics     →  mean absolute diff (MAD), zero-crossing rate, RMS
        └── Spectral     →  dominant frequency power via FFT
```

Plus two **aggregate** features across all APs:
- `agg__mean_std` — average signal variance across APs
- `agg__max_range` — maximum signal swing across APs

With 4 APs, this gives **54 features per window**.

---

## 🧪 The Simulator

Training on real hardware takes time and requires careful scene setup. The built-in `RSSISimulator` lets you generate realistic RSSI data instantly:

**Physics model:**
- **Ornstein-Uhlenbeck process** — slow signal drift toward a base RSSI (realistic path-loss + shadowing)
- **Human multipath bursts** — random ±3–8 dBm spikes (35% probability per sample) when `person_present=True`
- **Breathing / micro-movement** — sinusoidal ±1.5 dBm oscillation (12-sample period)

```python
from scanner import RSSISimulator

sim = RSSISimulator(person_present=True)
aps = sim.scan()   # returns List[AccessPoint] with realistic RSSI values
```

---

## 📡 Real Hardware Setup (Linux)

```bash
# Check your wireless interface name
iwconfig

# Test manual scan (requires root)
sudo iwlist wlan0 scan | grep -E "ESSID|Signal"

# Run collector with live hardware
sudo python main.py collect --live
```

> **macOS:** Uses the `airport` utility automatically — no sudo required.
>
> **Windows:** Not currently supported. Use WSL2 with a USB WiFi adapter for best results.

---

## 🗺️ Roadmap

- [ ] **CSI support** — Channel State Information (Intel 5300, ESP32) for 10× more signal detail
- [ ] **Multi-zone detection** — triangulate *which room* using multiple APs
- [ ] **Activity recognition** — classify sitting / walking / exercising from signal patterns  
- [ ] **LSTM model** — end-to-end deep learning directly on raw RSSI sequences
- [ ] **IoT / MQTT alerts** — push notifications when occupancy changes
- [ ] **Web dashboard** — live chart of signal variance + presence confidence over time

---

## 📦 Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

Made by **Harsh Kumar Verma**

[harsh9760verma@gmail.com](mailto:harsh9760verma@gmail.com) · [github.com/Harsh-lab-art](https://github.com/Harsh-lab-art)

*"The best sensor is the one already in the room."*

</div>
