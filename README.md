# ArthroScreen

A multimodal wearable for early screening of knee osteoarthritis (OA) — built for **Smart India Hackathon 2026**.

**Live demo:** https://harshraj1607.github.io/arthroscreen/ 
---

## The problem

Knee OA is usually caught late, once cartilage damage is already significant, because screening depends on subjective clinical exams or expensive imaging. ArthroScreen aims to flag at-risk knees early using a cheap, wearable sensor rig and an on-device ML model — cheap enough for primary-care and community screening camps.

## How it works

```
Acoustic mic  ─┐
Dual IMUs      ├─► ESP32 ─► BLE ─► App ─► 9-feature vector ─► Edge logistic regression ─► OA risk score
Pressure insole┘
```

During a simple **sit-to-stand test**, three sensors capture a knee-bend ("rep"):

| Sensor | Signal | Features extracted |
|---|---|---|
| Acoustic "bell" mic | Knee sounds (clicks/crepitus) | `band_energy`, `spectral_centroid`, `click_count` |
| Dual IMUs | Motion / range of motion | `ROM`, `peak_velocity`, `time_to_peak` |
| Plantar pressure insole | Foot loading | `peak_load`, `load_rate`, `center_of_pressure` |

Each rep's 9-value feature vector feeds a lightweight logistic regression classifier that outputs a risk score. Train/test splits are done **by person, not by bend**, so a subject's reps never leak across the split.

> Feature *directions* (does OA push a value up or down) are grounded in literature; only ROM currently has a literature-backed numeric range — the rest are direction-only or engineering-judgment ranges for now. This is called out explicitly in the data generator script, not glossed over.

## What's in this repo right now

This repo currently hosts the **clinician-facing dashboard**, built as a single self-contained `index.html` (no build step, no dependencies beyond Google Fonts) so it's easy to demo from any laptop or tablet.

The dashboard includes:

- **Risk Score** — a gauge (low/moderate/high bands) with the current OA risk score, plus a plain-language "why this score" breakdown of which features drove it
- **Knee Range of Motion** — a Left vs. Right flexion chart with a "restricted max flexion" callout, tappable data points, and toggleable series
- **Knee Acoustic Emissions** — a live-style waveform view with flagged click events
- **Gait Asymmetry Analysis** — left/right foot pressure heatmaps and an asymmetry index
- **Patient switching**, a full **recording flow** (3-rep capture → on-device inference → results & explanation), and **session history** with a risk trend
- A **Live Sensor Feed** panel bridging fake and real hardware: it streams simulated 9-feature packets on a timer today, and has a genuine **Web Bluetooth** connect flow (`navigator.bluetooth`) ready for the moment the ESP32 firmware exposes a matching GATT characteristic — no app-side code changes needed when that swap happens

All data in the current build is simulated for two example patients; nothing here is a real accuracy claim.

## Tech stack

- Vanilla HTML/CSS/JS (no framework, no build step) for the dashboard
- Hand-drawn SVG for the gauge, charts, waveform, and pressure maps
- Web Bluetooth API for the ESP32 link
- Python (numpy/pandas/scikit-learn) for the offline synthetic-data generator and logistic regression training/eval script
- ESP32 + BLE for the planned hardware side

## Running it locally

No install needed — just open `index.html` in a browser (Chrome or Edge if you want to try the Web Bluetooth flow, since it isn't supported on iOS Safari). It's also served live via GitHub Pages at the link above.


