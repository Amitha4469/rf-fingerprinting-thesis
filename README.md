# Swap Experiment: RF Fingerprinting with Two USRP B200 Radios

Thesis: Lightweight Device Authentication in Wireless Communication Using RF Fingerprinting
University: Kristianstad University (HKR), DT339G VT26, Computer Engineering
Supervisors: Prof. Qinghua Wang. Examiner: Ali

---

## What this experiment is

This is a preliminary experiment, a proof-of-concept run before the main thesis data collection. We had two USRP B200 software-defined radios and one question: can a small neural network tell them apart just from the radio signal they produce?

We recorded data in two configurations:

| Configuration | Transmitter | Receiver | CNN Label |
|---|---|---|---|
| Baseline | Serial 3288FAD | Serial 3288FF2 | Class 0 |
| Swapped | Serial 3288FF2 | Serial 3288FAD | Class 1 |

We then trained a lightweight 1D-CNN on those recordings and analysed what it learned.

The short answer is that we got 100% accuracy at all noise levels. More importantly, we found out exactly why, and it changes how we design the real thesis experiment.

---

## Quick results

| Metric | Value |
|---|---|
| Test accuracy at 20 dB SNR | 100.00% |
| Test accuracy at 10 dB SNR | 100.00% |
| Test accuracy at 0 dB SNR | 100.00% |
| Temporal holdout accuracy (all SNR) | 100.00% |
| CNN total parameters | 52,258 |
| Primary fingerprint feature | Carrier Frequency Offset (CFO) |
| CFO separation between devices | 1,161 Hz (std dev: 0.67 Hz) |

---

## Contents

- [Hardware Setup](#hardware-setup)
- [Data Pipeline](#data-pipeline)
- [IQ Signal Visualisations](#iq-signal-visualisations)
- [CNN Architecture](#cnn-architecture)
- [Results](#results)
- [Phase Analysis: What Did the CNN Learn?](#phase-analysis-what-did-the-cnn-learn)
- [Feature Summary](#feature-summary)
- [Signal at 0 dB SNR](#signal-at-0-db-snr)
- [Key Findings](#key-findings)
- [Thesis Implications](#thesis-implications)

---

## Hardware setup

Both radios are USRP B200 devices, physically identical units from the same manufacturer. They were placed 1 metre apart in a normal indoor environment, transmitting over-the-air with no cables.

```
[USRP B200 -- TX]  )))  ~~~  (((  [USRP B200 -- RX]
      0 dB TX gain       1 metre        30 dB RX gain
      10 kHz CW tone     OTA            1 MHz sample rate
```

| Parameter | Value | Reason |
|---|---|---|
| Signal type | Continuous Wave (CW), pure 10 kHz sine | Simple signal, hardware impairments easy to isolate |
| Sample rate | 1,000,000 samples/sec | Standard for 1 MHz bandwidth |
| TX Gain | 0 dB | Prevents receiver saturation at 1 m distance |
| RX Gain | 30 dB | Keeps received signal in usable amplitude range |
| Distance | 1 metre, fixed, line-of-sight | Consistent channel for both recordings |
| Transient skip | First 1,000,000 samples dropped | Removes hardware power-on spike |
| Segment size | 1,024 samples per CNN input window | ~1 ms of signal |

### Dataset size

| File | Device (TX) | Samples | Segments (1,024-pt) |
|---|---|---|---|
| `tx1_new.dat` | 3288FAD | 62,988,680 | 61,512 |
| `tx1_Swap_new.dat` | 3288FF2 | 63,592,520 | 62,102 |

Important limitation: in this experiment both TX and RX changed between classes. The CNN therefore learned the combined fingerprint of the full TX+RX hardware pair. The definitive thesis experiment uses a fixed receiver so only the transmitter changes.

---

## Data pipeline

```
Raw .dat file
     |
     v
[1] Drop first 1,000,000 samples  <- removes hardware power-on transient
     |
     v
[2] Normalise: divide by max(|IQ|)  <- NO mean subtraction (DC offset preserved)
     |
     v
[3] Reshape into (N x 1024 x 2) segments  <- I and Q as separate channels
     |
     v
[4] Add AWGN noise at 20 / 10 / 0 dB SNR  <- robustness testing
     |
     v
[5] Label: Class 0 or Class 1  <- combine both devices
     |
     v
[6] Temporal holdout split: train on t < 70%, test on t > 70%
```

### Why we keep the DC offset

A common preprocessing step is subtracting the mean from IQ data to remove DC offset. We deliberately skip this. The DC offset comes from the radio's Local Oscillator (LO) leaking into the received signal, a hardware impairment unique to each radio. Removing it would destroy a potential fingerprint.

Note: in this particular experiment, DC offset turned out to be identical between devices. But the preprocessing decision remains correct for the general case.

### AWGN verification

We verified that our artificially added noise was stronger than the real room noise at all SNR levels tested:

| Device | Signal Power | Real Room Noise | 20 dB OK? | 10 dB OK? | 0 dB OK? |
|---|---|---|---|---|---|
| 3288FAD | 0.769330 | 0.000540 | Yes | Yes | Yes |
| 3288FF2 | 0.612370 | 0.001482 | Yes | Yes | Yes |

30 dB SNR was excluded because the swap device's ambient noise exceeded the target noise power at that level.

---

## IQ signal visualisations

### 1. Time-domain IQ signal

Each radio transmits a 10 kHz cosine wave. The I channel (blue) and Q channel (red) are 90 degrees apart, which is the fundamental IQ representation. Even at this raw level, the two devices produce slightly different signals due to hardware impairments.

![IQ Time Domain](images/01_iq_time_domain.png)

What to look for: the frequency of the oscillation in each plot. Device 3288FAD completes its cycles slightly slower (9,418 Hz) while 3288FF2 runs faster (10,580 Hz). This is the CFO fingerprint visible directly in the time domain.

---

### 2. Constellation diagrams

A constellation diagram plots I (horizontal) vs Q (vertical) for every sample. For a perfect CW signal this would be a perfect circle. Real hardware impairments make the circle imperfect.

![Constellation](images/02_constellation.png)

What to look for: the red star marks the DC centre of each device's signal. The radius of the circle corresponds to signal amplitude. Both devices show a circular pattern (expected for CW), but at slightly different radii and with different noise spread.

---

### 3. Power Spectral Density (PSD)

The PSD shows how signal power is distributed across frequencies. For a CW signal there should be one sharp spike at the tone frequency. The position of that spike is the CFO fingerprint.

![PSD](images/03_psd.png)

What to look for: the red dashed line marks where each device's tone actually appears. Device 3288FAD's tone is at 9,418 Hz; Device 3288FF2's is at 10,580 Hz. The grey dotted line shows the nominal 10,000 Hz. Both devices are offset from nominal, in opposite directions.

---

## CNN architecture

We designed a lightweight 1D Convolutional Neural Network with 52,258 trainable parameters, roughly 80 times smaller than typical deep learning models.

```
Input: (1024 samples x 2 channels)
  |
  +-- Conv1D(32 filters, kernel=7)  +  AveragePooling1D(2)     [480 params]
  |        Detects short-range patterns: phase edges, zero-crossings
  |
  +-- Conv1D(64 filters, kernel=5)  +  AveragePooling1D(2)     [10,304 params]
  |        Mid-range signal patterns
  |
  +-- Conv1D(128 filters, kernel=3) +  AveragePooling1D(2)     [24,704 params]
  |        High-level fingerprint features
  |
  +-- GlobalAveragePooling1D                                    [0 params]
  |        Collapses time dimension, key to keeping model small
  |
  +-- Dense(128)  +  Dropout(0.5)                              [16,512 params]
  |        Final decision representation
  |
  +-- Dense(2, softmax)                                        [258 params]
           Output: probability for Class 0 and Class 1
```

Key design choices:

- AveragePooling (not MaxPooling): preserves subtle phase and timing features rather than picking only the highest-amplitude moments.
- GlobalAveragePooling1D (not Flatten): reduces the parameter count from ~4 million to ~52K by averaging across the time dimension.
- Adam, lr=0.0001: small learning rate for the subtle RF feature landscape.
- Early stopping (patience=5): all three models stopped before the 20-epoch maximum.

### Training convergence

| SNR Level | Epochs to Converge | Notes |
|---|---|---|
| 20 dB | 5 epochs | Cleanest signal, fastest learning |
| 10 dB | 5 epochs | Moderate noise, equally fast |
| 0 dB | 9 epochs | Noise equals signal power, needs slightly more time |

---

## Results

### Accuracy at all SNR levels

| SNR | Test Accuracy | Temporal Split Accuracy | Delta |
|---|---|---|---|
| 20 dB | 100.00% | 100.00% | +0.00% |
| 10 dB | 100.00% | 100.00% | +0.00% |
| 0 dB | 100.00% | 100.00% | +0.00% |

### Why we test both random and temporal splits

A standard random 80/20 split might accidentally put very similar signal chunks in both the training and test sets, a form of data leakage that inflates accuracy. The temporal split trains on the first 70% of each recording chronologically and tests on the last 30%. Zero delta between the two confirms the CNN learned something genuinely time-invariant, not a one-time artefact.

### Confusion matrices (all SNR levels)

All three confusion matrices are identical, no errors anywhere:

```
              Predicted Class 0    Predicted Class 1
True Class 0      100.00%               0.00%
True Class 1        0.00%             100.00%
```

---

## Phase analysis: what did the CNN learn?

After seeing 100% accuracy, we ran four follow-up analyses to find out what hardware feature the CNN exploited.

### Phase analysis plots

![Phase Analysis](images/04_phase_analysis.png)

Left panel: phase distribution at sample 0 of each segment. Both devices show similar uniform distributions, confirming the phase at any individual sample is not itself the fingerprint.

Middle panel: phase value at segment start over 200 segments. The values scatter randomly for both devices, as expected for a CW signal whose segments start at arbitrary phase offsets.

Right panel: unwrapped phase within one segment, with a linear fit. The slope of this line is the CFO. Device 3288FAD's phase rotates at a rate corresponding to 9,418 Hz; Device 3288FF2 rotates at 10,580 Hz. The slopes are visibly and measurably different.

---

### CFO stability across segments

We measured the CFO independently in 200 separate segments. The result is extraordinarily consistent:

![CFO Stability](images/05_cfo_stability.png)

| Device | Mean CFO | Std Dev | Stability |
|---|---|---|---|
| 3288FAD | 9,418.5 Hz | 0.67 Hz | Rock solid |
| 3288FF2 | 10,579.97 Hz | 0.67 Hz | Rock solid |
| Separation | 1,161.47 Hz | -- | Primary fingerprint |

The standard deviation of 0.67 Hz means the CNN can rely on this feature being consistent across the entire recording, across different time windows, and between training and test sets.

---

## Feature summary

### What each feature told us

![Feature Summary](images/07_feature_summary.png)

| Feature | Device 3288FAD | Device 3288FF2 | Difference | Fingerprint? |
|---|---|---|---|---|
| DC Offset |DC| | 0.000359 | 0.000359 | 0.000001 | No, identical |
| CFO (Hz) | 9,418.5 | 10,580.0 | 1,161.5 Hz | Yes, primary |
| Amplitude mean | 0.802352 | 0.836747 | 0.034 | Weak secondary |
| Phase std dev | 1.814 | 1.814 | 0.000 | No, identical |

### The CFO fingerprint explained

Every radio has a crystal oscillator that controls its operating frequency. No crystal is perfect. Each one has a small manufacturing-specific error that shifts the transmitted signal to a slightly different frequency. This shift is called the Carrier Frequency Offset (CFO).

```
Nominal CW tone:          10,000 Hz
                               |
          +--------------------+--------------------+
          |                                          |
  Device 3288FAD                              Device 3288FF2
  received at 9,418 Hz                        received at 10,580 Hz
  (-582 Hz from nominal)                      (+580 Hz from nominal)
          |                                          |
          +------------------------------------------+
                         Delta = 1,161 Hz
```

The CNN answers the question: is the tone closer to 9,418 Hz or 10,580 Hz? It can determine this from just a few samples at the start of each segment, which is confirmed by saliency maps that peak at samples 1 through 4 of 1,024.

Note on this experiment's limitation: the 1,161 Hz separation is the combined oscillator error of both TX and RX hardware pairs. In the real thesis experiment with a fixed receiver, only the TX oscillator changes between classes. The separation will be smaller, perhaps 50 to 200 Hz, making the problem harder and more scientifically rigorous.

---

## Signal at 0 dB SNR

At 0 dB SNR, the injected noise power equals the signal power. The signal is barely visible to the eye. The CNN still classifies correctly 100% of the time.

![Noisy IQ at 0 dB](images/08_noisy_iq.png)

Why this matters: in a real-world deployment, a 0 dB SNR corresponds to a very poor radio link, whether from low transmission power, long range, or heavy interference. The fact that the CFO fingerprint survives this level of noise degradation suggests it will remain useful in challenging deployment conditions.

---

### Amplitude characterisation

![Amplitude](images/06_amplitude.png)

Left: amplitude distribution. Device 3288FF2 has a slightly higher mean amplitude (0.837 vs 0.802), which could contribute as a secondary fingerprint but is a much weaker discriminator than CFO.

Middle: mean envelope shape. Both devices show a flat envelope (as expected for CW) with similar variance, confirming the signal is well-behaved between recordings.

Right: effect of noise on one segment. Shows how the I channel degrades from 20 dB to 10 dB to 0 dB SNR. Even at 0 dB the underlying tone frequency is preserved, which is why the CNN can still extract the CFO fingerprint.

---

## Key findings

### CFO is a genuine, stable hardware fingerprint

The 1,161 Hz separation is consistent to within 0.67 Hz across the entire recording. It survives all three noise levels including 0 dB SNR. The CNN needed only 5 to 9 epochs to learn it. This is a real, physical, hardware-level impairment, not a dataset artifact.

### DC offset was not the fingerprint in this experiment

Both devices had identical DC offsets (difference = 0.000001). This directly contradicts a common assumption in RF fingerprinting papers. The preprocessing decision to preserve DC offset remains correct for the general case, but this experiment alone cannot validate it as a discriminating feature.

### Temporal accuracy equals random split accuracy

There is zero performance gap between random and temporal train/test splits across all three SNR levels. The fingerprint is time-invariant within a session. The CNN generalises to unseen time windows, not just unseen shuffled segments.

### Lightweight CNN architecture validated

52,258 parameters, AveragePooling1D, GlobalAveragePooling1D. The model converges cleanly in under 10 epochs. The architecture is confirmed for use in the thesis experiment.

### Both TX and RX changed between classes

This is the key experimental limitation. The results reflect combined TX+RX hardware fingerprints, not transmitter-only fingerprints. The definitive experiment isolates the transmitter by keeping the receiver fixed.

---

## Thesis implications

### For the thesis paragraph (Results and Discussion chapter)

"Phase analysis of the swap experiment revealed that the primary discriminating feature was Carrier Frequency Offset (CFO). The two hardware configurations produced received CW tones at 9,418 Hz and 10,580 Hz respectively in the baseband, a separation of 1,161 Hz (std dev: 0.67 Hz across 200 independent segments). DC offset, commonly cited as the primary RF fingerprint in the literature, showed negligible difference between devices (delta |DC| = 0.000001). This finding demonstrates that the CNN implicitly learned oscillator-specific frequency deviation rather than amplitude-domain impairments. Since both transmitter and receiver changed between classes in this preliminary experiment, the measured CFO reflects the combined hardware offset of each RF chain pair. The definitive thesis experiment isolates transmitter-only fingerprints using a fixed receiver."

### Action items for the real experiment

| # | Action | Reason |
|---|---|---|
| 1 | Fix the receiver for all captures | Isolates TX-only fingerprint |
| 2 | Record both TX devices in the same session | Minimises channel variation between classes |
| 3 | Use temporal holdout split for all reported accuracy | Honest, publication-standard evaluation |
| 4 | Keep DC offset in preprocessing (no mean subtraction) | Conservative, correct for general case |
| 5 | Carry forward the same CNN architecture | Validated in this experiment |

---

## Repository structure

```
rf-fingerprinting/
|
+-- README.md                          <- this document
|
+-- notebooks/
|   +-- rf_fingerprinting_master.ipynb <- data loading, AWGN, CNN training
|   +-- swap_experiment_analysis.ipynb <- temporal split, DC offset, saliency
|   +-- phase_analysis.ipynb           <- CFO measurement, feature investigation
|
+-- images/
|   +-- 01_iq_time_domain.png
|   +-- 02_constellation.png
|   +-- 03_psd.png
|   +-- 04_phase_analysis.png
|   +-- 05_cfo_stability.png
|   +-- 06_amplitude.png
|   +-- 07_feature_summary.png
|   +-- 08_noisy_iq.png
|
+-- models/
    +-- rf_cnn_snr20dB.keras
    +-- rf_cnn_snr10dB.keras
    +-- rf_cnn_snr0dB.keras
```

---

## Technical notes

**Why AveragePooling instead of MaxPooling?**
MaxPooling keeps only the highest-amplitude feature in each window, which works well for image classification. For RF signals, subtle phase and timing patterns matter as much as amplitude peaks. AveragePooling preserves these by considering the full window rather than just the peak.

**Why GlobalAveragePooling1D instead of Flatten?**
After three conv layers, the tensor is shape (128, 128). Flattening gives 16,384 features going into the Dense layer, about 4 million parameters in that connection alone. GlobalAveragePooling1D reduces this to 128 features, around 16K parameters. Same representational power, 250 times fewer weights to overfit.

**Why 0.0001 learning rate?**
RF fingerprints are subtle. A large learning rate such as 0.001 causes the optimizer to overshoot the narrow loss minima where the hardware impairment features live. The slower learning rate was the fix that broke the initial 50% flatline.

---

Kristianstad University, DT339G VT26, Thesis Project, April 2025
