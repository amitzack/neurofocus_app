# -*- coding: utf-8 -*-
"""
features.py — EXACT re-implementation of the raw-EEG -> feature-vector
pipeline from CODE_MRMR_19_ELEC.py (the real preprocessing script).

This must stay byte-for-byte equivalent to process_subject() /
extract_window_features() in that file — that's what the saved
model (knn_model.joblib + selected_indices.npy) expects as input.
If you ever change the preprocessing script, mirror the change here too.

Feature order per window (817 per-channel + 684 coherence = 1501 total):
  time_features, frequency_power_features, frequency_ratio_features,
  spectral_entropy_features, dfa_features, hurst_features,
  fooof_features, advanced_features, coherence_features
"""

import numpy as np
import scipy.stats
from scipy.signal import butter, filtfilt, welch, coherence
from fooof import FOOOF
import antropy as ant
import pywt

FS = 128
WINDOW_SEC = 2
OVERLAP_PCT = 0.5
LPF_CUTOFF, LPF_ORDER = 30, 4
HPF_CUTOFF, HPF_ORDER = 0.5, 2


# ---------------------------------------------------------------------
# Filters & normalization  (same order as the original: lowpass -> highpass -> zscore)
# ---------------------------------------------------------------------
def lowpass_filter(sig):
    nyq = 0.5 * FS
    b, a = butter(LPF_ORDER, LPF_CUTOFF / nyq, btype="low")
    return filtfilt(b, a, sig, axis=0)


def highpass_filter(sig):
    nyq = 0.5 * FS
    b, a = butter(HPF_ORDER, HPF_CUTOFF / nyq, btype="high")
    return filtfilt(b, a, sig, axis=0)


def zscore_normalization(sig):
    std = np.std(sig, axis=0)
    std[std == 0] = 1
    return (sig - np.mean(sig, axis=0)) / std


def preprocess(raw):
    raw = np.asarray(raw, dtype=float)
    data_f = highpass_filter(lowpass_filter(raw))
    data_f = zscore_normalization(data_f)
    return data_f


# ---------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------
def segment_signal(data):
    win_len = int(WINDOW_SEC * FS)
    step = int(win_len * (1 - OVERLAP_PCT))
    return [data[i:i + win_len, :] for i in range(0, data.shape[0] - win_len + 1, step)]


# ---------------------------------------------------------------------
# Feature families — identical logic/order to the original script
# ---------------------------------------------------------------------
def band_power(freqs, psd, fmin, fmax):
    idx = (freqs >= fmin) & (freqs <= fmax)
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return trapz_fn(psd[idx], freqs[idx]) if np.any(idx) else 0


def time_features(window):
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]
        diff1 = np.diff(x)
        diff2 = np.diff(diff1)
        act = np.var(x)
        mob = np.sqrt(np.var(diff1) / act) if act > 0 else 0
        comp = np.sqrt(np.var(diff2) / np.var(diff1)) / mob if mob > 0 else 0
        feats.extend([
            np.mean(x), np.median(x), np.var(x, ddof=1), np.sqrt(np.mean(x ** 2)),
            np.ptp(x), scipy.stats.iqr(x), scipy.stats.skew(x), scipy.stats.kurtosis(x),
            np.sum(np.diff(np.sign(x)) != 0), np.sum(np.diff(np.sign(diff1)) != 0),
            act, mob, comp,
        ])
        entropy_fns = [ant.sample_entropy, ant.perm_entropy, ant.app_entropy,
                       getattr(ant, "fuzzy_entropy", None)]
        for fn in entropy_fns:
            if fn is None:
                feats.append(0)
                continue
            try:
                feats.append(fn(x) if fn != ant.perm_entropy else fn(x, normalize=True))
            except Exception:
                feats.append(0)
    return np.array(feats)


def frequency_power_features(window):
    feats = []
    for ch in range(window.shape[1]):
        freqs, psd = welch(window[:, ch], fs=FS, nperseg=FS)
        d, t, a, b = (band_power(freqs, psd, 1, 4), band_power(freqs, psd, 4, 8),
                      band_power(freqs, psd, 8, 13), band_power(freqs, psd, 13, 30))
        total = d + t + a + b + 1e-12
        feats.extend([d, t, a, b, d / total, t / total, a / total, b / total])
    return np.array(feats)


def frequency_ratio_features(window):
    feats = []
    for ch in range(window.shape[1]):
        freqs, psd = welch(window[:, ch], fs=FS, nperseg=FS)
        d, t, a, b = (band_power(freqs, psd, 0.5, 4), band_power(freqs, psd, 4, 8),
                      band_power(freqs, psd, 8, 13), band_power(freqs, psd, 13, 30))
        feats.extend([t / b if b > 0 else 0, t / a if a > 0 else 0, a / b if b > 0 else 0,
                      d / t if t > 0 else 0, (t + a) / b if b > 0 else 0])
    return np.array(feats)


def spectral_entropy_features(window):
    feats = []
    for ch in range(window.shape[1]):
        freqs, psd = welch(window[:, ch], fs=FS, nperseg=FS)
        psd_n = (psd + 1e-12) / np.sum(psd + 1e-12)
        feats.append(-np.sum(psd_n * np.log2(psd_n)))
    return np.array(feats)


def dfa_features(window):
    vals = []
    for ch in range(window.shape[1]):
        try:
            vals.append(ant.detrended_fluctuation(window[:, ch]))
        except Exception:
            vals.append(0)
    return np.array(vals)


def hurst_features(window):
    vals = []
    for ch in range(window.shape[1]):
        try:
            vals.append(ant.hurst_rs(window[:, ch]))
        except Exception:
            vals.append(0.5)
    return np.array(vals)


def fooof_features(window):
    feats = []
    fm = FOOOF(peak_width_limits=(1, 8), max_n_peaks=5, aperiodic_mode="fixed", verbose=False)
    for ch in range(window.shape[1]):
        freqs, psd = welch(window[:, ch], fs=FS, nperseg=FS)
        try:
            fm.fit(freqs, psd)
            off, exp = fm.aperiodic_params_
            pk = fm.get_params("peak_params")
            al = pk[(pk[:, 0] >= 8) & (pk[:, 0] <= 13)]
            feats.extend([off, exp, al[0, 0], al[0, 1]] if len(al) > 0 else [off, exp, 0, 0])
        except Exception:
            feats.extend([0, 0, 0, 0])
    return np.array(feats)


def advanced_features(window):
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]
        try:
            coeffs = pywt.wavedec(x, "db4", level=4)
            for c in coeffs:
                feats.append(np.sum(c ** 2))
        except Exception:
            feats.extend([0] * 5)
        try:
            feats.append(ant.lziv_complexity(x))
        except Exception:
            feats.append(0)
    return np.array(feats)


def coherence_features(window):
    """171 channel pairs x 4 bands = 684 features."""
    n_ch = window.shape[1]
    bands = [(1, 4), (4, 8), (8, 13), (13, 30)]
    feats = []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            f, Cxy = coherence(window[:, i], window[:, j], fs=FS, nperseg=FS)
            for fmin, fmax in bands:
                idx = (f >= fmin) & (f <= fmax)
                feats.append(np.mean(Cxy[idx]) if np.any(idx) else 0.0)
    return np.array(feats)


def extract_window_features(window):
    return np.concatenate([
        time_features(window), frequency_power_features(window),
        frequency_ratio_features(window), spectral_entropy_features(window),
        dfa_features(window), hurst_features(window),
        fooof_features(window), advanced_features(window),
        coherence_features(window),
    ])


def raw_recording_to_features(raw):
    """
    Full pipeline: raw (n_samples, 19) -> (n_windows, 1501)

    Used both by train_from_raw.py (if used) and by app.py at prediction
    time, so a live upload is processed exactly like the training data was.
    """
    filtered = preprocess(raw)
    windows = segment_signal(filtered)
    if not windows:
        raise ValueError(
            f"Recording too short: needs at least {int(WINDOW_SEC * FS)} samples "
            f"({WINDOW_SEC}s at {FS}Hz), got {raw.shape[0]}."
        )
    rows = [extract_window_features(w) for w in windows]
    return np.vstack(rows)
