import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from typing import List, Dict, Tuple
from tqdm import tqdm

SENSOR_COLS = ['Amplitude - 402b', 'Amplitude - 602b', 'Amplitude - 502b', 
               'Amplitude - 702b', 'Amplitude - tgs8100', 'Amplitude - 802b']

def extract_statistical_features(window: np.ndarray) -> np.ndarray:
    """Extract mean, median, std, var, min, max, range, skew, kurtosis."""
    mean = np.mean(window, axis=0)
    median = np.median(window, axis=0)
    std = np.std(window, axis=0)
    var = np.var(window, axis=0)
    vmin = np.min(window, axis=0)
    vmax = np.max(window, axis=0)
    vrange = vmax - vmin
    vskew = skew(window, axis=0, nan_policy='omit')
    vkurtosis = kurtosis(window, axis=0, nan_policy='omit')
    
    return np.concatenate([mean, median, std, var, vmin, vmax, vrange, vskew, vkurtosis])

def extract_transient_features(window: np.ndarray) -> np.ndarray:
    """Extract rise_time, slope, integral/AUC, peak_value, time_to_peak, settling_time."""
    n_steps = window.shape[0]
    
    vmin = np.min(window, axis=0)
    vmax = np.max(window, axis=0)
    
    features = []
    for i in range(window.shape[1]):
        col = window[:, i]
        min_v = vmin[i]
        max_v = vmax[i]
        
        t10 = np.argmax(col >= min_v + 0.1 * (max_v - min_v))
        t90 = np.argmax(col >= min_v + 0.9 * (max_v - min_v))
        rise_time = max(0, t90 - t10)
        
        slope = (max_v - min_v) / (t90 - t10 + 1e-8)
        
        integral = np.sum(col)
        
        peak_value = max_v
        time_to_peak = np.argmax(col)
        
        settling_time = n_steps - time_to_peak
        
        features.extend([rise_time, slope, integral, peak_value, time_to_peak, settling_time])
        
    return np.array(features)

def extract_frequency_features(window: np.ndarray) -> np.ndarray:
    """Extract FFT dominant freq, spectral energy, spectral centroid."""
    features = []
    for i in range(window.shape[1]):
        col = window[:, i]
        fft_vals = np.fft.fft(col)
        fft_mag = np.abs(fft_vals)[:len(col)//2]
        freqs = np.fft.fftfreq(len(col))[:len(col)//2]
        
        if len(fft_mag) == 0:
            features.extend([0, 0, 0])
            continue
            
        dominant_freq = freqs[np.argmax(fft_mag)]
        spectral_energy = np.sum(fft_mag**2)
        spectral_centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-8)
        
        features.extend([dominant_freq, spectral_energy, spectral_centroid])
        
    return np.array(features)

def extract_derivative_features(window: np.ndarray) -> np.ndarray:
    """Extract max(dx/dt), mean(dx/dt), zero_crossing_rate."""
    features = []
    for i in range(window.shape[1]):
        col = window[:, i]
        dx = np.diff(col)
        
        if len(dx) == 0:
            features.extend([0, 0, 0])
            continue
            
        max_dx = np.max(dx)
        mean_dx = np.mean(dx)
        
        zero_crossings = np.sum(np.diff(np.sign(dx)) != 0) / len(dx)
        
        features.extend([max_dx, mean_dx, zero_crossings])
        
    return np.array(features)

def extract_all_features(window: np.ndarray) -> np.ndarray:
    """Combine all 21 features * 6 sensors = 126 features."""
    stat = extract_statistical_features(window)
    transient = extract_transient_features(window)
    freq = extract_frequency_features(window)
    deriv = extract_derivative_features(window)
    
    return np.concatenate([stat, transient, freq, deriv])

def build_feature_matrix(windows: List[Dict], labels: List[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Apply feature extraction to all windows, save, and return X, y."""
    X = []
    y = []
    
    for w in tqdm(windows, desc="Extracting features"):
        window_data = w['window']
        if np.isnan(window_data).any():
            window_data = np.nan_to_num(window_data)
        
        feats = extract_all_features(window_data)
        X.append(feats)
        
        if labels is None:
            l = w['labels']
            y.append([l['h'], l['n'], l['c'], l['e']])
            
    X_arr = np.array(X)
    y_arr = np.array(y) if y else None
    
    out_dir = r"C:\Users\HARSH\.gemini\antigravity\scratch\enose_research\data\features"
    os.makedirs(out_dir, exist_ok=True)
    
    if len(X_arr) > 0:
        np.save(os.path.join(out_dir, "X_features.npy"), X_arr)
    if y_arr is not None and len(y_arr) > 0:
        np.save(os.path.join(out_dir, "y_labels.npy"), y_arr)
        
    return X_arr, y_arr
