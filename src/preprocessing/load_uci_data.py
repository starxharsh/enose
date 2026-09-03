import os
import zipfile
import warnings
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data'

def load_uci_487_co_regression(sample_rate: int = 50, max_samples: int = 6000) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Loads UCI 487 (Gas Sensor Array Temperature Modulation) for continuous CO concentration regression.
    Features: 14 MOX sensor resistances (R1-R14) + Temp + Humidity + Heater Voltage = 17 features.
    Target: CO concentration in ppm (continuous: 0 - 20 ppm).
    """
    feat_dir = DATA_DIR / 'features'
    x_cache = feat_dir / 'uci_487_X.npy'
    y_cache = feat_dir / 'uci_487_y.npy'
    
    if x_cache.exists() and y_cache.exists():
        print("  Loading cached UCI 487 regression features...")
        X = np.load(x_cache)
        y = np.load(y_cache)
        feature_names = [f'R{i+1}' for i in range(14)] + ['Humidity', 'Temp', 'Heater_V']
        return X, y, feature_names

    raw_dir = DATA_DIR / 'raw' / 'uci_487'
    inner_zip = raw_dir / 'gas-sensor-array-temperature-modulation.zip'
    outer_zip = raw_dir / 'temp_mod.zip'

    if not inner_zip.exists() and outer_zip.exists():
        with zipfile.ZipFile(outer_zip) as z:
            z.extract('gas-sensor-array-temperature-modulation.zip', str(raw_dir))

    if not inner_zip.exists():
        raise FileNotFoundError(f"UCI 487 zip not found at {inner_zip}")

    feature_cols = [f'R{i} (MOhm)' for i in range(1, 15)] + ['Humidity (%r.h.)', 'Temperature (C)', 'Heater voltage (V)']
    target_col = 'CO (ppm)'

    dfs = []
    with zipfile.ZipFile(inner_zip) as z:
        csv_files = [n for n in z.namelist() if n.endswith('.csv')]
        for fname in csv_files[:4]:  # use first 4 days for a balanced diversity
            with z.open(fname) as f:
                df = pd.read_csv(f)
                valid_cols = [c for c in feature_cols if c in df.columns]
                sub_df = df.iloc[::sample_rate][valid_cols + [target_col]].copy()
                dfs.append(sub_df)

    full_df = pd.concat(dfs, ignore_index=True).dropna()
    if len(full_df) > max_samples:
        full_df = full_df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    sensor_cols_found = [c for c in feature_cols if c in full_df.columns]
    X = full_df[sensor_cols_found].values
    y = full_df[target_col].values.reshape(-1, 1)

    feature_names = [c.replace(' (MOhm)', '').replace(' (%r.h.)', '').replace(' (C)', '').replace(' (V)', '') for c in sensor_cols_found]
    
    feat_dir.mkdir(parents=True, exist_ok=True)
    np.save(x_cache, X)
    np.save(y_cache, y)
    print(f"  UCI 487 parsed: X shape={X.shape}, y shape={y.shape} (CO 0-20 ppm)")
    return X, y, feature_names


def load_uci_309_mixture_regression() -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Loads UCI 309 (Turbulent gas mixtures) for continuous mixture concentration regression.
    Features: 8 MOX sensor responses (mean + transient max/min per experiment).
    Targets: [Ethylene_ppm, CO_or_Methane_ppm].
    """
    feat_dir = DATA_DIR / 'features'
    x_cache = feat_dir / 'uci_309_X.npy'
    y_cache = feat_dir / 'uci_309_y.npy'

    if x_cache.exists() and y_cache.exists():
        print("  Loading cached UCI 309 regression features...")
        X = np.load(x_cache)
        y = np.load(y_cache)
        feature_names = [f'S{i+1}_mean' for i in range(8)] + [f'S{i+1}_max' for i in range(8)]
        return X, y, feature_names

    zip_path = DATA_DIR / 'raw' / 'uci_309' / 'turbulent.zip'
    if not zip_path.exists():
        raise FileNotFoundError(f"UCI 309 zip not found at {zip_path}")

    levels = {'n': 0.0, 'L': 50.0, 'M': 150.0, 'H': 300.0}
    rows_X = []
    rows_y = []

    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith('/') or not name.startswith('dataset_twosources_raw/'):
                continue
            fname = name.split('/')[-1]
            parts = fname.split('_')
            if len(parts) >= 5:
                # 000_Et_H_CO_n
                c_et = levels.get(parts[2], 0.0)
                c_other = levels.get(parts[4], 0.0)
                with z.open(name) as f:
                    df = pd.read_csv(f, header=None)
                    # columns: Time, Temp, RH, S1..S8
                    sensors = df.iloc[:, 3:11].values
                    s_mean = np.mean(sensors, axis=0)
                    s_max = np.max(sensors, axis=0)
                    feat = np.concatenate([s_mean, s_max])
                    rows_X.append(feat)
                    rows_y.append([c_et, c_other])

    X = np.array(rows_X)
    y = np.array(rows_y)
    feature_names = [f'S{i+1}_mean' for i in range(8)] + [f'S{i+1}_max' for i in range(8)]

    feat_dir.mkdir(parents=True, exist_ok=True)
    np.save(x_cache, X)
    np.save(y_cache, y)
    print(f"  UCI 309 parsed: X shape={X.shape}, y shape={y.shape}")
    return X, y, feature_names


def load_uci_datasets() -> Dict[str, Any]:
    """Fallback loader for UCI datasets."""
    datasets = {}
    try:
        X_487, y_487, _ = load_uci_487_co_regression()
        datasets['temperature'] = {'X': X_487, 'y': y_487}
    except Exception as e:
        datasets['temperature'] = {'X': None, 'y': None}

    try:
        X_309, y_309, _ = load_uci_309_mixture_regression()
        datasets['turbulent'] = {'X': X_309, 'y': y_309}
    except Exception as e:
        datasets['turbulent'] = {'X': None, 'y': None}

    return datasets
