import pandas as pd
import numpy as np

def baseline_correction(df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Subtract clean air response."""
    # Assuming baseline_df has a single row of means or is aligned
    return df - baseline_df.mean()

def ewma_smoothing(df: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    """Exponentially weighted moving average."""
    return df.ewm(alpha=alpha).mean()

def moving_average(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Simple moving average."""
    return df.rolling(window=window, min_periods=1).mean()

def min_max_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Scale to [0,1]."""
    return (df - df.min()) / (df.max() - df.min() + 1e-8)

def standard_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalization."""
    return (df - df.mean()) / (df.std() + 1e-8)

def extract_transient(df: pd.DataFrame, n_points: int = 500) -> pd.DataFrame:
    """First N points after gas introduction."""
    return df.head(n_points)

def preprocess_pipeline(df: pd.DataFrame, baseline_df: pd.DataFrame = None, 
                        steps: list = ['baseline', 'ewma', 'minmax']) -> pd.DataFrame:
    """Chain multiple preprocessing steps."""
    res = df.copy()
    for step in steps:
        if step == 'baseline' and baseline_df is not None:
            res = baseline_correction(res, baseline_df)
        elif step == 'ewma':
            res = ewma_smoothing(res)
        elif step == 'moving_average':
            res = moving_average(res)
        elif step == 'minmax':
            res = min_max_normalize(res)
        elif step == 'standard':
            res = standard_scale(res)
        elif step == 'transient':
            res = extract_transient(res)
    return res
