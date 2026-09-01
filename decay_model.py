import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def exp_decay_function(t, n_0, decay_rate, n_base):
    return n_base + (n_0 - n_base) * np.exp(-decay_rate * t)

def fit_decay_model(df: pd.DataFrame):
    """
    Zirve noktasını tespit eder ve sonrasındaki sönümlenme katsayısını hesaplar.
    """
    if df.empty or len(df) < 5:
        raise ValueError("Analiz için yeterli veri noktası yok.")
        
    peak_idx = df["views"].idxmax()
    peak_row = df.iloc[peak_idx]
    
    decay_df = df.iloc[peak_idx:].copy().reset_index(drop=True)
    decay_df["t"] = (decay_df["date"] - decay_df["date"].iloc[0]).dt.days
    
    t_data = decay_df["t"].values
    y_data = decay_df["views"].values
    
    # Başlangıç parametreleri: [n_0, lambda, n_base]
    p0 = [float(y_data[0]), 0.05, float(np.percentile(y_data, 10))]
    bounds = ([0.0, 0.0001, 0.0], [np.inf, 2.0, np.inf])
    
    try:
        popt, _ = curve_fit(exp_decay_function, t_data, y_data, p0=p0, bounds=bounds, maxfev=5000)
        n_0_fit, lambda_fit, n_base_fit = popt
    except Exception:
        n_0_fit, lambda_fit, n_base_fit = y_data[0], 0.01, np.min(y_data)
        
    half_life_days = np.log(2) / lambda_fit if lambda_fit > 0 else 0.0
    decay_df["fitted_views"] = exp_decay_function(t_data, n_0_fit, lambda_fit, n_base_fit)
    
    metrics = {
        "peak_date": peak_row["date"].strftime("%Y-%m-%d"),
        "peak_views": int(peak_row["views"]),
        "baseline_views": int(n_base_fit),
        "decay_rate": float(lambda_fit),
        "half_life_days": round(float(half_life_days), 1)
    }
    
    return decay_df, metrics