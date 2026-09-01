import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from urllib.parse import quote

# --- 1. VERİ ÇEKME FONKSİYONU ---
def fetch_wikipedia_pageviews(article_title: str, start_date: str, end_date: str, lang: str = "en") -> pd.DataFrame:
    clean_title = article_title.strip().replace(" ", "_")
    encoded_title = quote(clean_title)
    
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{lang}.wikipedia/all-access/user/{encoded_title}/daily/{start_date}/{end_date}"
    )
    
    headers = {
        "User-Agent": "ChronosBiasAnalytics/1.0 (portfolio_research@example.com)"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 404:
        raise ValueError(f"'{article_title}' başlığı {lang}.wikipedia üzerinde bulunamadı.")
    elif response.status_code != 200:
        raise ConnectionError(f"API Hatası ({response.status_code}): {response.text}")
        
    data = response.json().get("items", [])
    if not data:
        raise ValueError("Belirtilen tarih aralığında veri bulunamadı.")
        
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"].str[:8], format="%Y%m%d")
    df["views"] = df["views"].astype(int)
    
    return df[["date", "views"]].sort_values("date").reset_index(drop=True)

# --- 2. ÜSTEL SÖNÜMLENME MODELİ (SAF NUMPY) ---
def fit_decay_model(df: pd.DataFrame):
    if df.empty or len(df) < 5:
        raise ValueError("Analiz için yeterli veri noktası yok.")
        
    peak_idx = df["views"].idxmax()
    peak_row = df.iloc[peak_idx]
    
    decay_df = df.iloc[peak_idx:].copy().reset_index(drop=True)
    decay_df["t"] = (decay_df["date"] - decay_df["date"].iloc[0]).dt.days
    
    t_data = decay_df["t"].values
    y_data = decay_df["views"].values.astype(float)
    
    n_0 = y_data[0]
    n_base = float(np.percentile(y_data, 10))
    
    # Log-linear fitting: ln(y - n_base) = ln(n_0 - n_base) - lambda * t
    y_adjusted = np.maximum(y_data - n_base, 1.0)
    
    if len(t_data) > 1 and (t_data[-1] - t_data[0]) > 0:
        log_y = np.log(y_adjusted)
        # Doğrusal regresyon eğimi -> -lambda
        slope, _ = np.polyfit(t_data, log_y, 1)
        decay_rate = max(-slope, 0.0001)
    else:
        decay_rate = 0.01
        
    half_life_days = np.log(2) / decay_rate if decay_rate > 0 else 0.0
    decay_df["fitted_views"] = n_base + (n_0 - n_base) * np.exp(-decay_rate * t_data)
    
    metrics = {
        "peak_date": peak_row["date"].strftime("%Y-%m-%d"),
        "peak_views": int(peak_row["views"]),
        "baseline_views": int(n_base),
        "decay_rate": float(decay_rate),
        "half_life_days": round(float(half_life_days), 1)
    }
    
    return decay_df, metrics

# --- 3. STREAMLIT ARAYÜZÜ ---
st.set_page_config(page_title="Chronos Bias", layout="wide", page_icon="⏳")

st.title("⏳ Chronos Bias: Kolektif Hafıza & Dijital Unutuş Analitiği")
st.markdown("Olayların, krizlerin ve figürlerin toplum hafızasındaki yarılanma ömrünü modelleyin.")

# Yan Panel
st.sidebar.header("🔍 Analiz Parametreleri")
article = st.sidebar.text_input("Wikipedia Başlığı", value="COVID-19 pandemic")
lang = st.sidebar.selectbox("Dil", options=["en", "tr"], index=0)
col_s, col_e = st.sidebar.columns(2)
start_date = col_s.text_input("Başlangıç", value="20200101")
end_date = col_e.text_input("Bitiş", value="20211231")

run_btn = st.sidebar.button("Analizi Çalıştır", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("Veriler Wikimedia üzerinden alınıp modelleniyor..."):
        try:
            df = fetch_wikipedia_pageviews(article, start_date, end_date, lang)
            decay_df, metrics = fit_decay_model(df)
            
            # Metrik Kartları
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Zirve Tarihi", metrics["peak_date"])
            m2.metric("Zirve Günlük Okunma", f"{metrics['peak_views']:,}")
            m3.metric("Tahmini Taban İlgi", f"{metrics['baseline_views']:,}")
            m4.metric("Dikkatin Yarılanma Ömrü (t½)", f"{metrics['half_life_days']} Gün")
            
            # Plotly Grafiği
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["views"],
                mode="lines", name="Organik Sayfa Görüntüleme",
                line=dict(color="#29B5E8", width=1.8),
                opacity=0.65
            ))
            fig.add_trace(go.Scatter(
                x=decay_df["date"], y=decay_df["fitted_views"],
                mode="lines", name=f"Sönümlenme Eğrisi (λ={metrics['decay_rate']:.4f})",
                line=dict(color="#FF4B4B", width=3, dash="dash")
            ))
            
            fig.update_layout(
                title=f"'{article}' İlgi Eğrisi ve Unutuluş Modeli",
                xaxis_title="Tarih",
                yaxis_title="Günlük Organik Okunma Sayısı",
                template="plotly_dark",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Hata: {str(e)}")