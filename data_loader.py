import requests
import pandas as pd
from urllib.parse import quote

def fetch_wikipedia_pageviews(article_title: str, start_date: str, end_date: str, lang: str = "en") -> pd.DataFrame:
    """
    Wikimedia REST API üzerinden organik günlük sayfa görüntüleme verisini çeker.
    Tarih formatı: YYYYMMDD
    """
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