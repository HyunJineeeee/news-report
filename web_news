import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib.parse
from pathlib import Path
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
import re

# === 설정 ===
KEYWORDS = ["일학습병행", "직업훈련"]
DATA_DIR = Path("data")
LOG_DIR = Path("logs")
WRITE_LOG_FILE = True

# === 유틸 ===
def make_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET"], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retries)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; NewsCrawler/1.0)"})
    return sess

def normalize_url(url: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(url)
        q = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        cleaned = []
        for k, v in q:
            kl = k.lower()
            if kl.startswith("utm_") or kl in {"hl","gl","ceid","oc"}: 
                continue
            cleaned.append((k, v))
        cleaned.sort(key=lambda x: x[0])
        new_query = urllib.parse.urlencode(cleaned, doseq=True)
        return urllib.parse.urlunsplit(
            (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, new_query, "")
        )
    except Exception:
        return url

def extract_domain(url: str) -> str:
    try:
        return urllib.parse.urlsplit(url).netloc.lower()
    except Exception:
        return ""

def parse_pub_date(text: str):
    if not text: 
        return pd.NaT
    return pd.to_datetime(text, utc=True, errors="coerce")

def to_kst_str_from_utc(ts):
    if pd.isna(ts):
        return None
    try:
        return ts.tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None

def safe_name(name: str) -> str:
    return re.sub(r"[\\/:*?\[\]]", "_", str(name))[:64] or "Sheet"

# === 크롤러 ===
def crawl_google_news_rss(session: requests.Session, keyword: str):
    encoded_kw = urllib.parse.quote(keyword)
    url = f"https://news.google.com/rss/search?q={encoded_kw}&hl=ko&gl=KR&ceid=KR:ko"
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "xml")
    items = soup.find_all("item")

    collected_at = pd.Timestamp.utcnow()
    rows = []
    for item in items:
        title = item.title.text if item.title else ""
        link = item.link.text if item.link else ""
        pub_date_raw = item.pubDate.text if item.pubDate else ""
        pub_ts = parse_pub_date(pub_date_raw)
        rows.append({
            "키워드": keyword,
            "제목": title,
            "링크": link,
            "발행일_UTC": pub_ts,
            "수집시각_UTC": collected_at,
            "_정규화링크": normalize_url(link),
            "출처": extract_domain(link),
        })
    print(f"✅ '{keyword}' 뉴스 {len(rows)}건 수집")
    return rows

# === 메인 ===
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    # 기존 ALL.csv 불러오기(누적)
    all_path = DATA_DIR / "ALL.csv"
    if all_path.exists():
        df_existing = pd.read_csv(all_path, dtype=str, encoding="utf-8-sig")
        for c in ["발행일(KST)", "수집시각(KST)"]:
            df_existing[c] = pd.to_datetime(df_existing[c], errors="coerce")
        df_existing["_정규화링크"] = df_existing["링크"].fillna("").apply(normalize_url)
        df_existing["_발행일_dt"] = pd.to_datetime(df_existing["발행일(KST)"], errors="coerce")
        df_existing["_수집시각_dt"] = pd.to_datetime(df_existing["수집시각(KST)"], errors="coerce")
        df_existing["_is_new"] = False
    else:
        df_existing = pd.DataFrame(columns=[
            "키워드","제목","링크","발행일(KST)","수집시각(KST)","출처",
            "_정규화링크","_발행일_dt","_수집시각_dt","_is_new"
        ])

    # 신규 수집
    all_rows = []
    for kw in KEYWORDS:
        all_rows.extend(crawl_google_news_rss(session, kw))
        time.sleep(0.5)

    df_new_raw = pd.DataFrame(all_rows)
    if not df_new_raw.empty:
        df_new_raw["발행일(KST)"] = df_new_raw["발행일_UTC"].apply(to_kst_str_from_utc)
        df_new_raw["수집시각(KST)"] = df_new_raw["수집시각_UTC"].apply(to_kst_str_from_utc)
        df_new_raw["_정규화링크"] = df_new_raw["_정규화링크"].fillna(df_new_raw["링크"]).apply(normalize_url)
        df_new_raw["_발행일_dt"] = pd.to_datetime(df_new_raw["발행일(KST)"], errors="coerce")
        df_new_raw["_수집시각_dt"] = pd.to_datetime(df_new_raw["수집시각(KST)"], errors="coerce")
        existing_norm = set(df_existing["_정규화링크"].dropna().astype(str))
        df_new_raw["_is_new"] = ~df_new_raw["_정규화링크"].astype(str).isin(existing_norm)
    else:
        df_new_raw = pd.DataFrame(columns=list(df_existing.columns))

    # 병합 + 중복 제거
    combined = pd.concat([df_existing, df_new_raw], ignore_index=True)
    combined = combined.sort_values("_수집시각_dt", ascending=False, na_position="last")
    combined = combined.drop_duplicates(subset=["_정규화링크"], keep="first")
    combined["_발행일_일"] = combined["_발행일_dt"].dt.date
    combined = combined.drop_duplicates(subset=["제목", "_발행일_일"], keep="first")

    # 표시용
    out_cols = ["키워드","제목","링크","발행일(KST)","수집시각(KST)","출처"]
    df_all = combined[out_cols].copy()
    df_all.to_csv(DATA_DIR / "ALL.csv", index=False, encoding="utf-8-sig")

    # 키워드별
    for kw, g in df_all.groupby("키워드", sort=False):
        g.to_csv(DATA_DIR / f"{safe_name(kw)}.csv", index=False, encoding="utf-8-sig")

    # NEW
    today = datetime.now().strftime("%Y%m%d")
    df_new = combined.loc[combined["_is_new"]==True, out_cols].copy()
    df_new = df_new.sort_values(["수집시각(KST)", "발행일(KST)"], ascending=False)
    df_new.to_csv(DATA_DIR / f"NEW_{today}.csv", index=False, encoding="utf-8-sig")
    df_new.to_csv(DATA_DIR / "NEW_latest.csv", index=False, encoding="utf-8-sig")

    print("🎉 CSV 저장 완료")

if __name__ == "__main__":
    main()

