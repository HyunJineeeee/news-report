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

# ============== 설정 ==============
KEYWORDS = ["일학습병행", "직업훈련"]
DATA_DIR = Path("data")

# ============== 유틸 ==============
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    ad = HTTPAdapter(max_retries=retries)
    s.mount("http://", ad); s.mount("https://", ad)
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; NewsCrawler/1.0)"})
    return s

def normalize_url(url: str) -> str:
    try:
        p = urllib.parse.urlsplit(url)
        q = urllib.parse.parse_qsl(p.query, keep_blank_values=True)
        kept = []
        for k, v in q:
            kl = k.lower()
            if kl.startswith("utm_") or kl in {"hl", "gl", "ceid", "oc"}:
                continue
            kept.append((k, v))
        kept.sort(key=lambda x: x[0])
        nq = urllib.parse.urlencode(kept, doseq=True)
        return urllib.parse.urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path, nq, ""))
    except Exception:
        return url

def extract_domain(url: str) -> str:
    try: return urllib.parse.urlsplit(url).netloc.lower()
    except: return ""

def parse_pub_date(text: str):
    if not text: return pd.NaT
    return pd.to_datetime(text, utc=True, errors="coerce")

def to_kst_str_from_utc(ts):
    if pd.isna(ts): return None
    try: return ts.tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M")
    except: return None

def safe_name(name: str) -> str:
    return re.sub(r"[\\/:*?\[\]]", "_", str(name))[:64] or "Sheet"

def resolve_final_url(session: requests.Session, url: str, timeout: float = 10.0) -> str:
    try:
        r = session.head(url, allow_redirects=True, timeout=timeout)
        fu = r.url
        if not fu or fu == url:
            r = session.get(url, allow_redirects=True, timeout=timeout)
            fu = r.url
        return fu or url
    except Exception:
        return url

# ============== 크롤링 ==============
def crawl_google_news_rss(session: requests.Session, keyword: str):
    q = urllib.parse.quote(keyword)
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "xml")  # lxml 필요
    items = soup.find_all("item")

    collected_at = pd.Timestamp.utcnow()
    rows = []
    for it in items:
        title = it.title.text if it.title else ""
        link = it.link.text if it.link else ""
        pub_ts = parse_pub_date(it.pubDate.text if it.pubDate else "")

        final_link = resolve_final_url(session, link)
        rows.append({
            "키워드": keyword,
            "제목": title,
            "원문링크": final_link,                # 표시/하이퍼링크용
            "출처": extract_domain(final_link) or extract_domain(link),
            "발행일_UTC": pub_ts,
            "수집시각_UTC": collected_at,
            "_정규화링크": normalize_url(link),    # 중복제거 키(구글뉴스 링크 정규화)
        })
    print(f"✅ '{keyword}' {len(rows)}건")
    return rows

# ============== 메인 ==============
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    # 1) 기존 ALL.csv 로드(없으면 빈 DF)
    all_path = DATA_DIR / "ALL.csv"
    if all_path.exists():
        df_existing = pd.read_csv(all_path, dtype=str, encoding="utf-8-sig")
        # 문자열을 날짜로 복원
        df_existing["발행일(KST)"] = pd.to_datetime(df_existing["발행일(KST)"], errors="coerce")
        df_existing["수집시각(KST)"] = pd.to_datetime(df_existing["수집시각(KST)"], errors="coerce")
    else:
        df_existing = pd.DataFrame(columns=["키워드","제목","원문링크","발행일(KST)","수집시각(KST)","출처"])

    # 내부용 컬럼 준비
    if not df_existing.empty:
        df_existing["_정규화링크"] = df_existing["원문링크"].fillna("").apply(normalize_url)
        df_existing["_발행일_dt"] = pd.to_datetime(df_existing["발행일(KST)"], errors="coerce")
        df_existing["_수집시각_dt"] = pd.to_datetime(df_existing["수집시각(KST)"], errors="coerce")
        df_existing["_is_new"] = False

    # 2) 신규 수집
    all_rows = []
    for kw in KEYWORDS:
        all_rows.extend(crawl_google_news_rss(session, kw))
        time.sleep(0.5)
    df_new = pd.DataFrame(all_rows)

    if not df_new.empty:
        df_new["발행일(KST)"] = df_new["발행일_UTC"].apply(to_kst_str_from_utc)
        df_new["수집시각(KST)"] = df_new["수집시각_UTC"].apply(to_kst_str_from_utc)
        df_new["_발행일_dt"] = pd.to_datetime(df_new["발행일(KST)"], errors="coerce")
        df_new["_수집시각_dt"] = pd.to_datetime(df_new["수집시각(KST)"], errors="coerce")

        # 이전 ALL 기준 '신규' 판정
        existing_norm = set(df_existing["_정규화링크"].dropna().astype(str)) if "_정규화링크" in df_existing.columns else set()
        df_new["_is_new"] = ~df_new["_정규화링크"].astype(str).isin(existing_norm)
    else:
        df_new = pd.DataFrame(columns=[
            "키워드","제목","원문링크","발행일(KST)","수집시각(KST)","출처",
            "_정규화링크","_발행일_dt","_수집시각_dt","_is_new"
        ])

    # 3) 병합 + 중복제거(수집 최신 우선)
    combined = pd.concat([df_existing, df_new], ignore_index=True)
    if not combined.empty:
        combined = combined.sort_values("_수집시각_dt", ascending=False, na_position="last")
        combined = combined.drop_duplicates(subset=["_정규화링크"], keep="first")
        combined["_발행일_일"] = combined["_발행일_dt"].dt.date
        combined = combined.drop_duplicates(subset=["제목","_발행일_일"], keep="first")

    # 4) 표시용 DF + 날짜 백필(빈 값 방지)
    out_cols = ["키워드","제목","원문링크","발행일(KST)","수집시각(KST)","출처"]
    combined_display = combined.sort_values("_수집시각_dt", ascending=False, na_position="last")
    df_all = combined_display[out_cols].copy()

    # 빈 날짜 보정: 발행일 없으면 수집시각으로, 수집시각 없으면 지금(KST)
    def fmt(s): return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    df_all["발행일(KST)"] = df_all["발행일(KST)"].where(df_all["발행일(KST)"].notna(), fmt(combined_display["_발행일_dt"]))
    df_all["발행일(KST)"] = df_all["발행일(KST)"].where(df_all["발행일(KST)"].notna(), fmt(combined_display["_수집시각_dt"]))
    df_all["수집시각(KST)"] = df_all["수집시각(KST)"].where(df_all["수집시각(KST)"].notna(), fmt(combined_display["_수집시각_dt"]))
    df_all["수집시각(KST)"] = df_all["수집시각(KST)"].fillna(pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M"))

    # NEW: 이번 실행에서 신규만
    df_new_final = combined_display.loc[combined_display["_is_new"] == True, out_cols].copy()
    df_new_final = df_new_final.sort_values(["수집시각(KST)","발행일(KST)"], ascending=False)

    # 5) 저장
    df_all.to_csv(DATA_DIR / "ALL.csv", index=False, encoding="utf-8-sig")
    for kw, g in df_all.groupby("키워드", sort=False):
        g.to_csv(DATA_DIR / f"{safe_name(kw)}.csv", index=False, encoding="utf-8-sig")
    df_new_final.to_csv(DATA_DIR / "NEW_latest.csv", index=False, encoding="utf-8-sig")


    print("🎉 저장 완료")

if __name__ == "__main__":
    main()
