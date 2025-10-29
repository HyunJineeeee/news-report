# web_news.py
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
KEYWORDS = ["일학습병행", "직업훈련", "고용노동부", "한국산업인력공단"]
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
    """구글뉴스 링크 정규화(utm_*, hl/gl/ceid/oc 제거)"""
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
    """리다이렉트를 따라가 최종(원문) URL 반환. 실패시 원본 유지."""
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
            "원문링크": final_link,                # 엑셀 하이퍼링크용
            "출처": extract_domain(final_link) or extract_domain(link),
            "발행일_UTC": pub_ts,
            "수집시각_UTC": collected_at,
            "_정규화링크": normalize_url(link),    # 중복제거 키(구글뉴스 기준)
        })
    print(f"✅ '{keyword}' {len(rows)}건")
    return rows

# ============== 메인 ==============
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    # 1) 기존 ALL.csv 로드
    all_path = DATA_DIR / "ALL.csv"
    if all_path.exists():
        df_existing = pd.read_csv(all_path, dtype=str, encoding="utf-8-sig")
        df_existing["발행일(KST)"] = pd.to_datetime(df_existing["발행일(KST)"], errors="coerce")
        df_existing["수집시각(KST)"] = pd.to_datetime(df_existing["수집시각(KST)"], errors="coerce")
    else:
        df_existing = pd.DataFrame(columns=["키워드","제목","원문링크","발행일(KST)","수집시각(KST)","출처"])

    if not df_existing.empty:
        # 내부용 컬럼 재구성
        # 과거 파일에도 _정규화링크가 없을 수 있으므로 원문링크로라도 생성 시도
        base_norm = df_existing.get("원문링크", pd.Series("", index=df_existing.index)).fillna("")
        df_existing["_정규화링크"] = base_norm.apply(normalize_url)
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

    # 4) 표시용 DF
    out_cols = ["키워드","제목","원문링크","발행일(KST)","수집시각(KST)","출처"]
    combined_display = combined.sort_values("_수집시각_dt", ascending=False, na_position="last")
    df_all = combined_display[out_cols].copy()

    # ⚠ 발행일은 절대 보정하지 않음(원본에 없으면 빈 칸 유지)
    # 수집시각은 비어있으면 내부 타임스탬프로 채우고, 그래도 없으면 지금(KST)
    now_kst = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M")
    backfill_collect = pd.to_datetime(combined_display["_수집시각_dt"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    df_all["수집시각(KST)"] = df_all["수집시각(KST)"].mask(df_all["수집시각(KST)"].isna(), backfill_collect)
    df_all["수집시각(KST)"] = df_all["수집시각(KST)"].fillna(now_kst)

    # NEW: 이번 실행에서 '신규'만
    df_new_final = combined_display.loc[combined_display["_is_new"] == True, out_cols].copy()
    df_new_final = df_new_final.sort_values(["수집시각(KST)","발행일(KST)"], ascending=False)

    # --- 날짜 형식을 문자열(24시간제)로 강제 변환 ---
    for col in ["발행일(KST)", "수집시각(KST)"]:
        if col in df_all.columns:
        df_all[col] = pd.to_datetime(df_all[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        if col in df_new_final.columns:
        df_new_final[col] = pd.to_datetime(df_new_final[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

    # 5) 저장
    df_all.to_csv(DATA_DIR / "ALL.csv", index=False, encoding="utf-8-sig")
    for kw, g in df_all.groupby("키워드", sort=False):
        g.to_csv(DATA_DIR / f"{safe_name(kw)}.csv", index=False, encoding="utf-8-sig")
    # 최신본만 저장 (날짜 버전 파일은 생성하지 않음)
    df_new_final.to_csv(DATA_DIR / "NEW_latest.csv", index=False, encoding="utf-8-sig")

    print("🎉 저장 완료")

if __name__ == "__main__":
    main()
