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


# -----------------------------
# 설정
# -----------------------------
KEYWORDS = ["일학습병행", "직업훈련"]     # 필요 시 수정
DATA_DIR = Path("data")
LOG_DIR = Path("logs")
WRITE_LOG_FILE = True


# -----------------------------
# 유틸
# -----------------------------
def make_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; NewsCrawler/1.0)"})
    return sess


def normalize_url(url: str) -> str:
    """구글뉴스 링크 중복 제거용 정규화(utm_*, hl/gl/ceid/oc 제거)"""
    try:
        parsed = urllib.parse.urlsplit(url)
        q = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        kept = []
        for k, v in q:
            kl = k.lower()
            if kl.startswith("utm_") or kl in {"hl", "gl", "ceid", "oc"}:
                continue
            kept.append((k, v))
        kept.sort(key=lambda x: x[0])
        new_query = urllib.parse.urlencode(kept, doseq=True)
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


def resolve_final_url(session: requests.Session, url: str, timeout: float = 10.0) -> str:
    """
    구글뉴스 등 리다이렉트 링크를 추적하여 실제 기사(원문) URL을 반환.
    - HEAD로 빠르게 시도 후, 필요하면 GET로 재시도
    - 실패 시 원본 url을 그대로 반환
    """
    try:
        r = session.head(url, allow_redirects=True, timeout=timeout)
        final_url = r.url
        if not final_url or final_url == url:
            r = session.get(url, allow_redirects=True, timeout=timeout)
            final_url = r.url
        return final_url or url
    except Exception:
        return url


# -----------------------------
# 크롤링
# -----------------------------
def crawl_google_news_rss(session: requests.Session, keyword: str):
    encoded_kw = urllib.parse.quote(keyword)
    url = f"https://news.google.com/rss/search?q={encoded_kw}&hl=ko&gl=KR&ceid=KR:ko"

    resp = session.get(url, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "xml")  # lxml 필요
    items = soup.find_all("item")

    collected_at = pd.Timestamp.utcnow()
    rows = []
    for item in items:
        title = item.title.text if item.title else ""
        link = item.link.text if item.link else ""
        pub_date_raw = item.pubDate.text if item.pubDate else ""
        pub_ts = parse_pub_date(pub_date_raw)

        # 긴 구글뉴스 링크 → 원문(최종) 링크로 해석
        final_link = resolve_final_url(session, link)

        rows.append({
            "키워드": keyword,
            "제목": title,
            # 내부적으로는 원본 링크를 정규화하여 중복 제거 키로 사용
            "_정규화링크": normalize_url(link),
            # 표시/하이퍼링크용은 원문 링크 사용
            "원문링크": final_link,
            "출처": extract_domain(final_link) or extract_domain(link),
            "발행일_UTC": pub_ts,
            "수집시각_UTC": collected_at,
        })
    print(f"✅ '{keyword}' 뉴스 {len(rows)}건 수집")
    return rows


# -----------------------------
# 메인
# -----------------------------
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    log_lines = []
    t0 = time.time()
    ts_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1) 기존 ALL.csv 로드(누적)
    all_path = DATA_DIR / "ALL.csv"
    if all_path.exists():
        df_existing = pd.read_csv(all_path, dtype=str, encoding="utf-8-sig")
        for c in ["발행일(KST)", "수집시각(KST)"]:
            df_existing[c] = pd.to_datetime(df_existing[c], errors="coerce")
        # 내부용 컬럼 재생성
        if "_정규화링크" not in df_existing.columns:
            # 과거 버전과 호환: 원문링크가 있으면 그걸, 없으면 빈 값
            df_existing["_정규화링크"] = df_existing.get("원문링크", pd.Series("", index=df_existing.index))
        df_existing["_발행일_dt"] = pd.to_datetime(df_existing["발행일(KST)"], errors="coerce")
        df_existing["_수집시각_dt"] = pd.to_datetime(df_existing["수집시각(KST)"], errors="coerce")
        df_existing["_is_new"] = False
    else:
        df_existing = pd.DataFrame(columns=[
            "키워드","제목","원문링크","발행일(KST)","수집시각(KST)","출처",
            "_정규화링크","_발행일_dt","_수집시각_dt","_is_new"
        ])

    log_lines.append(f"[{ts_label}] 기존 ALL.csv 로드: {len(df_existing)}건")

    # 2) 신규 수집
    all_rows = []
    per_kw_counts = {}
    errors = []
    for kw in KEYWORDS:
        try:
            rows = crawl_google_news_rss(session, kw)
            per_kw_counts[kw] = len(rows)
            all_rows.extend(rows)
            time.sleep(0.5)
        except Exception as e:
            msg = f"❌ '{kw}' 수집 오류: {e}"
            print(msg)
            errors.append(msg)
    log_lines.append("수집 결과(키워드별): " + ", ".join(f"{k}={v}" for k, v in per_kw_counts.items()))

    df_new_raw = pd.DataFrame(all_rows)
    if not df_new_raw.empty:
        df_new_raw["발행일(KST)"] = df_new_raw["발행일_UTC"].apply(to_kst_str_from_utc)
        df_new_raw["수집시각(KST)"] = df_new_raw["수집시각_UTC"].apply(to_kst_str_from_utc)
        # 내부 정렬/중복 제거용 타임스탬프
        df_new_raw["_발행일_dt"] = pd.to_datetime(df_new_raw["발행일(KST)"], errors="coerce")
        df_new_raw["_수집시각_dt"] = pd.to_datetime(df_new_raw["수집시각(KST)"], errors="coerce")

        existing_norm = set(df_existing["_정규화링크"].dropna().astype(str))
        df_new_raw["_is_new"] = ~df_new_raw["_정규화링크"].astype(str).isin(existing_norm)
    else:
        df_new_raw = pd.DataFrame(columns=list(df_existing.columns))

    # 3) 병합 + 중복 제거(최신 수집 우선)
    combined = pd.concat([df_existing, df_new_raw], ignore_index=True)
    combined = combined.sort_values("_수집시각_dt", ascending=False, na_position="last")

    before = len(combined)
    combined = combined.drop_duplicates(subset=["_정규화링크"], keep="first")
    combined["_발행일_일"] = combined["_발행일_dt"].dt.date
    combined = combined.drop_duplicates(subset=["제목", "_발행일_일"], keep="first")
    after = len(combined)

    log_lines.append(f"중복 제거: {before} -> {after} (제거 {before - after}건)")

    # 4) 표시용 정렬 및 컬럼
    combined_display = combined.sort_values("_수집시각_dt", ascending=False, na_position="last")
    out_cols = ["키워드", "제목", "원문링크", "발행일(KST)", "수집시각(KST)", "출처"]
    df_all = combined_display[out_cols].copy()

    # 5) NEW (이번 실행 신규만)
    df_new_final = combined_display.loc[combined_display["_is_new"] == True, out_cols].copy()
    df_new_final = df_new_final.sort_values(["수집시각(KST)", "발행일(KST)"], ascending=False)

    # 6) CSV 저장
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # ALL
    df_all.to_csv(DATA_DIR / "ALL.csv", index=False, encoding="utf-8-sig")
    # 키워드별
    for kw, g in df_all.groupby("키워드", sort=False):
        g.to_csv(DATA_DIR / f"{safe_name(kw)}.csv", index=False, encoding="utf-8-sig")
    # NEW
    today = datetime.now().strftime("%Y%m%d")
    df_new_final.to_csv(DATA_DIR / f"NEW_{today}.csv", index=False, encoding="utf-8-sig")
    df_new_final.to_csv(DATA_DIR / "NEW_latest.csv", index=False, encoding="utf-8-sig")

    # 7) 로그(선택)
    elapsed = time.time() - t0
    log_lines.append(f"총 소요 시간: {elapsed:.2f}s")
    if errors:
        log_lines.append("에러:"); log_lines.extend(errors)
    if WRITE_LOG_FILE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"crawl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print(f"🧾 로그 저장: {log_path}")

    print(f"🎉 CSV 저장 완료: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
