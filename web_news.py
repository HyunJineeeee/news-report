# web_news.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib.parse
from pathlib import Path
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import os
import google.generativeai as genai
from newspaper import Article

# ============== 설정 ==============
KEYWORDS = ["일학습병행", "직업훈련", "고용노동부", "한국산업인력공단"]
DATA_DIR = Path("data")

# 구글 제미나이 API 설정 (환경변수에서 키를 가져옴)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("[WARN] GEMINI_API_KEY가 설정되지 않았습니다. 요약 기능이 작동하지 않습니다.")

# ============== 유틸 ==============
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    ad = HTTPAdapter(max_retries=retries)
    s.mount("http://", ad)
    s.mount("https://", ad)
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; NewsCrawler/1.0)"})
    return s

def normalize_url(url: str) -> str:
    """구글뉴스 링크 정규화"""
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
    try:
        return urllib.parse.urlsplit(url).netloc.lower()
    except Exception:
        return ""

def parse_pub_date(text: str):
    if not text:
        return pd.NaT
    return pd.to_datetime(text, utc=True, errors="coerce")

def utc_to_kst_str(utc_ts):
    if utc_ts is None or pd.isna(utc_ts):
        return ""
    try:
        ts = pd.to_datetime(utc_ts, utc=True, errors="coerce")
        if pd.isna(ts):
            return ""
        return ts.tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""

def safe_name(name: str) -> str:
    return re.sub(r"[\\/:*?\[\]]", "_", str(name))[:64] or "Sheet"

def resolve_final_url(session: requests.Session, url: str, timeout: float = 8.0) -> str:
    try:
        r = session.head(url, allow_redirects=True, timeout=timeout)
        fu = r.url
        if not fu or fu == url:
            r = session.get(url, allow_redirects=True, timeout=timeout)
            fu = r.url
        return fu or url
    except Exception:
        return url

# ============== AI 및 본문 추출 ==============
def extract_article_content(url: str) -> str:
    """newspaper3k를 이용해 본문 텍스트 추출"""
    try:
        article = Article(url, language='ko')
        article.download()
        article.parse()
        text = article.text.strip()
        if len(text) < 50: # 본문이 너무 짧으면 실패로 간주
            return ""
        return text
    except Exception:
        return ""

def summarize_with_gemini(text: str) -> str:
    """Google Gemini Flash 모델을 이용한 3줄 요약"""
    if not GEMINI_API_KEY:
        return "API Key 미설정"
    if not text:
        return "본문 없음"
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            "너는 뉴스 요약 비서야. 다음 기사 내용을 한국어로 핵심만 추려서 "
            "3개의 불렛 포인트(- )로 요약해줘. 어조는 건조하고 전문적으로 해줘.\n\n"
            f"기사 내용:\n{text[:5000]}" # 토큰 제한 고려하여 길이 절삭
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"요약 실패"

# ============== 크롤링 ==============
def crawl_google_news_rss(session: requests.Session, keyword: str):
    q = urllib.parse.quote(keyword)
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] RSS 접속 실패 ({keyword}): {e}")
        return []

    soup = BeautifulSoup(resp.text, "xml")
    items = soup.find_all("item")
    collected_at_utc = pd.Timestamp.now(tz="UTC")

    rows = []
    for it in items:
        title = it.title.text if it.title else ""
        link = it.link.text if it.link else ""
        pub_ts_utc = parse_pub_date(it.pubDate.text if it.pubDate else "")
        
        # 1차적으로 리다이렉트 해소 (본문 추출을 위해 필수)
        final_link = resolve_final_url(session, link)

        rows.append({
            "키워드": keyword,
            "제목": title,
            "원문링크": final_link,
            "출처": extract_domain(final_link) or extract_domain(link),
            "발행일_UTC": pub_ts_utc,
            "수집시각_UTC": collected_at_utc,
            "_정규화링크": normalize_url(final_link), # 정규화 기준을 final_link로 변경
            "요약": "" # 초기엔 빈 값
        })
    print(f"✅ '{keyword}' RSS {len(rows)}건 확인")
    return rows

# ============== 메인 ==============
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    # 1) 기존 데이터 로드
    all_path = DATA_DIR / "ALL.csv"
    required_cols = ["키워드","제목","원문링크","발행일(KST)","수집시각(KST)","출처","요약",
                     "_정규화링크","_발행일_dt","_수집시각_dt","_is_new"]
    
    if all_path.exists():
        df_existing = pd.read_csv(all_path, dtype=str, encoding="utf-8-sig")
        # 컬럼 보정
        for col in required_cols:
            if col not in df_existing.columns:
                df_existing[col] = ""
        # 기존 데이터의 링크 집합 (중복 확인용)
        existing_links = set(df_existing["_정규화링크"].dropna().astype(str))
    else:
        df_existing = pd.DataFrame(columns=required_cols)
        existing_links = set()

    # 2) 신규 수집 (일단 긁어옴)
    raw_rows = []
    for kw in KEYWORDS:
        raw_rows.extend(crawl_google_news_rss(session, kw))
        time.sleep(0.5)
    
    if not raw_rows:
        print("수집된 데이터가 없습니다.")
        return

    df_crawled = pd.DataFrame(raw_rows)

    # 3) 중복 제거 후 '진짜 신규' 식별
    # 정규화 링크 기준으로 기존에 없는 것만 필터링
    df_crawled["_is_new"] = ~df_crawled["_정규화링크"].astype(str).isin(existing_links)
    
    # 중복 제거 (이번 수집 내에서 중복 방지)
    df_crawled = df_crawled.drop_duplicates(subset=["_정규화링크"], keep="first")
    
    # 진짜 처리해야 할 신규 데이터
    df_to_process = df_crawled[df_crawled["_is_new"] == True].copy()
    
    print(f"🔎 전체 {len(df_crawled)}건 중 신규 기사 {len(df_to_process)}건 발견. 요약 시작...")

    # 4) 신규 데이터에 대해 본문 추출 & 요약 수행
    processed_rows = []
    # DataFrame 순회 대신 리스트로 변환 후 처리 (속도/안전성)
    for idx, row in df_to_process.iterrows():
        url = row["원문링크"]
        print(f"   Processing: {row['제목'][:20]}...")
        
        # A. 본문 추출
        content = extract_article_content(url)
        
        # B. 요약 (본문이 있을 때만)
        summary = ""
        if content:
            summary = summarize_with_gemini(content)
            # 무료 티어 속도 제한 고려 (분당 15회 등) -> 안전하게 4초 대기
            time.sleep(4) 
        else:
            summary = "본문 추출 실패"
        
        row["요약"] = summary
        processed_rows.append(row)

    # 처리된 신규 데이터 DF 생성
    if processed_rows:
        df_new_processed = pd.DataFrame(processed_rows)
    else:
        df_new_processed = pd.DataFrame(columns=df_crawled.columns)

    # 5) 기존 데이터와 병합 전 포맷팅
    # 날짜 변환
    if not df_new_processed.empty:
        df_new_processed["발행일(KST)"] = df_new_processed["발행일_UTC"].apply(utc_to_kst_str)
        df_new_processed["수집시각(KST)"] = df_new_processed["수집시각_UTC"].apply(utc_to_kst_str)
        df_new_processed["_발행일_dt"] = pd.to_datetime(df_new_processed["발행일(KST)"], errors="coerce")
        df_new_processed["_수집시각_dt"] = pd.to_datetime(df_new_processed["수집시각(KST)"], errors="coerce")
    
    # 기존 데이터와 스키마 맞추기 (이미 있는 데이터는 건드리지 않음)
    df_final_new = df_new_processed[required_cols] if not df_new_processed.empty else pd.DataFrame(columns=required_cols)
    
    # 병합
    combined = pd.concat([df_existing, df_final_new], ignore_index=True)
    
    # 최종 중복 제거 (링크 기준)
    combined = combined.drop_duplicates(subset=["_정규화링크"], keep="last") # 최신 정보(요약포함) 우선
    combined = combined.sort_values("_수집시각_dt", ascending=False)

    # 6) 저장
    # 표시용 컬럼 정의
    display_cols = ["키워드","제목","요약","원문링크","발행일(KST)","수집시각(KST)","출처"]
    
    # 메인 파일 저장
    combined[display_cols].to_csv(DATA_DIR / "ALL.csv", index=False, encoding="utf-8-sig")
    
    # 키워드별 파일 저장
    for kw, g in combined.groupby("키워드"):
        g[display_cols].to_csv(DATA_DIR / f"{safe_name(kw)}.csv", index=False, encoding="utf-8-sig")
        
    # 이번 실행의 신규 파일 저장
    if not df_new_processed.empty:
        df_new_processed[display_cols].to_csv(DATA_DIR / "NEW_latest.csv", index=False, encoding="utf-8-sig")
        print(f"🎉 신규 {len(df_new_processed)}건 요약 및 저장 완료")
    else:
        # 빈 파일이라도 생성하여 워크플로우 에러 방지
        pd.DataFrame(columns=display_cols).to_csv(DATA_DIR / "NEW_latest.csv", index=False, encoding="utf-8-sig")
        print("🎉 신규 기사 없음 (기존 데이터 유지)")

if __name__ == "__main__":
    main()
