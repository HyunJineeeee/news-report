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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import google.generativeai as genai
from newspaper import Article

# ============== 설정 ==============
KEYWORDS = ["일학습병행", "직업훈련", "고용노동부", "한국산업인력공단"]
DATA_DIR = Path("data")

# 환경변수 로드
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EMAIL_USER = os.environ.get("EMAIL_USER")        # 보내는 사람 이메일
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD") # 앱 비밀번호
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER") # 받는 사람 이메일

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("[WARN] GEMINI_API_KEY 미설정. 요약 불가.")

# ============== 유틸 ==============
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "HEAD"], raise_on_status=False)
    ad = HTTPAdapter(max_retries=retries)
    s.mount("http://", ad)
    s.mount("https://", ad)
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; NewsCrawler/1.0)"})
    return s

def normalize_url(url: str) -> str:
    try:
        p = urllib.parse.urlsplit(url)
        q = urllib.parse.parse_qsl(p.query, keep_blank_values=True)
        kept = []
        for k, v in q:
            kl = k.lower()
            if kl.startswith("utm_") or kl in {"hl", "gl", "ceid", "oc"}: continue
            kept.append((k, v))
        kept.sort(key=lambda x: x[0])
        nq = urllib.parse.urlencode(kept, doseq=True)
        return urllib.parse.urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path, nq, ""))
    except: return url

def extract_domain(url: str) -> str:
    try: return urllib.parse.urlsplit(url).netloc.lower()
    except: return ""

def parse_pub_date(text: str):
    if not text: return pd.NaT
    return pd.to_datetime(text, utc=True, errors="coerce")

def utc_to_kst_str(utc_ts):
    if utc_ts is None or pd.isna(utc_ts): return ""
    try:
        ts = pd.to_datetime(utc_ts, utc=True, errors="coerce")
        if pd.isna(ts): return ""
        return ts.tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M")
    except: return ""

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
    except: return url

# ============== AI & 본문 추출 ==============
def extract_article_content(url: str) -> str:
    try:
        article = Article(url, language='ko')
        article.download()
        article.parse()
        text = article.text.strip()
        return text if len(text) >= 50 else ""
    except: return ""

def summarize_with_gemini(text: str) -> str:
    if not GEMINI_API_KEY or not text: return ""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"너는 뉴스 요약 비서야. 아래 기사를 한국어로 핵심만 3줄(불렛 포인트 - 사용)로 요약해줘.\n\n기사 내용:\n{text[:5000]}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except: return "요약 실패"

# ============== 이메일 발송 (NEW) ==============
def send_email_report(df_new):
    """신규 기사가 있을 때만 이메일 발송"""
    if not EMAIL_USER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        print("[WARN] 이메일 설정 누락. 이메일을 보내지 않습니다.")
        return
    
    if df_new.empty:
        print("📭 신규 기사가 없어 이메일을 보내지 않습니다.")
        return

    try:
        # 이메일 본문 (HTML) 만들기
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .news-item {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .title {{ color: #2c3e50; font-size: 18px; font-weight: bold; text-decoration: none; }}
                .meta {{ color: #7f8c8d; font-size: 12px; margin-bottom: 10px; }}
                .summary {{ background-color: #f9f9f9; padding: 10px; border-left: 4px solid #3498db; }}
                .keyword {{ display: inline-block; background: #eee; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-right: 5px; }}
            </style>
        </head>
        <body>
            <h2>📢 오늘의 직업훈련 뉴스 리포트 ({len(df_new)}건)</h2>
            <p>오늘 수집된 새로운 기사 요약입니다.</p>
            <hr>
        """

        for _, row in df_new.iterrows():
            summ_html = row['요약'].replace('\n', '<br>')
            html_body += f"""
            <div class="news-item">
                <div>
                    <span class="keyword">{row['키워드']}</span>
                    <span class="meta">{row['출처']} | {row['발행일(KST)']}</span>
                </div>
                <a href="{row['원문링크']}" class="title" target="_blank">{row['제목']}</a>
                <div class="summary">
                    {summ_html}
                </div>
            </div>
            """
        
        html_body += "</body></html>"

        # 메일 객체 생성
        msg = MIMEMultipart()
        msg['Subject'] = f"📰 [뉴스리포트] {pd.Timestamp.now().strftime('%Y-%m-%d')} 신규기사 알림"
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_RECEIVER
        msg.attach(MIMEText(html_body, 'html'))

        # SMTP 서버 접속 (Gmail 기준)
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)
        
        print(f"📧 이메일 발송 성공! (To: {EMAIL_RECEIVER})")

    except Exception as e:
        print(f"❌ 이메일 발송 실패: {e}")

# ============== 크롤링 ==============
def crawl_google_news_rss(session, keyword):
    q = urllib.parse.quote(keyword)
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
    except: return []

    soup = BeautifulSoup(resp.text, "xml")
    rows = []
    collected_at_utc = pd.Timestamp.now(tz="UTC")
    
    for it in soup.find_all("item"):
        title = it.title.text if it.title else ""
        link = it.link.text if it.link else ""
        final_link = resolve_final_url(session, link)
        
        rows.append({
            "키워드": keyword,
            "제목": title,
            "원문링크": final_link,
            "출처": extract_domain(final_link),
            "발행일_UTC": parse_pub_date(it.pubDate.text if it.pubDate else ""),
            "수집시각_UTC": collected_at_utc,
            "_정규화링크": normalize_url(final_link),
            "요약": ""
        })
    print(f"✅ '{keyword}' {len(rows)}건")
    return rows

# ============== 메인 ==============
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    all_path = DATA_DIR / "ALL.csv"
    req_cols = ["키워드","제목","원문링크","발행일(KST)","수집시각(KST)","출처","요약",
                "_정규화링크","_발행일_dt","_수집시각_dt","_is_new"]
    
    if all_path.exists():
        df_existing = pd.read_csv(all_path, dtype=str, encoding="utf-8-sig")
        for c in req_cols: 
            if c not in df_existing.columns: df_existing[c] = ""
        existing_links = set(df_existing["_정규화링크"].dropna().astype(str))
    else:
        df_existing = pd.DataFrame(columns=req_cols)
        existing_links = set()

    raw_rows = []
    for kw in KEYWORDS:
        raw_rows.extend(crawl_google_news_rss(session, kw))
        time.sleep(0.5)
    
    if not raw_rows: return

    df_crawled = pd.DataFrame(raw_rows)
    df_crawled["_is_new"] = ~df_crawled["_정규화링크"].astype(str).isin(existing_links)
    df_crawled = df_crawled.drop_duplicates(subset=["_정규화링크"], keep="first")
    
    df_to_process = df_crawled[df_crawled["_is_new"] == True].copy()
    print(f"🔎 신규 {len(df_to_process)}건 발견.")

    processed_rows = []
    for idx, row in df_to_process.iterrows():
        print(f"   Processing: {row['제목'][:20]}...")
        content = extract_article_content(row["원문링크"])
        if content:
            summary = summarize_with_gemini(content)
            time.sleep(4)
        else:
            summary = "본문 추출 실패"
        row["요약"] = summary
        processed_rows.append(row)

    df_new_processed = pd.DataFrame(processed_rows) if processed_rows else pd.DataFrame(columns=df_crawled.columns)

    if not df_new_processed.empty:
        df_new_processed["발행일(KST)"] = df_new_processed["발행일_UTC"].apply(utc_to_kst_str)
        df_new_processed["수집시각(KST)"] = df_new_processed["수집시각_UTC"].apply(utc_to_kst_str)
        df_new_processed["_발행일_dt"] = pd.to_datetime(df_new_processed["발행일(KST)"], errors="coerce")
        df_new_processed["_수집시각_dt"] = pd.to_datetime(df_new_processed["수집시각(KST)"], errors="coerce")
        
        # ★★★ 이메일 발송 실행 ★★★
        send_email_report(df_new_processed)

    df_final_new = df_new_processed[req_cols] if not df_new_processed.empty else pd.DataFrame(columns=req_cols)
    combined = pd.concat([df_existing, df_final_new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["_정규화링크"], keep="last").sort_values("_수집시각_dt", ascending=False)

    display_cols = ["키워드","제목","요약","원문링크","발행일(KST)","수집시각(KST)","출처"]
    combined[display_cols].to_csv(DATA_DIR / "ALL.csv", index=False, encoding="utf-8-sig")
    for kw, g in combined.groupby("키워드"):
        g[display_cols].to_csv(DATA_DIR / f"{safe_name(kw)}.csv", index=False, encoding="utf-8-sig")
    
    if not df_new_processed.empty:
        df_new_processed[display_cols].to_csv(DATA_DIR / "NEW_latest.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=display_cols).to_csv(DATA_DIR / "NEW_latest.csv", index=False, encoding="utf-8-sig")
    
    print("🎉 완료")

if __name__ == "__main__":
    main()
