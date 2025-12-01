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
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ============== 유틸 ==============
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "HEAD"], raise_on_status=False)
    ad = HTTPAdapter(max_retries=retries)
    s.mount("http://", ad)
    s.mount("https://", ad)
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
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

def resolve_final_url(session: requests.Session, url: str, timeout: float = 10.0) -> str:
    try:
        # 구글 리다이렉트가 봇을 감지하면 중간 페이지에서 멈춤. 
        # 최대한 브라우저인척 헤더를 넣어서 시도
        r = session.get(url, allow_redirects=True, timeout=timeout)
        return r.url
    except: return url

# ============== AI & 본문 추출 ==============
def extract_article_content(url: str) -> str:
    try:
        # User-Agent 설정이 된 config 객체를 사용하면 성공률이 조금 오름
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
        prompt = (
            "너는 뉴스 리포트 비서야. 아래 기사 내용을 한국어로 2~3줄로 요약해줘.\n"
            "단, 기사 제목에 있는 내용을 단순히 반복하지 말고, 제목이 설명하지 못하는 '구체적인 수치', '배경', '향후 계획' 위주로 요약해.\n"
            "문장은 '- '로 시작하는 개조식으로 작성해줘.\n\n"
            f"기사 내용:\n{text[:5000]}"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except: return ""

def clean_html_tags(text: str) -> str:
    """RSS description에 있는 HTML 태그 제거"""
    return re.sub(r'<[^>]+>', '', text).strip()

# ============== 이메일 발송 ==============
def send_email_report(df_new):
    if not EMAIL_USER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        print("[WARN] 이메일 설정 누락. 발송 생략.")
        return
    
    if df_new.empty: return

    today_str = pd.Timestamp.now(tz="Asia/Seoul").strftime('%Y-%m-%d')
    subject = f"[일병리포트] {today_str} 신규기사 알림"

    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Apple SD Gothic Neo', 'Malgun Gothic', Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #f4f6f8; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
            .keyword-group {{ margin-bottom: 30px; border-bottom: 2px solid #eee; padding-bottom: 20px; }}
            .keyword-title {{ color: #2980b9; font-size: 18px; font-weight: bold; margin-bottom: 15px; border-left: 5px solid #2980b9; padding-left: 10px; }}
            .news-item {{ margin-bottom: 15px; }}
            .news-title {{ font-size: 15px; font-weight: bold; color: #2c3e50; text-decoration: none; }}
            .news-title:hover {{ text-decoration: underline; }}
            .news-meta {{ font-size: 12px; color: #7f8c8d; margin-left: 5px; }}
            .news-summary {{ margin-top: 5px; margin-left: 15px; font-size: 13px; color: #555; background-color: #fafafa; padding: 8px; border-radius: 4px; }}
            .footer {{ font-size: 11px; color: #aaa; text-align: center; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2 style="margin:0;">📢 오늘의 직업훈련 뉴스 리포트</h2>
                <p style="margin:5px 0 0 0; font-size:14px; color:#666;">
                    총 {len(df_new)}건의 새로운 기사가 수집되었습니다.
                </p>
            </div>
    """

    grouped = df_new.groupby("키워드")
    
    for kw in KEYWORDS:
        if kw in grouped.groups:
            group_df = grouped.get_group(kw)
            html_body += f'<div class="keyword-group">'
            html_body += f'<div class="keyword-title">📃 키워드: {kw}</div>'
            
            for i, (_, row) in enumerate(group_df.iterrows(), 1):
                title = row['제목']
                link = row['원문링크']
                source = row['출처']
                date = row['발행일(KST)']
                summary = row['요약']

                # 요약이 비어있으면 표시하지 않거나 안내 문구
                if not summary:
                    summary_html = "<span style='color:#ccc; font-size:12px;'>(요약 없음)</span>"
                else:
                    summary_html = summary.replace('\n', '<br>')

                html_body += f"""
                <div class="news-item">
                    <div>
                        <span style="color:#e67e22; font-weight:bold; margin-right:5px;">{i}.</span>
                        <a href="{link}" class="news-title" target="_blank">{title}</a>
                        <span class="news-meta">({source} | {date})</span>
                    </div>
                    <div class="news-summary">
                        {summary_html}
                    </div>
                </div>
                """
            html_body += '</div>'

    html_body += """
            <div class="footer">
                본 메일은 자동화 봇에 의해 발송되었습니다.<br>
                GitHub Actions & Google Gemini API
            </div>
        </div>
    </body>
    </html>
    """

    try:
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_RECEIVER
        msg.attach(MIMEText(html_body, 'html'))

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"📧 이메일 발송 성공! ({subject})")
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
        # RSS에 포함된 기본 설명글 (HTML 태그 포함됨)
        description = it.description.text if it.description else ""
        
        final_link = resolve_final_url(session, link)
        
        rows.append({
            "키워드": keyword,
            "제목": title,
            "원문링크": final_link,
            "출처": extract_domain(final_link) or extract_domain(link),
            "발행일_UTC": parse_pub_date(it.pubDate.text if it.pubDate else ""),
            "수집시각_UTC": collected_at_utc,
            "_정규화링크": normalize_url(final_link),
            "요약": "", # 나중에 채움
            "_rss_desc": clean_html_tags(description) # 백업용 RSS 설명 저장
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
    
    if not raw_rows: 
        print("수집된 데이터가 없습니다.")
        return

    df_crawled = pd.DataFrame(raw_rows)
    df_crawled["_is_new"] = ~df_crawled["_정규화링크"].astype(str).isin(existing_links)
    df_crawled = df_crawled.drop_duplicates(subset=["_정규화링크"], keep="first")
    
    df_to_process = df_crawled[df_crawled["_is_new"] == True].copy()
    print(f"🔎 신규 {len(df_to_process)}건 발견.")

    processed_rows = []
    for idx, row in df_to_process.iterrows():
        print(f"   Processing: {row['제목'][:20]}...")
        content = extract_article_content(row["원문링크"])
        
        summary = ""
        # 1. 본문이 있으면 AI 요약 시도
        if content:
            ai_summary = summarize_with_gemini(content)
            if ai_summary and "실패" not in ai_summary:
                summary = ai_summary
                time.sleep(4)
        
        # 2. 본문 추출 실패했거나 AI 요약 실패시 -> RSS 기본 설명(description) 사용
        if not summary or summary == "요약 생성 실패":
            # RSS 설명이 너무 짧으면(제목과 같으면) 그냥 "내용 확인" 문구로 대체
            rss_desc = row.get("_rss_desc", "")
            if len(rss_desc) > 10 and rss_desc != row['제목']:
                summary = f"- (AI 요약 불가) {rss_desc[:150]}..."
            else:
                summary = "👉 원문 링크에서 내용을 확인하세요."

        row["요약"] = summary
        processed_rows.append(row)

    if processed_rows:
        df_new_processed = pd.DataFrame(processed_rows)
    else:
        df_new_processed = pd.DataFrame(columns=df_crawled.columns)

    if not df_new_processed.empty:
        df_new_processed["발행일(KST)"] = df_new_processed["발행일_UTC"].apply(utc_to_kst_str)
        df_new_processed["수집시각(KST)"] = df_new_processed["수집시각_UTC"].apply(utc_to_kst_str)
        df_new_processed["_발행일_dt"] = pd.to_datetime(df_new_processed["발행일(KST)"], errors="coerce")
        df_new_processed["_수집시각_dt"] = pd.to_datetime(df_new_processed["수집시각(KST)"], errors="coerce")
        
        send_email_report(df_new_processed)

    # 저장 시 _rss_desc 컬럼은 제외
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
