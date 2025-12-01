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
from datetime import datetime, timedelta

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
    # 일반 브라우저처럼 위장
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    })
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
    """구글 뉴스 리다이렉트 최종 주소 추적 (강화됨)"""
    try:
        # 1. news.google.com이 아니면 그냥 반환
        if "news.google.com" not in url:
            return url
            
        # 2. 리다이렉트 추적
        r = session.get(url, allow_redirects=True, timeout=timeout)
        return r.url
    except: 
        return url

# ============== AI & 본문 추출 ==============
def extract_article_content(url: str) -> str:
    try:
        # news.google.com 링크는 newspaper3k가 못 읽음. 원문이어야 함.
        if "news.google.com" in url:
            return "" 

        article = Article(url, language='ko')
        article.download()
        article.parse()
        text = article.text.strip()
        return text if len(text) >= 100 else "" # 너무 짧으면 실패 간주
    except: return ""

def summarize_with_gemini(text: str) -> str:
    if not GEMINI_API_KEY or not text: return ""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            "너는 직업훈련 뉴스 요약 비서야. 아래 기사 내용을 한국어로 2~3줄로 요약해줘.\n"
            "단, 기사 제목에 있는 내용을 단순히 반복하지 말고, 제목이 설명하지 못하는 '구체적인 수치', '배경', '향후 계획' 위주로 요약해.\n"
            "문장은 '- '로 시작하는 개조식으로 작성해줘.\n\n"
            f"기사 내용:\n{text[:5000]}"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except: return ""

# ============== 이메일 발송 ==============
def send_email_report(df_new, target_date_str):
    if not EMAIL_USER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        print("[WARN] 이메일 설정 누락. 발송 생략.")
        return
    
    if df_new.empty: 
        print("📭 발송할 기사가 없습니다.")
        return

    subject = f"[일병리포트] {target_date_str} 주요 뉴스 알림"

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
                <h2 style="margin:0;">📢 어제({target_date_str})의 직업훈련 뉴스</h2>
                <p style="margin:5px 0 0 0; font-size:14px; color:#666;">
                    총 {len(df_new)}건의 기사가 수집되었습니다.
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

                if not summary:
                    # 요약이 정말 없을 때
                    summary_html = "<span style='color:#ccc; font-size:12px;'>👉 클릭하여 원문 확인</span>"
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
    # when:1d 옵션으로 최근 24시간(또는 하루) 기사만 검색 유도
    url = f"https://news.google.com/rss/search?q={q}+when:1d&hl=ko&gl=KR&ceid=KR:ko"
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
        pub_date_str = it.pubDate.text if it.pubDate else ""
        pub_ts_utc = parse_pub_date(pub_date_str)
        
        # 1차 리다이렉트 해석 시도 (중요: AI 요약을 위해 진짜 주소 필요)
        final_link = resolve_final_url(session, link)
        
        rows.append({
            "키워드": keyword,
            "제목": title,
            "원문링크": final_link,
            "출처": extract_domain(final_link) or extract_domain(link),
            "발행일_UTC": pub_ts_utc,
            "수집시각_UTC": collected_at_utc,
            "_정규화링크": normalize_url(final_link),
            "요약": "", 
            "_rss_desc": "" # description은 제거 (요약 퀄리티 저하 원인)
        })
    return rows

# ============== 메인 ==============
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    # 타겟 날짜: "어제" (한국 시간 기준)
    now_kst = pd.Timestamp.now(tz="Asia/Seoul")
    yesterday_kst = now_kst - pd.Timedelta(days=1)
    target_date_str = yesterday_kst.strftime("%Y-%m-%d")
    print(f"🎯 타겟 날짜(어제): {target_date_str} (기사 필터링 기준)")

    all_path = DATA_DIR / "ALL.csv"
    req_cols = ["키워드","제목","원문링크","발행일(KST)","수집시각(KST)","출처","요약",
                "_정규화링크","_발행일_dt","_수집시각_dt","_is_new"]
    
    # 1. 기존 데이터 로드 및 타입 강제 변환 (에러 수정 핵심)
    if all_path.exists():
        df_existing = pd.read_csv(all_path, dtype=str, encoding="utf-8-sig")
        for c in req_cols: 
            if c not in df_existing.columns: df_existing[c] = ""
        
        # ★★★ 여기서 날짜 타입으로 강제 변환해줘야 에러가 안 남 ★★★
        df_existing["_수집시각_dt"] = pd.to_datetime(df_existing["_수집시각_dt"], errors="coerce")
        
        existing_links = set(df_existing["_정규화링크"].dropna().astype(str))
    else:
        df_existing = pd.DataFrame(columns=req_cols)
        existing_links = set()

    # 2. 크롤링
    raw_rows = []
    for kw in KEYWORDS:
        print(f"📡 수집 중: {kw}...")
        raw_rows.extend(crawl_google_news_rss(session, kw))
        time.sleep(1) # 차단 방지 딜레이
    
    if not raw_rows: 
        print("수집된 데이터가 없습니다.")
        return

    df_crawled = pd.DataFrame(raw_rows)
    
    # 3. 날짜 필터링 (어제 날짜인 것만 남김)
    # 발행일(UTC)을 KST로 변환 후 문자열 비교
    df_crawled["발행일(KST)"] = df_crawled["발행일_UTC"].apply(utc_to_kst_str)
    # 'YYYY-MM-DD' 부분만 잘라서 어제 날짜와 비교
    df_crawled = df_crawled[df_crawled["발행일(KST)"].str.startswith(target_date_str)]
    
    if df_crawled.empty:
        print(f"📅 {target_date_str} 날짜에 해당하는 기사가 없습니다.")
        return

    # 4. 중복 제거 (기존 DB에 없는 것만)
    df_crawled["_is_new"] = ~df_crawled["_정규화링크"].astype(str).isin(existing_links)
    df_crawled = df_crawled.drop_duplicates(subset=["_정규화링크"], keep="first")
    
    df_to_process = df_crawled[df_crawled["_is_new"] == True].copy()
    print(f"🔎 {target_date_str} 기사 중 신규 {len(df_to_process)}건 발견.")

    # 5. 본문 추출 및 요약
    processed_rows = []
    for idx, row in df_to_process.iterrows():
        print(f"   Processing: {row['제목'][:20]}...")
        
        # 진짜 URL이어야만 본문 추출 가능
        real_url = row["원문링크"]
        content = extract_article_content(real_url)
        
        summary = ""
        if content:
            # AI 요약 시도
            ai_summary = summarize_with_gemini(content)
            if ai_summary:
                summary = ai_summary
                time.sleep(4) # API 제한 고려
        
        # AI 실패 시: '본문 추출 실패' 대신 RSS 제목 반복을 피하고 깔끔하게 처리
        if not summary:
             summary = "" # 공란으로 두면 메일 템플릿에서 '클릭하여 확인'으로 처리

        row["요약"] = summary
        processed_rows.append(row)

    if processed_rows:
        df_new_processed = pd.DataFrame(processed_rows)
    else:
        df_new_processed = pd.DataFrame(columns=df_crawled.columns)

    # 6. 메일 발송 (어제 기사만 모아서)
    if not df_new_processed.empty:
        # 나머지 컬럼 채우기
        df_new_processed["수집시각(KST)"] = df_new_processed["수집시각_UTC"].apply(utc_to_kst_str)
        df_new_processed["_발행일_dt"] = pd.to_datetime(df_new_processed["발행일(KST)"], errors="coerce")
        df_new_processed["_수집시각_dt"] = pd.to_datetime(df_new_processed["수집시각(KST)"], errors="coerce")
        
        send_email_report(df_new_processed, target_date_str)

    # 7. 저장 (기존 + 신규)
    df_final_new = df_new_processed[req_cols] if not df_new_processed.empty else pd.DataFrame(columns=req_cols)
    
    # 병합
    combined = pd.concat([df_existing, df_final_new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["_정규화링크"], keep="last")
    
    # 정렬 (여기서 에러 안 나게 _수집시각_dt가 datetime인지 확인)
    combined["_수집시각_dt"] = pd.to_datetime(combined["_수집시각_dt"], errors="coerce")
    combined = combined.sort_values("_수집시각_dt", ascending=False)

    display_cols = ["키워드","제목","요약","원문링크","발행일(KST)","수집시각(KST)","출처"]
    combined[display_cols].to_csv(DATA_DIR / "ALL.csv", index=False, encoding="utf-8-sig")
    
    # 최신 파일은 '오늘 수집한 어제 뉴스'만 저장
    if not df_new_processed.empty:
        df_new_processed[display_cols].to_csv(DATA_DIR / "NEW_latest.csv", index=False, encoding="utf-8-sig")
    
    print("🎉 완료")

if __name__ == "__main__":
    main()
