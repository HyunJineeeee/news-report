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
from newspaper import Article, Config
import base64

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
    # 구글 봇 차단 우회용 헤더 및 쿠키
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
    })
    # 쿠키 추가 (Consent 페이지 우회 시도)
    s.cookies.set("CONSENT", "YES+KR.ko+V14+BX")
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

def get_real_url(session, url):
    """
    구글 뉴스 리다이렉트 URL을 추적하여 진짜 URL을 가져옵니다.
    단순 requests뿐만 아니라 base64 디코딩 로직을 사용하여 원문을 찾습니다.
    """
    if "news.google.com" not in url:
        return url
    
    try:
        # 1차 시도: 헤더와 쿠키를 달고 리다이렉트 추적
        r = session.get(url, allow_redirects=True, timeout=5)
        if "news.google.com" not in r.url:
            return r.url
            
        # 2차 시도: 구글 뉴스 URL 구조상 base64로 인코딩된 부분이 있을 수 있음 (단순화된 로직)
        # (구글의 암호화 방식은 복잡하여 완벽한 디코딩은 어려우나, 리다이렉트된 HTML 내에서 찾는 방식 사용)
        soup = BeautifulSoup(r.text, "html.parser")
        # 구글 리다이렉트 페이지의 <a> 태그나 <c-wiz> 등을 뒤져봄
        links = soup.find_all("a", href=True)
        for link in links:
            href = link['href']
            # 구글 내부 링크가 아니고 http로 시작하면 원문일 확률 높음
            if href.startswith("http") and "google.com" not in href:
                return href
                
        return r.url # 실패하면 원래 URL 반환
    except:
        return url

# ============== AI & 본문 추출 ==============
def extract_article_content(url: str) -> str:
    try:
        # 여전히 구글 링크라면 newspaper3k는 실패함
        if "news.google.com" in url:
            return ""

        config = Config()
        # 봇 차단 방지용 User-Agent
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        config.request_timeout = 10

        article = Article(url, language='ko', config=config)
        article.download()
        article.parse()
        text = article.text.strip()
        
        return text if len(text) >= 100 else "" 
    except Exception as e:
        return ""

def summarize_with_gemini(text: str) -> str:
    if not GEMINI_API_KEY or not text: return ""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            "너는 뉴스 요약 전문가야. 아래 뉴스 기사 내용을 읽고, "
            "바쁜 직장인이 핵심만 파악할 수 있도록 3줄 이내로 요약해줘.\n"
            "형식: '- '로 시작하는 문장.\n\n"
            f"기사 내용:\n{text[:4000]}"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except: return ""

# ============== 이메일 발송 (인라인 스타일 적용) ==============
def send_email_report(df_new, target_date_str):
    if not EMAIL_USER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        print("[WARN] 이메일 설정 누락. 발송 생략.")
        return
    
    if df_new.empty: 
        print("📭 발송할 기사가 없습니다.")
        return

    subject = f"[일병리포트] {target_date_str} 주요 뉴스 알림"

    # ★★★ 인라인 스타일(Inline Style) 적용 ★★★
    # <style> 태그를 쓰지 않고, 태그 안에 style="..."을 직접 넣어서 모든 이메일 클라이언트 호환성 확보
    
    html_body = f"""
    <div style="font-family: 'Malgun Gothic', sans-serif; background-color: #f4f4f4; padding: 20px; color: #333;">
        <div style="max-width: 700px; margin: 0 auto; background-color: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            
            <div style="text-align: center; margin-bottom: 30px; border-bottom: 2px solid #3498db; padding-bottom: 20px;">
                <h1 style="color: #2c3e50; font-size: 24px; margin: 0;">📰 {target_date_str} 뉴스 리포트</h1>
                <p style="color: #7f8c8d; font-size: 14px; margin-top: 10px;">
                    어제 수집된 총 <span style="color:#e67e22; font-weight:bold;">{len(df_new)}</span>건의 기사 요약입니다.
                </p>
            </div>
    """

    grouped = df_new.groupby("키워드")
    
    for kw in KEYWORDS:
        if kw in grouped.groups:
            group_df = grouped.get_group(kw)
            
            # 키워드 제목
            html_body += f"""
            <div style="margin-bottom: 30px;">
                <div style="background-color: #3498db; color: white; padding: 6px 15px; display: inline-block; border-radius: 15px; font-weight: bold; font-size: 16px; margin-bottom: 15px;">
                    # {kw}
                </div>
            """
            
            for idx, row in group_df.iterrows():
                title = row['제목']
                link = row['원문링크']
                source = row['출처']
                date = row['발행일(KST)']
                summary = row['요약']

                # 요약 HTML 처리
                if summary:
                    summary_html = summary.replace('\n', '<br>')
                    summary_style = "background-color: #f9f9f9; padding: 15px; border-left: 4px solid #3498db; color: #555; font-size: 14px; line-height: 1.6; border-radius: 4px;"
                else:
                    summary_html = "👉 클릭하여 원문 내용을 확인하세요."
                    summary_style = "background-color: #f0f0f0; padding: 10px; color: #888; font-size: 13px; text-align: center; border-radius: 4px;"

                # 기사 카드
                html_body += f"""
                <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; margin-bottom: 15px; background-color: #fff;">
                    <a href="{link}" target="_blank" style="font-size: 18px; font-weight: bold; color: #2c3e50; text-decoration: none; display: block; margin-bottom: 8px; line-height: 1.4;">
                        {title}
                    </a>
                    
                    <div style="font-size: 12px; color: #95a5a6; margin-bottom: 15px;">
                        {source} | {date}
                    </div>
                    
                    <div style="{summary_style}">
                        {summary_html}
                    </div>
                    
                    <div style="text-align: right; margin-top: 10px;">
                        <a href="{link}" target="_blank" style="display: inline-block; background-color: #ecf0f1; color: #555; padding: 5px 12px; border-radius: 4px; text-decoration: none; font-size: 12px;">
                            원문 보러가기 →
                        </a>
                    </div>
                </div>
                """
            html_body += '</div>' # 키워드 섹션 닫기

    html_body += """
            <div style="text-align: center; margin-top: 40px; font-size: 12px; color: #bdc3c7; border-top: 1px solid #eee; padding-top: 20px;">
                Automated by GitHub Actions & Google Gemini
            </div>
        </div>
    </div>
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
        
        # 진짜 URL 추적 시도
        final_link = get_real_url(session, link)
        
        rows.append({
            "키워드": keyword,
            "제목": title,
            "원문링크": final_link,
            "출처": extract_domain(final_link) or extract_domain(link),
            "발행일_UTC": pub_ts_utc,
            "수집시각_UTC": collected_at_utc,
            "_정규화링크": normalize_url(final_link),
            "요약": ""
        })
    return rows

# ============== 메인 ==============
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    # 타겟 날짜: "어제"
    now_kst = pd.Timestamp.now(tz="Asia/Seoul")
    yesterday_kst = now_kst - pd.Timedelta(days=1)
    target_date_str = yesterday_kst.strftime("%Y-%m-%d")
    print(f"🎯 타겟 날짜(어제): {target_date_str}")

    all_path = DATA_DIR / "ALL.csv"
    req_cols = ["키워드","제목","원문링크","발행일(KST)","수집시각(KST)","출처","요약",
                "_정규화링크","_발행일_dt","_수집시각_dt","_is_new"]
    
    if all_path.exists():
        df_existing = pd.read_csv(all_path, dtype=str, encoding="utf-8-sig")
        for c in req_cols: 
            if c not in df_existing.columns: df_existing[c] = ""
        df_existing["_수집시각_dt"] = pd.to_datetime(df_existing["_수집시각_dt"], errors="coerce")
        existing_links = set(df_existing["_정규화링크"].dropna().astype(str))
    else:
        df_existing = pd.DataFrame(columns=req_cols)
        existing_links = set()

    # 크롤링
    raw_rows = []
    for kw in KEYWORDS:
        print(f"📡 수집 중: {kw}...")
        raw_rows.extend(crawl_google_news_rss(session, kw))
        time.sleep(1)
    
    if not raw_rows: 
        print("수집된 데이터가 없습니다.")
        return

    df_crawled = pd.DataFrame(raw_rows)
    
    # 어제 날짜 필터링
    df_crawled["발행일(KST)"] = df_crawled["발행일_UTC"].apply(utc_to_kst_str)
    df_crawled = df_crawled[df_crawled["발행일(KST)"].str.startswith(target_date_str)]
    
    if df_crawled.empty:
        print(f"📅 {target_date_str} 날짜에 해당하는 기사가 없습니다.")
        return

    # 중복 제거
    df_crawled["_is_new"] = ~df_crawled["_정규화링크"].astype(str).isin(existing_links)
    df_crawled = df_crawled.drop_duplicates(subset=["_정규화링크"], keep="first")
    
    df_to_process = df_crawled[df_crawled["_is_new"] == True].copy()
    print(f"🔎 {target_date_str} 기사 중 신규 {len(df_to_process)}건 발견.")

    # 요약
    processed_rows = []
    for idx, row in df_to_process.iterrows():
        print(f"   Processing: {row['제목'][:20]}...")
        real_url = row["원문링크"]
        
        # 요약 시도
        content = extract_article_content(real_url)
        summary = ""
        if content:
            summary = summarize_with_gemini(content)
            time.sleep(2) 
            
        row["요약"] = summary
        processed_rows.append(row)

    if processed_rows:
        df_new_processed = pd.DataFrame(processed_rows)
        # 메일 발송
        df_new_processed["수집시각(KST)"] = df_new_processed["수집시각_UTC"].apply(utc_to_kst_str)
        df_new_processed["_발행일_dt"] = pd.to_datetime(df_new_processed["발행일(KST)"], errors="coerce")
        df_new_processed["_수집시각_dt"] = pd.to_datetime(df_new_processed["수집시각(KST)"], errors="coerce")
        send_email_report(df_new_processed, target_date_str)
    else:
        df_new_processed = pd.DataFrame(columns=df_crawled.columns)

    # 저장
    df_final_new = df_new_processed[req_cols] if not df_new_processed.empty else pd.DataFrame(columns=req_cols)
    combined = pd.concat([df_existing, df_final_new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["_정규화링크"], keep="last")
    combined["_수집시각_dt"] = pd.to_datetime(combined["_수집시각_dt"], errors="coerce")
    combined = combined.sort_values("_수집시각_dt", ascending=False)

    display_cols = ["키워드","제목","요약","원문링크","발행일(KST)","수집시각(KST)","출처"]
    combined[display_cols].to_csv(DATA_DIR / "ALL.csv", index=False, encoding="utf-8-sig")
    
    if not df_new_processed.empty:
        df_new_processed[display_cols].to_csv(DATA_DIR / "NEW_latest.csv", index=False, encoding="utf-8-sig")
    
    print("🎉 완료")

if __name__ == "__main__":
    main()
