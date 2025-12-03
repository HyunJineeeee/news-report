# web_news.py
import requests
import pandas as pd
import os
import smtplib
import time
import re
import json
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import trafilatura
import difflib
import urllib3

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============== 설정 ==============
KEYWORDS = ["일학습병행", "직업훈련", "고용노동부", "한국산업인력공단"]
DATA_DIR = Path("data")

# ★ 중복 제거 기준: 10% (0.1) 이상 비슷하면 제거
SIMILARITY_THRESHOLD = 0.1 

KEYWORD_COLORS = {
    "일학습병행": "#3498db", "직업훈련": "#e67e22",
    "고용노동부": "#7f8c8d", "한국산업인력공단": "#2c3e50"
}

# 환경변수 로드
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")
NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")

# ============== 유틸 ==============
def clean_html(raw_html):
    if not raw_html: return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    # 특수문자 및 지저분한 기호 정리
    cleantext = cleantext.replace("&quot;", "'").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return cleantext

def normalize_for_comparison(title):
    """
    중복 비교를 위해 제목을 정규화하는 함수
    1. 지정된 키워드 제거 (일학습병행, 직업훈련 등)
    2. 특수문자/공백 제거
    """
    # 1. 키워드 제거
    for kw in KEYWORDS:
        title = title.replace(kw, "")
    
    # 2. 한글/영어/숫자만 남기고 다 제거
    return re.sub(r'[^가-힣a-zA-Z0-9]', '', title)

def is_similar(text1, text2):
    """
    두 텍스트(키워드 제거됨)의 유사도가 10% 이상인지 확인
    """
    if not text1 or not text2: return False
    
    # 유사도 계산 (0.0 ~ 1.0)
    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
    
    # 10% 이상이면 중복으로 간주 (True 반환)
    return similarity >= SIMILARITY_THRESHOLD

# ============== AI 기능 (REST API + 실패 시 조용히 처리) ==============
def call_gemini_silent(prompt):
    if not GEMINI_API_KEY: return None
    
    # 시도할 모델 리스트 (v1beta)
    models = ["gemini-1.5-flash", "gemini-pro"]
    
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    }

    for model_name in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        except:
            continue
            
    return None # 모든 시도 실패 시 None 반환 (에러 출력 X)

def summarize_article(text: str) -> str:
    prompt = (
        "뉴스 요약 봇. 다음 내용을 2줄 이내로 핵심만 요약.\n"
        "형식: '- '로 시작.\n"
        f"내용:\n{text[:3000]}"
    )
    return call_gemini_silent(prompt)

def repair_snippet(snippet: str) -> str:
    prompt = (
        "문장 완성 봇. 아래 문장은 잘려있다. 내용을 추측하여 자연스러운 한 문장으로 완성하라.\n"
        "형식: '- '로 시작.\n"
        f"입력:\n{snippet}"
    )
    return call_gemini_silent(prompt)

# ============== 본문 추출 (네이버 전용) ==============
def extract_article_content(url: str) -> str:
    if not url: return ""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://news.naver.com/'
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
            if text and len(text) >= 50: return text
        return ""
    except: return ""

# ============== 네이버 뉴스 검색 API ==============
def crawl_naver_news(keyword, target_date_str):
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        print("[ERROR] 네이버 API 키 누락")
        return []

    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    params = {"query": keyword, "display": 100, "start": 1, "sort": "date"}

    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
    except:
        return []

    rows = []
    collected_at = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M")
    
    for item in data.get('items', []):
        try:
            pub_date_dt = datetime.strptime(item['pubDate'], "%a, %d %b %Y %H:%M:%S +0900")
            pub_date_str = pub_date_dt.strftime("%Y-%m-%d %H:%M")
            pub_date_day = pub_date_dt.strftime("%Y-%m-%d")
        except: continue

        if pub_date_day != target_date_str: continue
            
        raw_link = item['link']
        if "news.naver.com" not in raw_link: continue 

        title = clean_html(item['title'])
        desc = clean_html(item['description'])
        
        # 중복 비교용 제목 생성 (키워드 제거됨)
        norm_title = normalize_for_comparison(title)
        
        rows.append({
            "키워드": keyword,
            "제목": title,
            "원문링크": raw_link,
            "출처": "NaverNews",
            "발행일(KST)": pub_date_str,
            "수집시각(KST)": collected_at,
            "요약": "",
            "_api_desc": desc,
            "_title_norm": norm_title
        })
    return rows

# ============== 이메일 발송 ==============
def send_email_report(df_new, target_date_str):
    if not EMAIL_USER or not EMAIL_PASSWORD or not EMAIL_RECEIVER: return
    if df_new.empty: return

    subject = f"[일병리포트] {target_date_str} 주요 뉴스 알림"

    html_body = f"""
    <div style="font-family: 'Malgun Gothic', sans-serif; background-color: #f4f4f4; padding: 20px; color: #333;">
        <div style="max-width: 700px; margin: 0 auto; background-color: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <div style="text-align: center; margin-bottom: 30px; border-bottom: 2px solid #555; padding-bottom: 20px;">
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
            kw_color = KEYWORD_COLORS.get(kw, "#333333")
            
            html_body += f"""
            <div style="margin-bottom: 30px;">
                <div style="background-color: {kw_color}; color: white; padding: 6px 15px; display: inline-block; border-radius: 15px; font-weight: bold; font-size: 16px; margin-bottom: 15px;">
                    # {kw}
                </div>
            """
            for idx, row in group_df.iterrows():
                title = row['제목']
                link = row['원문링크']
                date = row['발행일(KST)']
                summary = row['요약']
                
                # 요약 성공 시 줄바꿈
                summary_html = summary.replace('\n', '<br>')
                
                # 테두리 색상: AI 실패/성공 상관없이 키워드 색상 유지 (깔끔하게 보이기 위함)
                border_color = kw_color 
                
                html_body += f"""
                <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; margin-bottom: 15px; background-color: #fff;">
                    <a href="{link}" target="_blank" style="font-size: 18px; font-weight: bold; color: #2c3e50; text-decoration: none; display: block; margin-bottom: 8px; line-height: 1.4;">
                        {title}
                    </a>
                    <div style="font-size: 12px; color: #95a5a6; margin-bottom: 15px;">
                        {date}
                    </div>
                    <div style="background-color: #f9f9f9; padding: 15px; border-left: 4px solid {border_color}; color: #555; font-size: 14px; line-height: 1.6; border-radius: 4px;">
                        {summary_html}
                    </div>
                    <div style="text-align: right; margin-top: 10px;">
                        <a href="{link}" target="_blank" style="display: inline-block; background-color: #ecf0f1; color: #555; padding: 5px 12px; border-radius: 4px; text-decoration: none; font-size: 12px;">
                            원문 보러가기 →
                        </a>
                    </div>
                </div>
                """
            html_body += '</div>'

    html_body += """
            <div style="text-align: center; margin-top: 40px; font-size: 12px; color: #bdc3c7; border-top: 1px solid #eee; padding-top: 20px;">
                Automated by GitHub Actions & Naver API
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

# ============== 메인 ==============
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    now_kst = pd.Timestamp.now(tz="Asia/Seoul")
    yesterday_kst = now_kst - pd.Timedelta(days=1)
    target_date_str = yesterday_kst.strftime("%Y-%m-%d")
    print(f"🎯 타겟 날짜(어제): {target_date_str}")

    all_path = DATA_DIR / "ALL.csv"
    req_cols = ["키워드","제목","원문링크","발행일(KST)","수집시각(KST)","출처","요약","_api_desc","_title_norm"]
    
    if all_path.exists():
        df_existing = pd.read_csv(all_path, dtype=str, encoding="utf-8-sig")
        for c in req_cols: 
            if c not in df_existing.columns: df_existing[c] = ""
        existing_titles = list(df_existing["_title_norm"].dropna().astype(str))
    else:
        df_existing = pd.DataFrame(columns=req_cols)
        existing_titles = []

    raw_rows = []
    for kw in KEYWORDS:
        print(f"📡 수집 중 (Naver): {kw}...")
        raw_rows.extend(crawl_naver_news(kw, target_date_str))
        time.sleep(0.5)
    
    if not raw_rows: 
        print(f"📅 {target_date_str} 날짜에 해당하는 기사가 없습니다.")
        return

    # ★★★ 강력한 중복 제거 (키워드 제외 후 10% 유사도 체크) ★★★
    unique_rows = []
    print(f"🧹 중복 제거(유사도 {int(SIMILARITY_THRESHOLD*100)}%) 수행 중...")
    
    for row in raw_rows:
        new_title_norm = row["_title_norm"]
        is_duplicate = False
        
        # 1. 기존 DB와 비교
        for exist_title in existing_titles:
            if is_similar(new_title_norm, exist_title):
                is_duplicate = True
                break
        if is_duplicate: continue
        
        # 2. 이번 수집 내 비교
        for accepted in unique_rows:
            if is_similar(new_title_norm, accepted["_title_norm"]):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_rows.append(row)

    df_to_process = pd.DataFrame(unique_rows)
    print(f"🔎 {len(raw_rows)}건 중 중복 제거 후 {len(df_to_process)}건 처리 시작.")

    processed_rows = []
    for idx, row in df_to_process.iterrows():
        print(f"   Processing: {row['제목'][:20]}...")
        target_url = row["원문링크"]
        keyword = row["키워드"]
        api_desc = row["_api_desc"]
        
        content = extract_article_content(target_url)
        summary = ""
        
        if content:
            if keyword not in content and keyword not in row['제목']:
                print(f"   ❌ [제외] 본문에 '{keyword}' 없음")
                continue 
            summary = summarize_article(content)
            time.sleep(2)
        
        # ★ 최종 안전장치: AI가 실패하면 네이버 요약 원본을 그대로 보여줌 (에러 메시지 X)
        if not summary:
            # AI 복원 시도
            restored = repair_snippet(api_desc)
            if restored:
                summary = restored
            else:
                # AI가 완전 죽었으면 그냥 네이버 요약이라도 보여줌 (빈칸보다는 나음)
                summary = f"- {api_desc} (내용 확인 필요)"
            
        row["요약"] = summary
        processed_rows.append(row)

    if processed_rows:
        df_new_processed = pd.DataFrame(processed_rows)
        send_email_report(df_new_processed, target_date_str)
    else:
        print("🧹 처리할 신규 기사가 없습니다.")
        df_new_processed = pd.DataFrame(columns=req_cols)

    df_final_new = df_new_processed[req_cols] if not df_new_processed.empty else pd.DataFrame(columns=req_cols)
    combined = pd.concat([df_existing, df_final_new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["_title_norm"], keep="last")
    combined = combined.sort_values("수집시각(KST)", ascending=False)

    display_cols = ["키워드","제목","요약","원문링크","발행일(KST)","수집시각(KST)"]
    combined[display_cols].to_csv(DATA_DIR / "ALL.csv", index=False, encoding="utf-8-sig")
    
    if not df_new_processed.empty:
        df_new_processed[display_cols].to_csv(DATA_DIR / "NEW_latest.csv", index=False, encoding="utf-8-sig")
    
    print("🎉 완료")

if __name__ == "__main__":
    main()
