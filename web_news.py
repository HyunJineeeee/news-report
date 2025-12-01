# web_news.py
import requests
import pandas as pd
import os
import smtplib
import time
import re
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import google.generativeai as genai
from datetime import datetime, timedelta
import trafilatura
import difflib
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============== 설정 ==============
KEYWORDS = ["일학습병행", "직업훈련", "고용노동부", "한국산업인력공단"]
DATA_DIR = Path("data")
SIMILARITY_THRESHOLD = 0.4

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

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ============== 유틸 ==============
def clean_html(raw_html):
    if not raw_html: return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.replace("&quot;", "'").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")

def normalize_title(title):
    return re.sub(r'[^가-힣a-zA-Z0-9]', '', title)

def is_similar(text1, text2):
    if not text1 or not text2: return False
    return difflib.SequenceMatcher(None, text1, text2).ratio() >= SIMILARITY_THRESHOLD

# ============== AI 기능 (핵심 수정: 안전 필터 해제) ==============
def generate_content_safe(prompt):
    if not GEMINI_API_KEY: return ""
    
    # ★ 안전 설정: 뉴스 내용이 차단되지 않도록 필터링 수준을 낮춤
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    # ★ 모델 변경: 가장 안정적인 'gemini-pro' 사용
    try:
        model = genai.GenerativeModel('gemini-pro') 
        response = model.generate_content(prompt, safety_settings=safety_settings)
        
        # 응답이 정상적인지 확인
        if response.text:
            return response.text.strip()
        else:
            print("⚠️ [AI 응답 없음] 빈 결과 반환")
            return ""
            
    except Exception as e:
        # 에러 발생 시 로그에 상세 출력
        print(f"❌ [AI 생성 실패] 원인: {e}")
        return ""

def summarize_article(text: str) -> str:
    prompt = (
        "너는 뉴스 리포트 봇이야. 아래 기사 본문을 읽고 핵심 내용을 2~3줄로 요약해.\n"
        "형식: '- '로 시작하는 개조식 문장.\n"
        "조건: 감정을 배제하고 건조한 보고서체 사용.\n"
        "주의: 서론 없이 바로 요약 내용만 출력.\n\n"
        f"기사 본문:\n{text[:3500]}" # 토큰 제한 고려하여 길이 조정
    )
    result = generate_content_safe(prompt)
    if result: return result
    return f"- (AI 요약 실패) 원문 확인 필요"

def repair_snippet(snippet: str) -> str:
    prompt = (
        "너는 문장 교정 전문가야. 아래 텍스트는 기사 요약의 일부인데 문장이 잘려 있어.\n"
        "내용을 추론하여 **완전한 하나의 요약 문장**으로 다듬어줘.\n"
        "형식: '- '로 시작.\n\n"
        f"입력 텍스트:\n{snippet}"
    )
    result = generate_content_safe(prompt)
    # AI가 성공했으면 그 결과 반환, 실패했으면 원본(네이버 요약)이라도 보여줌
    if result: return result
    return f"- {snippet}"

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
    except Exception as e:
        print(f"   [API Error] {e}")
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
        
        rows.append({
            "키워드": keyword,
            "제목": title,
            "원문링크": raw_link,
            "출처": "NaverNews",
            "발행일(KST)": pub_date_str,
            "수집시각(KST)": collected_at,
            "요약": "",
            "_api_desc": desc,
            "_title_norm": normalize_title(title)
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
                summary_html = summary.replace('\n', '<br>')
                
                # 실패 문구가 보이면 회색, 아니면 키워드 색상
                border_color = kw_color if summary and "실패" not in summary else "#ddd"
                
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

    unique_rows = []
    for row in raw_rows:
        new_title_norm = row["_title_norm"]
        is_duplicate = False
        for exist_title in existing_titles:
            if is_similar(new_title_norm, exist_title):
                is_duplicate = True
                break
        if is_duplicate: continue
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
        
        if not summary or "부족합니다" in summary:
            # 실패 시 복원 (안전 설정 해제됨)
            restored = repair_snippet(api_desc)
            if restored == api_desc: 
                summary = f"{api_desc} (AI 작동 실패)"
            else:
                summary = restored
            
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
