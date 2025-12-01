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
import difflib # ★ 텍스트 유사도 비교 라이브러리

# ============== 설정 ==============
KEYWORDS = ["일학습병행", "직업훈련", "고용노동부", "한국산업인력공단"]
DATA_DIR = Path("data")

# ★ 중복 제거 기준 (0.5 = 50% 이상 비슷하면 중복 처리)
SIMILARITY_THRESHOLD = 0.5 

# 키워드별 색상
KEYWORD_COLORS = {
    "일학습병행": "#3498db",      # 파랑
    "직업훈련": "#e67e22",        # 주황
    "고용노동부": "#7f8c8d",      # 회색
    "한국산업인력공단": "#2c3e50" # 남색
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
    """특수문자 제거 후 비교용 문자열 생성"""
    return re.sub(r'[^가-힣a-zA-Z0-9]', '', title)

def is_similar(text1, text2):
    """두 텍스트의 유사도가 설정값(50%) 이상인지 확인"""
    if not text1 or not text2: return False
    # SequenceMatcher를 이용해 유사도(0.0 ~ 1.0) 계산
    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
    return similarity >= SIMILARITY_THRESHOLD

# ============== AI & 본문 추출 ==============
def extract_article_content(url: str) -> str:
    if not url: return ""
    try:
        # 1. Trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if text and len(text) >= 100: return text
        
        # 2. Requests (Fallback)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            text = trafilatura.extract(resp.text, include_comments=False)
            if text and len(text) >= 100: return text
        return ""
    except: return ""

def summarize_with_gemini(text: str) -> str:
    if not GEMINI_API_KEY or not text: return ""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            "너는 뉴스 요약 전문가야. 아래 텍스트를 읽고 핵심 내용을 2~3줄로 요약해줘.\n"
            "형식: '- '로 시작하는 문장.\n"
            "어조: 건조하고 객관적인 보고서체.\n\n"
            f"텍스트:\n{text[:4000]}"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
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
    
    # 넉넉하게 100개 가져와서 필터링
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
            
        final_link = item['originallink'] if item['originallink'] else item['link']
        if not final_link: continue

        title = clean_html(item['title'])
        desc = clean_html(item['description'])
        
        rows.append({
            "키워드": keyword,
            "제목": title,
            "원문링크": final_link,
            "출처": "NaverAPI",
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
                border_color = kw_color if "- " in summary else "#ddd"
                
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
        # 기존 DB 제목들 (중복 방지용)
        existing_titles = list(df_existing["_title_norm"].dropna().astype(str))
    else:
        df_existing = pd.DataFrame(columns=req_cols)
        existing_titles = []

    # 크롤링
    raw_rows = []
    for kw in KEYWORDS:
        print(f"📡 수집 중 (Naver): {kw}...")
        raw_rows.extend(crawl_naver_news(kw, target_date_str))
        time.sleep(0.5)
    
    if not raw_rows: 
        print(f"📅 {target_date_str} 날짜에 해당하는 기사가 없습니다.")
        return

    # ★★★ 강력한 중복 제거 (유사도 50% 기준) ★★★
    unique_rows = []
    print("🧹 중복 제거(유사도 50%) 수행 중...")
    
    for row in raw_rows:
        new_title_norm = row["_title_norm"]
        is_duplicate = False
        
        # 1. 기존 DB와 비교
        for exist_title in existing_titles:
            if is_similar(new_title_norm, exist_title):
                is_duplicate = True
                break
        
        if is_duplicate: continue

        # 2. 이번 수집 목록 내 비교 (방금 추가한 것들과 비교)
        for accepted in unique_rows:
            if is_similar(new_title_norm, accepted["_title_norm"]):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_rows.append(row)
            # (주의: 기존 DB 리스트에는 추가하지 않음 - 이번 루프 성능 위해)

    df_to_process = pd.DataFrame(unique_rows)
    print(f"🔎 {len(raw_rows)}건 중 중복 제거 후 {len(df_to_process)}건 처리 시작.")

    processed_rows = []
    for idx, row in df_to_process.iterrows():
        print(f"   Processing: {row['제목'][:20]}...")
        real_url = row["원문링크"]
        keyword = row["키워드"]
        api_desc = row["_api_desc"]
        
        content = extract_article_content(real_url)
        summary = ""
        
        if content:
            # 본문 필터링
            if keyword not in content and keyword not in row['제목']:
                print(f"   ❌ [제외] 본문에 '{keyword}' 없음")
                continue 

            summary = summarize_with_gemini(content)
            time.sleep(2)
        
        if not summary or "부족합니다" in summary:
            if api_desc:
                summary = f"- (본문 접속 불가로 요약 대체) {api_desc}..."
            else:
                summary = "- 요약할 내용을 가져올 수 없습니다. 원문을 확인해주세요."
            
        row["요약"] = summary
        processed_rows.append(row)

    if processed_rows:
        df_new_processed = pd.DataFrame(processed_rows)
        send_email_report(df_new_processed, target_date_str)
    else:
        print("🧹 처리할 신규 기사가 없습니다.")
        df_new_processed = pd.DataFrame(columns=req_cols)

    # 저장
    df_final_new = df_new_processed[req_cols] if not df_new_processed.empty else pd.DataFrame(columns=req_cols)
    combined = pd.concat([df_existing, df_final_new], ignore_index=True)
    # 저장할 때는 혹시 모를 완전 동일 중복 한 번 더 제거
    combined = combined.drop_duplicates(subset=["_title_norm"], keep="last")
    combined = combined.sort_values("수집시각(KST)", ascending=False)

    display_cols = ["키워드","제목","요약","원문링크","발행일(KST)","수집시각(KST)"]
    combined[display_cols].to_csv(DATA_DIR / "ALL.csv", index=False, encoding="utf-8-sig")
    
    if not df_new_processed.empty:
        df_new_processed[display_cols].to_csv(DATA_DIR / "NEW_latest.csv", index=False, encoding="utf-8-sig")
    
    print("🎉 완료")

if __name__ == "__main__":
    main()
