# tools/make_wordcloud.py
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

DATA = Path("data/news_master.csv")
OUTDIR = Path("output")
OUTDIR.mkdir(exist_ok=True)
OUTPNG = OUTDIR / "wordcloud_titles.png"

# 1) 데이터 읽기
df = pd.read_csv(DATA, encoding="utf-8-sig")

# 2) 제목에서 한글 단어만 추출(2글자 이상), 불용어 제거
titles = df["제목"].dropna().astype(str)

# 불용어(원하는 대로 추가/수정 가능)
STOPWORDS = {
    "일학습병행","직업훈련","고용노동부","한국산업인력공단",  # 메인 키워드(크게 뜨는 거 방지)
    "기자","사진","영상","인터뷰","속보","종합",
    "기사","관련","정부","사업","지원","추진","발표",
    "오늘","내일","올해","지난","현장","한국","대한민국",
    "네이버","다음","naver","daum","com","co","kr",
}

def extract_korean_words(text: str) -> list[str]:
    # 한글 2글자 이상 단어만 추출
    words = re.findall(r"[가-힣]{2,}", text)
    # 불용어 제거
    return [w for w in words if w not in STOPWORDS]

tokens = []
for t in titles:
    tokens.extend(extract_korean_words(t))

if not tokens:
    print("⚠️ 추출된 단어가 없습니다. data/news_master.csv를 확인하세요.")
    raise SystemExit(0)

# 3) 워드클라우드 생성(폰트 자동탐색: 윈도우/맥/리눅스 순서)
font_candidates = [
    "C:/Windows/Fonts/malgun.ttf",                 # Windows: 맑은 고딕
    "/System/Library/Fonts/AppleGothic.ttf",       # macOS
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux
]
font_path = None
for fp in font_candidates:
    if Path(fp).exists():
        font_path = fp
        break

wc = WordCloud(
    font_path=font_path,
    background_color="white",
    width=1200,
    height=700
).generate(" ".join(tokens))

# 4) 저장
plt.figure(figsize=(10, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUTPNG, dpi=150)
print(f"✅ 저장 완료: {OUTPNG}")
