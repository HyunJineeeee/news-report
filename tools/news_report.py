# -*- coding: utf-8 -*-
"""
news_report.py
--------------------------------
data/news_master.csv 기반 자동 PDF 리포트 생성
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

# ========== 경로 ==========
DATA_PATH = Path("data/news_master.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

today_str = datetime.now().strftime("%Y-%m-%d")
PDF_PATH = OUTPUT_DIR / f"news_report_{today_str}.pdf"

# ========== 데이터 로드 ==========
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["발행일(KST)"] = pd.to_datetime(df["발행일(KST)"], errors="coerce")

# 기본 통계
total_count = len(df)
by_keyword = df["키워드"].value_counts()
by_press = df["출처"].value_counts().head(5)

# ========== 일별 기사 추이 ==========
plt.figure(figsize=(6,3))
daily = df.groupby(df["발행일(KST)"].dt.date).size()
daily.plot(kind="line", marker="o")
plt.title("일별 기사 건수 추이")
plt.xlabel("날짜")
plt.ylabel("기사 수")
plt.grid(True)
chart_path = OUTPUT_DIR / "daily_trend.png"
plt.tight_layout()
plt.savefig(chart_path, dpi=150)
plt.close()

# ========== 주요 기사 10건 ==========
top_articles = df.sort_values("발행일(KST)", ascending=False).head(10)[
    ["키워드", "제목", "원문링크", "출처"]
]

# ========== PDF 생성 ==========
styles = getSampleStyleSheet()
doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4)
story = []

story.append(Paragraph(f"<b>일학습병행 관련 주요 뉴스 리포트 ({today_str})</b>", styles["Title"]))
story.append(Spacer(1, 12))
story.append(Paragraph(f"총 기사 수: <b>{total_count}</b>", styles["Normal"]))
story.append(Spacer(1, 6))

# 키워드별 기사 수
story.append(Paragraph("<b>키워드별 기사 수</b>", styles["Heading2"]))
data_kw = [["키워드", "건수"]] + [[k, v] for k, v in by_keyword.items()]
t_kw = Table(data_kw, hAlign="LEFT")
t_kw.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                          ("BACKGROUND", (0,0), (-1,0), colors.lightgrey)]))
story.append(t_kw)
story.append(Spacer(1, 12))

# 언론사 TOP5
story.append(Paragraph("<b>언론사별 보도 TOP5</b>", styles["Heading2"]))
data_press = [["언론사", "건수"]] + [[k, v] for k, v in by_press.items()]
t_press = Table(data_press, hAlign="LEFT")
t_press.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                             ("BACKGROUND", (0,0), (-1,0), colors.lightgrey)]))
story.append(t_press)
story.append(Spacer(1, 12))

# 차트 이미지 삽입
story.append(Paragraph("<b>일별 기사 추이</b>", styles["Heading2"]))
story.append(Image(str(chart_path), width=400, height=200))
story.append(Spacer(1, 12))

# 주요 기사 테이블
story.append(Paragraph("<b>최근 주요 기사 10건</b>", styles["Heading2"]))
articles_data = [["키워드", "제목", "언론사", "원문링크"]] + [
    [row["키워드"], row["제목"], row["출처"], row["원문링크"]] for _, row in top_articles.iterrows()
]
t_articles = Table(articles_data, colWidths=[60, 230, 80, 120])
t_articles.setStyle(TableStyle([
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
]))
story.append(t_articles)
story.append(Spacer(1, 12))

story.append(Paragraph(f"<i>생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", styles["Normal"]))
story.append(Paragraph("<i>자동 생성 보고서 (news_pipeline.py 기반)</i>", styles["Normal"]))

doc.build(story)
print(f"✅ PDF 생성 완료: {PDF_PATH}")
