# -*- coding: utf-8 -*-
"""
news_report_monthly.py
--------------------------------
data/news_master.csv 기반 월간 PDF 리포트 생성
usage:
  python tools/news_report_monthly.py                # 이번 달(KST)
  python tools/news_report_monthly.py --month 2025-10
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

DATA_PATH = Path("data/news_master.csv")
OUT_DIR = Path("output/monthly")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Matplotlib 한글 폰트/테마 (가능하면 Nanum/Noto, 없으면 기본) ----------
def setup_matplotlib():
    try:
        matplotlib.rcParams["font.family"] = "NanumGothic"
    except Exception:
        pass
    matplotlib.rcParams["axes.unicode_minus"] = False
    plt.style.use("ggplot")
    matplotlib.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

def month_bounds_kst(month_str: str | None):
    # KST 기준 월 시작/종료
    now_kst = pd.Timestamp.now(tz="Asia/Seoul")
    if month_str:
        start = pd.Timestamp(f"{month_str}-01", tz="Asia/Seoul")
    else:
        start = now_kst.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # 다음달 1일 - 1초
    if start.month == 12:
        next_start = start.replace(year=start.year + 1, month=1)
    else:
        next_start = start.replace(month=start.month + 1)
    end = next_start - pd.Timedelta(seconds=1)
    return start, end

def build_month_label(start: pd.Timestamp):
    return f"{start.year:04d}-{start.month:02d}"

def save_line_chart(daily_counts: pd.Series, title: str, path: Path):
    if daily_counts.empty:
        # 빈 그래프라도 만들어 PDF 삽입 가능하게
        fig, ax = plt.subplots(figsize=(7,3))
        ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(daily_counts.index, daily_counts.values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("날짜")
    ax.set_ylabel("기사 수")
    ax.grid(True, axis="y")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", help="리포트 대상 월 (YYYY-MM). 미지정 시 이번 달(KST)")
    args = parser.parse_args()

    setup_matplotlib()
    start_kst, end_kst = month_bounds_kst(args.month)
    month_label = build_month_label(start_kst)
    pdf_path = OUT_DIR / f"news_report_M_{month_label}.pdf"

    # ----- 데이터 로드 -----
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    if df.empty:
        # 빈 PDF라도 생성
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(f"<b>월간 뉴스 리포트 ({month_label})</b>", styles["Title"]),
            Spacer(1, 12),
            Paragraph("데이터가 없습니다.", styles["Normal"]),
        ]
        doc.build(story)
        print(f"✅ PDF 생성: {pdf_path} (빈 데이터)")
        return

    # 날짜 파싱(KST 문자열 → datetime)
    df["발행일(KST)"] = pd.to_datetime(df["발행일(KST)"], errors="coerce")
    # 대상 월 필터
    mask = (df["발행일(KST)"] >= start_kst.tz_convert(None)) & (df["발행일(KST)"] <= end_kst.tz_convert(None))
    month_df = df.loc[mask].copy()

    # ----- 통계 산출 -----
    total_count = len(month_df)
    by_keyword = month_df["키워드"].value_counts()
    by_press = month_df["출처"].value_counts().head(10)

    # 일별 추이(월 범위 내)
    if not month_df.empty:
        daily = month_df.groupby(month_df["발행일(KST)"].dt.date).size()
        # 월 달력 전체에 대해 0도 채워 넣기
        all_days = pd.date_range(start=start_kst.tz_convert(None).date(),
                                 end=end_kst.tz_convert(None).date(), freq="D")
        daily = daily.reindex(all_days.date, fill_value=0)
    else:
        daily = pd.Series(dtype=int)

    chart_path = OUT_DIR / f"daily_trend_{month_label}.png"
    save_line_chart(daily, f"{month_label} 일별 기사 건수 추이", chart_path)

    # 주요 기사 20건(최신순)
    top_articles = month_df.sort_values("발행일(KST)", ascending=False).head(20)[
        ["키워드", "제목", "원문링크", "출처", "발행일(KST)"]
    ]

    # ----- PDF 생성 -----
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12))
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    story = []

    # 표지/개요
    story.append(Paragraph(f"<b>일학습병행 관련 월간 뉴스 리포트 ({month_label})</b>", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"기간: {start_kst.strftime('%Y-%m-%d')} ~ {end_kst.strftime('%Y-%m-%d')}", styles["Normal"]))
    story.append(Paragraph(f"총 기사 수: <b>{total_count}</b>", styles["Normal"]))
    story.append(Paragraph(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Small"]))
    story.append(Spacer(1, 12))

    # 키워드별 기사 수
    story.append(Paragraph("<b>키워드별 기사 수</b>", styles["Heading2"]))
    data_kw = [["키워드", "건수"]] + [[k, int(v)] for k, v in by_keyword.items()]
    if len(data_kw) == 1:
        data_kw.append(["-", 0])
    t_kw = Table(data_kw, hAlign="LEFT", colWidths=[120, 60])
    t_kw.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (1,1), (1,-1), "RIGHT"),
    ]))
    story.append(t_kw)
    story.append(Spacer(1, 10))

    # 언론사 TOP10
    story.append(Paragraph("<b>언론사별 보도 TOP10</b>", styles["Heading2"]))
    data_press = [["언론사", "건수"]] + [[k, int(v)] for k, v in by_press.items()]
    if len(data_press) == 1:
        data_press.append(["-", 0])
    t_press = Table(data_press, hAlign="LEFT", colWidths=[200, 60])
    t_press.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (1,1), (1,-1), "RIGHT"),
    ]))
    story.append(t_press)
    story.append(Spacer(1, 10))

    # 일별 추이 차트
    story.append(Paragraph("<b>일별 기사 추이</b>", styles["Heading2"]))
    story.append(Image(str(chart_path), width=440, height=210))
    story.append(PageBreak())

    # 주요 기사 20건
    story.append(Paragraph("<b>월간 주요 기사 20건</b>", styles["Heading2"]))
    articles_data = [["발행일", "키워드", "언론사", "제목", "원문링크"]]
    for _, row in top_articles.iterrows():
        date_str = pd.to_datetime(row["발행일(KST)"]).strftime("%Y-%m-%d %H:%M") if pd.notna(row["발행일(KST)"]) else ""
        articles_data.append([date_str, row["키워드"], row["출처"], row["제목"], row["원문링크"]])

    t_articles = Table(articles_data, colWidths=[80, 70, 80, 240, 120])
    t_articles.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(t_articles)

    doc.build(story)
    print(f"✅ PDF 생성 완료: {pdf_path}")

if __name__ == "__main__":
    main()
