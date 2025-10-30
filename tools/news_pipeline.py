# -*- coding: utf-8 -*-
"""
news_pipeline.py
- append-only 원본 누적(raw)
- 제목 유사도 기반 중복 제거 → 대표 기사 1건 선정(master)
- 제목 기반 Top 키워드 Bar 차트 (주/월/분기)

사용 예:
    python tools/news_pipeline.py \
        --new-batch "data/*.csv" \
        --raw "data/news_raw.csv" \
        --master "data/news_master.csv" \
        --outdir "output"

필요 패키지:
    pandas, numpy, scikit-learn, matplotlib, python-dateutil
"""

import argparse
import glob
import os
import re
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


# ---------- 유틸: 컬럼 맵핑/유추 ----------
COL_CANDIDATES = {
    "title": ["title", "subject", "headline", "제목"],
    "url": ["url", "link", "기사링크", "news_url"],
    "pub_date": ["pub_date", "date", "published_at", "publish_date", "발행일", "작성일"],
    "press": ["press", "source", "publisher", "언론사", "매체"],
    "collected_at": ["collected_at", "staged_at", "crawled_at", "수집일", "수집일자"],
    "content": ["content", "summary", "본문", "요약"],  # 현재는 키워드 분석에 미사용(제목만)
}


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """레포 CSV의 다양한 헤더를 표준 컬럼으로 정규화."""
    mapping = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for std, cands in COL_CANDIDATES.items():
        chosen = None
        for cand in cands:
            if cand in df.columns:
                chosen = cand
                break
            if cand.lower() in cols_lower:
                chosen = cols_lower[cand.lower()]
                break
        if chosen is None:
            # 없으면 생성 (NaN)
            df[std] = pd.Series([np.nan] * len(df))
        else:
            df.rename(columns={chosen: std}, inplace=True)
            mapping[std] = chosen

    # 날짜 파싱
    for dcol in ["pub_date", "collected_at"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    # press 없으면 URL에서 도메인 추출
    if "press" in df.columns and "url" in df.columns:
        df["press"] = df["press"].fillna("")
        null_press = df["press"].astype(str).str.strip().eq("")
        if null_press.any():
            df.loc[null_press, "press"] = df.loc[null_press, "url"].apply(extract_domain)

    # collected_at 없으면 지금 시각 부여(배치 적재 시각)
    if "collected_at" in df.columns and df["collected_at"].isna().all():
        df["collected_at"] = pd.Timestamp.now()

    return df


def extract_domain(u: str) -> str:
    try:
        netloc = urlparse(str(u)).netloc
        return netloc.lower()
    except Exception:
        return ""


# ---------- 1) 원본 누적(append-only) ----------
def load_csvs(patterns):
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    frames = []
    for path in sorted(set(paths)):
        try:
            df = pd.read_csv(path)
            df["__source_file"] = os.path.basename(path)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] CSV 로드 실패: {path} - {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def anti_join_by_url_title(raw: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    # URL + 제목 키 조합으로 기존 존재 여부를 체크 (느슨한 1차 필터)
    def keyify(s):
        return s.fillna("").astype(str)

    raw_keys = set(zip(keyify(raw.get("url", pd.Series(dtype=str))),
                       keyify(raw.get("title", pd.Series(dtype=str)))))

    mask = []
    for u, t in zip(keyify(new_df.get("url", pd.Series(dtype=str))),
                    keyify(new_df.get("title", pd.Series(dtype=str)))):
        mask.append((u, t) not in raw_keys)
    return new_df[pd.Series(mask)]


def append_only_raw(raw_path: str, new_batch_paths) -> pd.DataFrame:
    # 기존 raw 불러오기
    if os.path.exists(raw_path):
        raw = pd.read_csv(raw_path)
        raw = map_columns(raw)
    else:
        raw = pd.DataFrame(columns=list(COL_CANDIDATES.keys()))

    # 신규 배치
    new_df = load_csvs(new_batch_paths)
    if new_df.empty:
        print("[INFO] 신규 배치 없음.")
        return raw

    new_df = map_columns(new_df)

    # 부족한 날짜 보정
    if "collected_at" in new_df.columns:
        new_df["collected_at"] = new_df["collected_at"].fillna(pd.Timestamp.now())

    # 원본 중복 1차 필터링 후 append
    new_df = anti_join_by_url_title(raw, new_df)
    if new_df.empty:
        print("[INFO] 추가할 신규 레코드가 없음.")
        return raw

    appended = pd.concat([raw, new_df], ignore_index=True)
    appended.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"[OK] raw 갱신 완료: {raw_path} (추가 {len(new_df)} rows)")
    return appended


# ---------- 2) 제목 유사도 기반 중복 제거(클러스터링) ----------
PREFERRED_DOMAINS = [
    # 조직 기준에 맞게 조정 가능(상위일수록 가점)
    "yna.co.kr", "news1.kr", "newsis.com", "sedaily.com", "hankyung.com",
    "mk.co.kr", "heraldcorp.com", "chosun.com", "joongang.co.kr", "hani.co.kr",
]


def domain_score(u: str) -> int:
    host = extract_domain(u)
    if not host:
        return 0
    # 뒤에서부터 점수 부여(리스트 앞쪽이 높은 점수)
    for rank, dom in enumerate(reversed(PREFERRED_DOMAINS), start=1):
        if dom in host:
            return rank
    return 0


def build_title_clusters(df: pd.DataFrame, sim_threshold=0.85, n_neighbors=10):
    titles = df["title"].fillna("").astype(str)

    # char n-gram TF-IDF
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    X = vec.fit_transform(titles)

    # 최근접 이웃 (코사인 거리)
    nn = NearestNeighbors(metric="cosine", n_neighbors=min(n_neighbors, X.shape[0])).fit(X)
    distances, indices = nn.kneighbors(X)

    # Union-Find by threshold
    thr_dist = 1 - sim_threshold
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(df)):
        for d, j in zip(distances[i], indices[i]):
            if i == j:
                continue
            if d <= thr_dist:
                union(i, j)

    groups = {}
    for i in range(len(df)):
        r = find(i)
        groups.setdefault(r, []).append(i)

    return groups


def choose_representative(df: pd.DataFrame, idxs: list[int]) -> int:
    subset = df.iloc[idxs].copy()

    subset["dom_score"] = subset["url"].apply(domain_score)
    # 날짜: pub_date 우선, 없으면 collected_at
    date_col = "pub_date" if "pub_date" in subset.columns else "collected_at"
    if date_col in subset.columns:
        subset["_date"] = pd.to_datetime(subset[date_col], errors="coerce")
    else:
        subset["_date"] = pd.NaT

    subset["title_len"] = subset["title"].astype(str).str.len()

    # 정렬 규칙: dom_score desc → _date asc → title_len desc
    subset = subset.sort_values(by=["dom_score", "_date", "title_len"],
                                ascending=[False, True, False])
    return subset.index[0]


def dedupe_by_title_similarity(raw_df: pd.DataFrame,
                               sim_threshold=0.85) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    # 기본 키존재 보정
    for c in ["title", "url"]:
        if c not in raw_df.columns:
            raw_df[c] = ""

    groups = build_title_clusters(raw_df, sim_threshold=sim_threshold)

    rep_indices = [choose_representative(raw_df, rows) for rows in groups.values()]
    deduped = raw_df.loc[sorted(set(rep_indices))].copy()
    deduped.reset_index(drop=True, inplace=True)
    return deduped


# ---------- 3) 키워드 빈도(제목만) → Bar 차트 ----------
STOPWORDS = {
    # 업무용 고유명사는 불용어 처리 (필요시 추가)
    "일학습병행", "직업훈련", "고용노동부", "한국산업인력공단",
    "기사", "사진", "영상", "보도", "속보", "단독", "종합", "인터뷰",
    "기자", "뉴스", "관련", "정부", "사업", "지원", "추진"
}


def normalize_title_korean(text: str) -> list[str]:
    """
    간단한 토큰화:
      - 한글/영문/숫자만 남기고 공백 분리
      - 1글자 토큰 제거
      - 불용어 제거
    형태소 분석기 없이도 '제목'에서는 꽤 깔끔하게 동작
    """
    if not isinstance(text, str):
        return []
    cleaned = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    toks = [t for t in re.split(r"\s+", cleaned.strip()) if len(t) > 1]
    toks = [t for t in toks if t not in STOPWORDS]
    return toks


def top_keywords_bar(df: pd.DataFrame, outdir: str, period: str = "W", top_k=30):
    """
    period: 'W' (주), 'M' (월), 'Q' (분기)
    """
    if df.empty:
        return
    os.makedirs(outdir, exist_ok=True)

    # 기준 날짜: pub_date -> collected_at -> today
    base_date = None
    if "pub_date" in df.columns and not df["pub_date"].isna().all():
        base_date = pd.to_datetime(df["pub_date"], errors="coerce")
    elif "collected_at" in df.columns and not df["collected_at"].isna().all():
        base_date = pd.to_datetime(df["collected_at"], errors="coerce")
    else:
        base_date = pd.Series([pd.Timestamp.now()] * len(df))

    tmp = df.copy()
    tmp["__date"] = base_date
    tmp = tmp.dropna(subset=["__date"])
    tmp["__period"] = tmp["__date"].dt.to_period(period).dt.to_timestamp()

    # 기간별 집계
    for pval, chunk in tmp.groupby("__period"):
        tokens = []
        for t in chunk["title"].fillna("").astype(str):
            tokens.extend(normalize_title_korean(t))
        if not tokens:
            continue
        vc = pd.Series(tokens).value_counts().head(top_k)

        plt.figure(figsize=(10, 6))
        plt.barh(vc.index[::-1], vc.values[::-1])
        plt.title(f"Top Keywords ({period}) - {pval.date()}")
        plt.tight_layout()

        fname = f"top_keywords_{period}_{pval.date()}.png"
        save_path = os.path.join(outdir, fname)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[OK] 저장: {save_path}")


# ---------- 엔드투엔드 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-batch", nargs="+", default=["data/*.csv"],
                    help="신규 배치 CSV 경로(글롭 가능) 다건")
    ap.add_argument("--raw", default="data/news_raw.csv",
                    help="append-only 원본 CSV 경로")
    ap.add_argument("--master", default="data/news_master.csv",
                    help="중복 제거/정제 결과 CSV 경로")
    ap.add_argument("--outdir", default="output",
                    help="차트/결과물 출력 폴더")
    ap.add_argument("--sim-threshold", type=float, default=0.85,
                    help="제목 유사도 임계값 (기본 0.85)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.raw), exist_ok=True)
    os.makedirs(os.path.dirname(args.master), exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    # 1) 원본 append-only
    raw_df = append_only_raw(args.raw, args.new_batch)

    # 2) 제목 유사도 중복 제거 → 대표 1건
    master_df = dedupe_by_title_similarity(raw_df, sim_threshold=args.sim_threshold)
    master_df.to_csv(args.master, index=False, encoding="utf-8-sig")
    print(f"[OK] master 저장: {args.master} (rows={len(master_df)})")

    # 3) 제목기반 Top 키워드 bar 차트 (주/월/분기)
    for period in ["W", "M", "Q"]:
        top_keywords_bar(master_df, args.outdir, period=period, top_k=30)


if __name__ == "__main__":
    main()
