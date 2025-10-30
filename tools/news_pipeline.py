# -*- coding: utf-8 -*-
"""
news_pipeline.py
- append-only 원본 누적(raw)
- 제목 유사도 기반 중복 제거 → 대표 기사 1건 선정(master)
- 제목 기반 Top 키워드 Bar 차트 (주/월/분기)
- ✅ 저장 시 컬럼 순서 고정:
    ["키워드", "제목", "원문링크", "발행일(KST)", "수집시각(KST)", "출처"]

사용 예:
    python tools/news_pipeline.py \
        --new-batch "data/*.csv" \
        --raw "data/news_raw.csv" \
        --master "data/news_master.csv" \
        --outdir "output" \
        --sim-threshold 0.85
"""

import argparse
import glob
import os
import re
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


# ---------- 고정 출력 컬럼 ----------
KST_TZ = "Asia/Seoul"
OUTPUT_ORDER = ["키워드", "제목", "원문링크", "발행일(KST)", "수집시각(KST)", "출처"]

# ---------- 입력 컬럼 유추 후보 ----------
COL_CANDIDATES = {
    "keyword": ["keyword", "키워드"],
    "title": ["title", "subject", "headline", "제목"],
    "url": ["url", "link", "기사링크", "news_url", "원문링크"],
    "pub_date": ["pub_date", "date", "published_at", "publish_date", "발행일", "작성일"],
    "press": ["press", "source", "publisher", "언론사", "매체", "출처"],
    "collected_at": ["collected_at", "staged_at", "crawled_at", "수집일", "수집일자", "수집시각"],
    "content": ["content", "summary", "본문", "요약"],  # 제목만 분석하므로 분석엔 미사용
}


# ---------- 유틸 ----------
def extract_domain(u: str) -> str:
    try:
        return urlparse(str(u)).netloc.lower()
    except Exception:
        return ""


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """레포 CSV의 다양한 헤더를 표준 컬럼으로 정규화."""
    cols_lower = {c.lower(): c for c in df.columns}

    def choose(cands):
        for cand in cands:
            if cand in df.columns:
                return cand
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        return None

    # 존재하지 않으면 생성
    for std, cands in COL_CANDIDATES.items():
        chosen = choose(cands)
        if chosen is None:
            df[std] = pd.Series([np.nan] * len(df))
        else:
            if chosen != std:
                df.rename(columns={chosen: std}, inplace=True)

    # 날짜 파싱
    for dcol in ["pub_date", "collected_at"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    # press 없으면 URL에서 도메인 추출
    if "press" in df.columns and "url" in df.columns:
        df["press"] = df["press"].fillna("")
        empty_press = df["press"].astype(str).str.strip().eq("")
        if empty_press.any():
            df.loc[empty_press, "press"] = df.loc[empty_press, "url"].apply(extract_domain)

    return df


def to_kst_ts(ts):
    if pd.isna(ts):
        return pd.NaT
    ts = pd.to_datetime(ts, errors="coerce")
    if ts is pd.NaT or ts is None:
        return pd.NaT
    # tz-naive면 KST로 간주, tz-aware면 KST로 변환
    try:
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            return ts.tz_localize(KST_TZ)
        else:
            return ts.tz_convert(KST_TZ)
    except Exception:
        return pd.NaT


def format_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    내부 표준 컬럼(keyword/title/url/pub_date/collected_at/press)을
    최종 출력 컬럼(한글명, KST 시각 문자열)로 변환 & 순서 고정.
    """
    out = pd.DataFrame(index=df.index.copy())

    # 키워드: 있으면 사용, 없으면 공백
    out["키워드"] = df.get("keyword", "").fillna("").astype(str)

    # 제목 / 원문링크 / 출처
    out["제목"] = df.get("title", "").astype(str)
    out["원문링크"] = df.get("url", "").astype(str)
    out["출처"] = df.get("press", "").astype(str)

    # 발행일/수집시각 → KST 문자열
    pub_kst = pd.to_datetime(df.get("pub_date", pd.NaT), errors="coerce").apply(to_kst_ts)
    col_kst = pd.to_datetime(df.get("collected_at", pd.NaT), errors="coerce").apply(to_kst_ts)

    out["발행일(KST)"] = pub_kst.dt.strftime("%Y-%m-%d %H:%M").fillna("")
    out["수집시각(KST)"] = col_kst.dt.strftime("%Y-%m-%d %H:%M").fillna("")

    # 순서 강제
    out = out.reindex(columns=OUTPUT_ORDER)
    return out


# ---------- 1) 원본 누적(append-only) ----------
def load_csvs(patterns):
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    frames = []
    for path in sorted(set(paths)):
        try:
            df = pd.read_csv(path)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] CSV 로드 실패: {path} - {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def anti_join_by_url_title(raw: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    # URL + 제목 키 조합으로 기존 존재 여부를 체크
    def keyify(s):
        return s.fillna("").astype(str)

    raw_keys = set(zip(keyify(raw.get("url", pd.Series(dtype=str))),
                       keyify(raw.get("title", pd.Series(dtype=str)))))

    mask = []
    for u, t in zip(keyify(new_df.get("url", pd.Series(dtype=str))),
                    keyify(new_df.get("title", pd.Series(dtype=str)))):
        mask.append((u, t) not in raw_keys)
    return new_df[pd.Series(mask, index=new_df.index)]


def append_only_raw(raw_path: str, new_batch_paths) -> pd.DataFrame:
    # 기존 raw 불러오기
    if os.path.exists(raw_path):
        raw = pd.read_csv(raw_path)
        # 기존 파일이 이미 한글 컬럼일 수 있으므로 다시 표준 컬럼으로 복원 시도
        raw_std = raw.rename(columns={
            "키워드": "keyword",
            "제목": "title",
            "원문링크": "url",
            "발행일(KST)": "pub_date",       # 문자열일 수 있음(아래에서 재가공)
            "수집시각(KST)": "collected_at",  # 문자열일 수 있음
            "출처": "press",
        })
        raw = map_columns(raw_std)
    else:
        raw = pd.DataFrame(columns=list(COL_CANDIDATES.keys()))

    # 신규 배치
    new_df = load_csvs(new_batch_paths)
    if new_df.empty:
        print("[INFO] 신규 배치 없음.")
        return raw

    new_df = map_columns(new_df)

    # collected_at 비어 있으면 지금 시각 부여
    if "collected_at" in new_df.columns:
        new_df["collected_at"] = new_df["collected_at"].fillna(pd.Timestamp.now())

    # 원본 중복 1차 필터링 후 append
    new_df = anti_join_by_url_title(raw, new_df)
    if new_df.empty:
        print("[INFO] 추가할 신규 레코드가 없음.")
        return raw

    appended = pd.concat([raw, new_df], ignore_index=True)
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
    for rank, dom in enumerate(reversed(PREFERRED_DOMAINS), start=1):
        if dom in host:
            return rank
    return 0


def build_title_clusters(df: pd.DataFrame, sim_threshold=0.85, n_neighbors=10):
    titles = df["title"].fillna("").astype(str)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    X = vec.fit_transform(titles)

    nn = NearestNeighbors(metric="cosine", n_neighbors=min(n_neighbors, X.shape[0])).fit(X)
    distances, indices = nn.kneighbors(X)

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
    if "pub_date" in subset.columns:
        subset["_date"] = pd.to_datetime(subset["pub_date"], errors="coerce")
    else:
        subset["_date"] = pd.to_datetime(subset["collected_at"], errors="coerce")

    subset["title_len"] = subset["title"].astype(str).str.len()

    # 정렬 규칙: dom_score desc → _date asc → title_len desc
    subset = subset.sort_values(by=["dom_score", "_date", "title_len"],
                                ascending=[False, True, False])
    return subset.index[0]


def dedupe_by_title_similarity(raw_df: pd.DataFrame,
                               sim_threshold=0.85) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

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
    # 업무 고유명사는 불용어 처리 (필요시 추가)
    "일학습병행", "직업훈련", "고용노동부", "한국산업인력공단",
    "기사", "사진", "영상", "보도", "속보", "단독", "종합", "인터뷰",
    "기자", "뉴스", "관련", "정부", "사업", "지원", "추진"
}


def normalize_title_korean(text: str) -> list[str]:
    """
    간단 토큰화:
      - 한글/영문/숫자만 남기고 공백 분리
      - 1글자 토큰 제거
      - 불용어 제거
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
    기준 날짜: pub_date → collected_at
    """
    if df.empty:
        return
    os.makedirs(outdir, exist_ok=True)

    # 기준 날짜
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
                    help="append-only 원본 CSV 경로(저장 시 컬럼 순서 고정)")
    ap.add_argument("--master", default="data/news_master.csv",
                    help="중복 제거/정제 결과 CSV 경로(저장 시 컬럼 순서 고정)")
    ap.add_argument("--outdir", default="output",
                    help="차트/결과물 출력 폴더")
    ap.add_argument("--sim-threshold", type=float, default=0.85,
                    help="제목 유사도 임계값 (기본 0.85)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.raw), exist_ok=True)
    os.makedirs(os.path.dirname(args.master), exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    # 1) 원본 append-only (아직은 내부 표준 컬럼)
    raw_std = append_only_raw(args.raw, args.new_batch)

    # 2) 제목 유사도 중복 제거 → 대표 1건 (내부 표준 컬럼)
    master_std = dedupe_by_title_similarity(raw_std, sim_threshold=args.sim_threshold)

    # 3) 저장 직전에 컬럼을 한글/순서 고정으로 변환
    raw_out = format_output_columns(raw_std)
    master_out = format_output_columns(master_std)

    # 4) CSV 저장 (UTF-8 with BOM, 엑셀 호환)
    raw_out.to_csv(args.raw, index=False, encoding="utf-8-sig")
    print(f"[OK] raw 저장(고정 컬럼): {args.raw} (rows={len(raw_out)})")

    master_out.to_csv(args.master, index=False, encoding="utf-8-sig")
    print(f"[OK] master 저장(고정 컬럼): {args.master} (rows={len(master_out)})")

    # 5) 제목기반 Top 키워드 bar 차트 (주/월/분기) - master 기준
    for period in ["W", "M", "Q"]:
        top_keywords_bar(master_std, args.outdir, period=period, top_k=30)


if __name__ == "__main__":
    main()
