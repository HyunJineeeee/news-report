# -*- coding: utf-8 -*-
"""
news_pipeline.py (정리 슬림 버전 + 버그픽스)
---------------------------------------
- append-only 원본 누적(raw)
- 제목 유사도 기반 중복 제거 → 대표 기사 1건 선정(master)
- 저장 시 컬럼 순서 고정:
  ["키워드","제목","원문링크","발행일(KST)","수집시각(KST)","출처"]
- 포인트:
  * collected_at을 tz-aware(KST)로 기록
  * 문자열 컬럼은 fillna("") 후 astype(str) → "nan" 방지
  * 입력 CSV 헤더 표준화 + 중복 헤더 병합
  * data/*.csv 로딩 시 news_raw.csv/news_master.csv 제외
"""

import argparse
import glob
import os
from urllib.parse import urlparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------
# 설정 / 컬럼 정의
# ---------------------------------------------------------------------
KST_TZ = "Asia/Seoul"
OUTPUT_ORDER = ["키워드", "제목", "원문링크", "발행일(KST)", "수집시각(KST)", "출처"]

COL_CANDIDATES = {
    "keyword": ["keyword", "키워드"],
    "title": ["title", "subject", "headline", "제목"],
    "url": ["url", "link", "기사링크", "news_url", "원문링크"],
    "pub_date": [
        "pub_date", "date", "published_at", "publish_date",
        "발행일", "작성일", "발행일(KST)", "등록일", "게시일"
    ],
    "press": ["press", "source", "publisher", "언론사", "매체", "출처"],
    "collected_at": [
        "collected_at", "staged_at", "crawled_at",
        "수집일", "수집일자", "수집시각", "수집시각(KST)"
    ]
}



# ---------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------
def extract_domain(u: str) -> str:
    try:
        return urlparse(str(u)).netloc.lower()
    except Exception:
        return ""


def to_kst_ts(ts):
    """tz-aware/naive 모든 입력을 KST tz-aware로 변환"""
    if pd.isna(ts):
        return pd.NaT
    ts = pd.to_datetime(ts, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    try:
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            return ts.tz_localize(KST_TZ)
        else:
            return ts.tz_convert(KST_TZ)
    except Exception:
        return pd.NaT


def format_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """저장용 6개 컬럼으로 정리 (순서 고정, 공백 처리, KST 포맷)"""
    out = pd.DataFrame(index=df.index.copy())
    out["키워드"] = df.get("keyword", "").fillna("").astype(str)
    out["제목"] = df.get("title", "").fillna("").astype(str)
    out["원문링크"] = df.get("url", "").fillna("").astype(str)
    out["출처"] = df.get("press", "").fillna("").astype(str)

    pub_kst = pd.to_datetime(df.get("pub_date", pd.NaT), errors="coerce").apply(to_kst_ts)
    col_kst = pd.to_datetime(df.get("collected_at", pd.NaT), errors="coerce").apply(to_kst_ts)

    out["발행일(KST)"] = pub_kst.dt.strftime("%Y-%m-%d %H:%M").fillna("")
    out["수집시각(KST)"] = col_kst.dt.strftime("%Y-%m-%d %H:%M").fillna("")

    return out.loc[:, OUTPUT_ORDER]


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """입력 CSV의 다양한 헤더를 표준 컬럼으로 정규화 + 중복 헤더 병합 + 헤더 클린업"""
    # 0) 🔧 헤더 클린업: 앞뒤 공백 제거 + BOM 제거
    if len(df.columns):
        cleaned = {}
        for c in df.columns:
            nc = str(c)
            nc = nc.replace("\ufeff", "")  # BOM 제거
            nc = nc.strip()                # 앞뒤 공백 제거
            cleaned[c] = nc
        if any(k != v for k, v in cleaned.items()):
            df = df.rename(columns=cleaned)

    cols_lower = {c.lower(): c for c in df.columns}

    def choose(cands):
        for cand in cands:
            # 정확 일치
            if cand in df.columns:
                return cand
            # 소문자 비교(공백/BOM 제거 후)
            cl = cand.lower()
            if cl in cols_lower:
                return cols_lower[cl]
        return None

    # 1) 표준명으로 리네임(없으면 생성)
    for std, cands in COL_CANDIDATES.items():
        chosen = choose(cands)
        if chosen is None:
            df[std] = pd.Series([pd.NA] * len(df))
        else:
            if chosen != std:
                df.rename(columns={chosen: std}, inplace=True)

    # 2) 중복 헤더 병합 (먼저 채워진 값 우선)
    if df.columns.duplicated().any():
        dup_names = set(df.columns[df.columns.duplicated(keep=False)])
        for name in dup_names:
            same_cols = [i for i, c in enumerate(df.columns) if c == name]
            merged = df.iloc[:, same_cols].bfill(axis=1).iloc[:, 0]
            df.drop(columns=[df.columns[i] for i in same_cols], inplace=True)
            df[name] = merged

    # 3) 날짜 파싱 (문자열 → Timestamp)
    for dcol in ["pub_date", "collected_at"]:
        if dcol in df.columns:
            ser = df[dcol]
            if isinstance(ser, pd.DataFrame):
                ser = ser.bfill(axis=1).iloc[:, 0]
            df[dcol] = pd.to_datetime(ser, errors="coerce")

    # 4) press 없으면 URL에서 도메인 추출
    if "press" in df.columns and "url" in df.columns:
        df["press"] = df["press"].fillna("")
        empty_press = df["press"].astype(str).str.strip().eq("")
        if empty_press.any():
            df.loc[empty_press, "press"] = df.loc[empty_press, "url"].apply(extract_domain)

    return df



# ---------------------------------------------------------------------
# 적재 / append-only
# ---------------------------------------------------------------------
def load_csvs(patterns):
    """입력 CSV 로딩 (출력 파일 제외)"""
    paths, frames = [], []
    ignore = {"news_raw.csv", "news_master.csv"}
    for p in patterns:
        paths.extend(glob.glob(p))
    for path in sorted(set(paths)):
        base = os.path.basename(path)
        if base in ignore:
            continue
        try:
            df = pd.read_csv(path)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] CSV 로드 실패: {path} - {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def anti_join_by_url_title(raw: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """동일 URL+제목 1차 중복 제거 (성능/용량 보호)"""
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
    # 기존 raw 로딩(한글 컬럼 → 표준 컬럼 역매핑)
    if os.path.exists(raw_path):
        raw = pd.read_csv(raw_path)
        raw_std = raw.rename(columns={
            "키워드": "keyword",
            "제목": "title",
            "원문링크": "url",
            "발행일(KST)": "pub_date",
            "수집시각(KST)": "collected_at",
            "출처": "press",
        })
        raw = map_columns(raw_std)
    else:
        raw = pd.DataFrame(columns=list(COL_CANDIDATES.keys()))

    # 신규 배치 로딩
    new_df = load_csvs(new_batch_paths)
    if new_df.empty:
        print("[INFO] 신규 배치 없음.")
        return raw

    new_df = map_columns(new_df)

    # 수집시각: tz-aware(KST)로 명시적 기록
    if "collected_at" in new_df.columns:
        new_df["collected_at"] = new_df["collected_at"].fillna(pd.Timestamp.now(tz=KST_TZ))

    # 1차 중복 제거
    new_df = anti_join_by_url_title(raw, new_df)
    if new_df.empty:
        print("[INFO] 추가할 신규 레코드가 없음.")
        return raw

    return pd.concat([raw, new_df], ignore_index=True)


# ---------------------------------------------------------------------
# 제목 유사도 기반 중복 제거
# ---------------------------------------------------------------------
PREFERRED_DOMAINS = [
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
    subset["_date"] = pd.to_datetime(
        subset["pub_date"] if "pub_date" in subset.columns else subset["collected_at"],
        errors="coerce"
    )
    subset["title_len"] = subset["title"].astype(str).str.len()
    subset = subset.sort_values(by=["dom_score", "_date", "title_len"],
                                ascending=[False, True, False])
    return subset.index[0]


def dedupe_by_title_similarity(raw_df: pd.DataFrame, sim_threshold=0.85) -> pd.DataFrame:
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


# ---------------------------------------------------------------------
# 엔드투엔드
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-batch", nargs="+", default=["data/*.csv"], help="신규 배치 CSV 경로(글롭 가능)")
    ap.add_argument("--raw", default="data/news_raw.csv", help="append-only 원본 CSV 경로")
    ap.add_argument("--master", default="data/news_master.csv", help="중복 제거/정제 결과 CSV 경로")
    ap.add_argument("--sim-threshold", type=float, default=0.85, help="제목 유사도 임계값 (기본 0.85)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.raw), exist_ok=True)
    os.makedirs(os.path.dirname(args.master), exist_ok=True)

    raw_std = append_only_raw(args.raw, args.new_batch)
    master_std = dedupe_by_title_similarity(raw_std, sim_threshold=args.sim_threshold)

    # 저장 직전, 6개 컬럼만(순서 고정)으로 투영
    raw_out = format_output_columns(raw_std).loc[:, OUTPUT_ORDER]
    master_out = format_output_columns(master_std).loc[:, OUTPUT_ORDER]

    raw_out.to_csv(args.raw, index=False, encoding="utf-8-sig")
    print(f"[OK] raw 저장: {args.raw} (rows={len(raw_out)})")

    master_out.to_csv(args.master, index=False, encoding="utf-8-sig")
    print(f"[OK] master 저장: {args.master} (rows={len(master_out)})")


if __name__ == "__main__":
    main()
