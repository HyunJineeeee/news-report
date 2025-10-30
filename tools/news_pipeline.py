# -*- coding: utf-8 -*-
"""
news_pipeline.py (최종 버전)
---------------------------------------
- append-only 원본 누적(raw)
- 제목 유사도 기반 중복 제거 → 대표 기사 1건 선정(master)
- 제목 기반 Top 키워드 Bar 차트 (주/월/분기)
- ✅ 저장 시 컬럼 순서 고정:
    ["키워드", "제목", "원문링크", "발행일(KST)", "수집시각(KST)", "출처"]
- ✅ 차트 디자인 개선 / 한글 폰트(Nanum, Noto) 적용 / 값 라벨 표시
"""

import argparse
import glob
import os
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------
# 0. 그래프 테마 및 폰트 설정
# ---------------------------------------------------------------------
def setup_matplotlib_theme():
    """폰트/스타일 기본 세팅"""
    preferred_fonts = ["NanumGothic", "Noto Sans CJK KR", "AppleGothic", "Malgun Gothic"]
    for fam in preferred_fonts:
        try:
            matplotlib.rcParams["font.family"] = fam
            break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False
    plt.style.use("ggplot")
    matplotlib.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
        "axes.edgecolor": "#DDDDDD",
        "grid.alpha": 0.4,
    })


# ---------------------------------------------------------------------
# 1. 기본 설정 / 컬럼 정의
# ---------------------------------------------------------------------
KST_TZ = "Asia/Seoul"
OUTPUT_ORDER = ["키워드", "제목", "원문링크", "발행일(KST)", "수집시각(KST)", "출처"]

COL_CANDIDATES = {
    "keyword": ["keyword", "키워드"],
    "title": ["title", "subject", "headline", "제목"],
    "url": ["url", "link", "기사링크", "news_url", "원문링크"],
    "pub_date": ["pub_date", "date", "published_at", "publish_date", "발행일", "작성일"],
    "press": ["press", "source", "publisher", "언론사", "매체", "출처"],
    "collected_at": ["collected_at", "staged_at", "crawled_at", "수집일", "수집일자", "수집시각"],
    "content": ["content", "summary", "본문", "요약"],
}


# ---------------------------------------------------------------------
# 2. 유틸 함수
# ---------------------------------------------------------------------
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

    # 1) 표준 컬럼명으로 리네임 / 없으면 생성
    for std, cands in COL_CANDIDATES.items():
        chosen = choose(cands)
        if chosen is None:
            df[std] = pd.Series([np.nan] * len(df))
        else:
            if chosen != std:
                df.rename(columns={chosen: std}, inplace=True)

    # 2) 🔧 같은 이름의 컬럼이 둘 이상이면 하나로 병합 (먼저 나오는 값 우선)
    #    예: ['pub_date','pub_date'] 같은 중복 헤더
    if df.columns.duplicated().any():
        dup_names = set(df.columns[df.columns.duplicated(keep=False)])
        for name in dup_names:
            # 해당 이름의 모든 컬럼을 모아 앞쪽 값으로 채우기
            same_cols = [i for i, c in enumerate(df.columns) if c == name]
            merged = df.iloc[:, same_cols].bfill(axis=1).iloc[:, 0]
            # 기존 중복 컬럼들 제거 후 하나만 남김
            df.drop(columns=[df.columns[i] for i in same_cols], inplace=True)
            df[name] = merged

    # 3) 날짜 파싱 (여기서부터는 단일 Series 보장)
    for dcol in ["pub_date", "collected_at"]:
        if dcol in df.columns:
            ser = df[dcol]
            # 혹시 모를 DataFrame 방지(이중 안전장치)
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


def to_kst_ts(ts):
    if pd.isna(ts):
        return pd.NaT
    ts = pd.to_datetime(ts, errors="coerce")
    if ts is pd.NaT or ts is None:
        return pd.NaT
    try:
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            return ts.tz_localize(KST_TZ)
        else:
            return ts.tz_convert(KST_TZ)
    except Exception:
        return pd.NaT


def format_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """최종 저장용 컬럼 순서/형식 변환"""
    out = pd.DataFrame(index=df.index.copy())
    out["키워드"] = df.get("keyword", "").fillna("").astype(str)
    out["제목"] = df.get("title", "").astype(str)
    out["원문링크"] = df.get("url", "").astype(str)
    out["출처"] = df.get("press", "").astype(str)
    pub_kst = pd.to_datetime(df.get("pub_date", pd.NaT), errors="coerce").apply(to_kst_ts)
    col_kst = pd.to_datetime(df.get("collected_at", pd.NaT), errors="coerce").apply(to_kst_ts)
    out["발행일(KST)"] = pub_kst.dt.strftime("%Y-%m-%d %H:%M").fillna("")
    out["수집시각(KST)"] = col_kst.dt.strftime("%Y-%m-%d %H:%M").fillna("")
    return out.reindex(columns=OUTPUT_ORDER)


# ---------------------------------------------------------------------
# 3. 데이터 적재 / append-only
# ---------------------------------------------------------------------
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

    new_df = load_csvs(new_batch_paths)
    if new_df.empty:
        print("[INFO] 신규 배치 없음.")
        return raw

    new_df = map_columns(new_df)
    if "collected_at" in new_df.columns:
        new_df["collected_at"] = new_df["collected_at"].fillna(pd.Timestamp.now())

    new_df = anti_join_by_url_title(raw, new_df)
    if new_df.empty:
        print("[INFO] 추가할 신규 레코드가 없음.")
        return raw

    appended = pd.concat([raw, new_df], ignore_index=True)
    return appended


# ---------------------------------------------------------------------
# 4. 제목 유사도 기반 중복 제거
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
    if "pub_date" in subset.columns:
        subset["_date"] = pd.to_datetime(subset["pub_date"], errors="coerce")
    else:
        subset["_date"] = pd.to_datetime(subset["collected_at"], errors="coerce")
    subset["title_len"] = subset["title"].astype(str).str.len()
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


# ---------------------------------------------------------------------
# 5. 키워드 분석 / 시각화
# ---------------------------------------------------------------------
STOPWORDS = {
    "일학습병행","직업훈련","고용노동부","한국산업인력공단",
    "네이버","다음","daum","naver","kr","com","net","co","news","연합뉴스",
    "기사","사진","영상","보도","속보","단독","종합","인터뷰",
    "기자","관련","정부","사업","지원","추진"
}


def normalize_title_korean(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    cleaned = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    toks = [t for t in re.split(r"\s+", cleaned.strip()) if t]
    out = []
    for t in toks:
        if len(t) < 2:
            continue
        if re.fullmatch(r"\d+", t):
            continue
        if re.fullmatch(r"20\d{2}", t):
            continue
        if re.fullmatch(r"[A-Za-z]{2,5}", t) and t.lower() in {"kr","com","net","co"}:
            continue
        if t in STOPWORDS:
            continue
        out.append(t)
    return out


def _prettify_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def top_keywords_bar(df: pd.DataFrame, outdir: str, period: str = "W", top_k=25):
    if df.empty:
        return
    os.makedirs(outdir, exist_ok=True)

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

        h = max(5, 0.35 * len(vc) + 1.5)
        fig, ax = plt.subplots(figsize=(10, h))
        bars = ax.barh(vc.index[::-1], vc.values[::-1])

        for rect in bars:
            w = rect.get_width()
            ax.text(w + max(vc.values) * 0.01, rect.get_y() + rect.get_height()/2,
                    f"{int(w)}", va="center", ha="left", fontsize=10)

        ax.set_xlabel("빈도")
        ax.set_ylabel("")
        ax.set_title(f"Top Keywords ({period}) - {pval.date()}")
        ax.grid(axis="x")
        _prettify_axes(ax)
        ax.margins(x=0.05)

        fname = f"top_keywords_{period}_{pval.date()}.png"
        save_path = os.path.join(outdir, fname)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] 저장: {save_path}")


# ---------------------------------------------------------------------
# 6. 엔드투엔드 실행
# ---------------------------------------------------------------------
def main():
    setup_matplotlib_theme()

    ap = argparse.ArgumentParser()
    ap.add_argument("--new-batch", nargs="+", default=["data/*.csv"],
                    help="신규 배치 CSV 경로(글롭 가능)")
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

    raw_std = append_only_raw(args.raw, args.new_batch)
    master_std = dedupe_by_title_similarity(raw_std, sim_threshold=args.sim_threshold)

    raw_out = format_output_columns(raw_std)
    master_out = format_output_columns(master_std)

    raw_out.to_csv(args.raw, index=False, encoding="utf-8-sig")
    print(f"[OK] raw 저장(고정 컬럼): {args.raw} (rows={len(raw_out)})")

    master_out.to_csv(args.master, index=False, encoding="utf-8-sig")
    print(f"[OK] master 저장(고정 컬럼): {args.master} (rows={len(master_out)})")

    for period in ["W", "M", "Q"]:
        top_keywords_bar(master_std, args.outdir, period=period, top_k=25)


if __name__ == "__main__":
    main()
