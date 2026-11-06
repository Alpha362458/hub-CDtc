"""Hugging Face 股票日线数据集加载与 5 维特征工程管线。

数据源：
    Hugging Face Hub: no-ry/world-stock-prices-daily-updating
    直链: https://huggingface.co/datasets/no-ry/world-stock-prices-daily-updating/resolve/main/World-Stock-Prices-Dataset.csv

五维特征向量 x = [x1, x2, x3, x4, x5]^T：
    x1 (cagr)          : 10年期（或窗口期）年化收益率 (P_end / P_start)^(1 / years) - 1
    x2 (volatility)    : 年化波动率 std(r_daily) * sqrt(252)
    x3 (max_drawdown)  : 历史最大回撤 max_t ((Peak_t - P_t) / Peak_t)
    x4 (peak_drop)     : 期末距历史最高点回落幅度 (P_max - P_end) / P_max
    x5 (swing_amp)     : 60日滚动波段高低点振幅均值 mean((P_max60 - P_min60) / P_min60)

五个股票分类标签 y in {0, 1, 2, 3, 4}：
    0: 10年期暴利大牛股 (ten_bagger)      —— 代表: NVDA, MSFT, AAPL, COST, AMZN
    1: 长期稳健持有股   (steady_hold)     —— 代表: KO, PG, JNJ, MCD, UL
    2: 适合炒波段股     (swing_trade)     —— 代表: TSLA, AMD, SHOP, COIN, RBLX, CROX
    3: 风口暴涨及时撤退 (bubble_retreat)  —— 代表: PTON, ZM, SQ, ZI, UBSFY
    4: 完全不考虑股     (avoid)           —— 代表: DIS, MMM, HMC, LUV, PHG
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HF_DATASET_ID = "no-ry/world-stock-prices-daily-updating"
HF_CSV_URL = (
    "https://huggingface.co/datasets/no-ry/world-stock-prices-daily-updating/"
    "resolve/main/World-Stock-Prices-Dataset.csv"
)

FEATURE_NAMES: List[str] = [
    "cagr",          # x1: 年化收益率
    "volatility",    # x2: 年化波动率
    "max_drawdown",  # x3: 历史最大回撤
    "peak_drop",     # x4: 峰值回落幅度
    "swing_amp",     # x5: 60日波段振幅均值
]

FEATURE_LABELS_ZH: List[str] = [
    "x1: 10年年化收益率 (CAGR)",
    "x2: 年化波动率 (Volatility)",
    "x3: 最大回撤 (Max Drawdown)",
    "x4: 峰值回落比 (Peak Drop)",
    "x5: 60日波段振幅 (Swing Amp)",
]

CLASS_NAMES_ZH: List[str] = [
    "0: 10年暴利大牛股",
    "1: 长期稳健持有股",
    "2: 适合炒波段股",
    "3: 风口暴涨及时撤退",
    "4: 完全不考虑股",
]

CLASS_SHORT_ZH: List[str] = [
    "10年暴利大牛",
    "长期稳健持有",
    "适合炒波段",
    "风口暴涨撤退",
    "完全不考虑",
]


@dataclass
class StockDataset:
    """标准化后的训练/测试数据集与原始特征统计量。"""

    X_train: np.ndarray          # (N_train, 5) 标准化后特征
    y_train: np.ndarray          # (N_train,)   类别标签 0..4
    X_test: np.ndarray           # (N_test, 5)  标准化后特征
    y_test: np.ndarray           # (N_test,)    类别标签 0..4
    X_train_raw: np.ndarray      # (N_train, 5) 原始未标准化特征
    X_test_raw: np.ndarray       # (N_test, 5)  原始未标准化特征
    mean: np.ndarray             # (5,)         训练集均值（用于推理标准化）
    std: np.ndarray              # (5,)         训练集标准差（用于推理标准化）
    tickers_train: List[str]
    tickers_test: List[str]
    showcase_stocks: pd.DataFrame  # 代表性知名股票最新10年窗口特征快照

    def normalize(self, x_raw: np.ndarray) -> np.ndarray:
        """将任意原始 5 维向量 (5,) 或 (B, 5) 按训练集均值/标准差做 Z-score 标准化。"""
        return (np.asarray(x_raw, dtype=np.float64) - self.mean) / self.std


def clean_split_artifacts(prices: np.ndarray) -> np.ndarray:
    """修复日更数据集中拆股（如 NVDA 1拆10、CMG 1拆50）造成的单日断层。"""
    p = np.asarray(prices, dtype=np.float64).copy()
    p = p[p > 0]
    if len(p) < 2:
        return p
    ratios = p[1:] / p[:-1]
    jump_indices = np.where((ratios < 0.35) | (ratios > 2.8))[0]
    for idx in jump_indices:
        factor = p[idx + 1] / p[idx]
        p[: idx + 1] *= factor
    return p


def compute_5d_features(prices: np.ndarray) -> np.ndarray:
    """从一段日收盘价序列计算出 5 维特征向量 x = [cagr, vol, mdd, peak_drop, swing_amp]。"""
    n = len(prices)
    years = n / 252.0
    cagr = float((prices[-1] / prices[0]) ** (1.0 / years) - 1.0)

    daily_ret = np.diff(prices) / prices[:-1]
    volatility = float(np.std(daily_ret) * np.sqrt(252.0))

    peaks = np.maximum.accumulate(prices)
    drawdowns = (peaks - prices) / peaks
    max_drawdown = float(np.max(drawdowns))

    p_max = float(np.max(prices))
    peak_drop = float((p_max - prices[-1]) / p_max)

    win = min(60, n)
    windows = sliding_window_view(prices, window_shape=win)
    w_max = np.max(windows, axis=1)
    w_min = np.min(windows, axis=1)
    swing_amp = float(np.mean((w_max - w_min) / np.maximum(w_min, 1e-12)))

    return np.array([cagr, volatility, max_drawdown, peak_drop, swing_amp], dtype=np.float64)


def classify_5d_vector(x: np.ndarray) -> int:
    """根据五维量化指标判定股票所属的 5 大类型（0..4）。"""
    cagr, vol, mdd, pdrop, swing = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])

    # 类别 3：风口暴涨需及时撤退（高波动/高振幅 + 历史回撤极深 + 期末距高点暴跌逾 62%）
    if pdrop >= 0.62 and mdd >= 0.72 and (vol >= 0.40 or swing >= 0.32):
        return 3

    # 类别 0：10年期暴利大牛股（年化收益率 >= 19% 且期末保持在高位附近）
    if cagr >= 0.19 and pdrop < 0.30:
        return 0

    # 类别 1：长期稳健持有股（年化 5%~19%，波动率低 <= 28%，最大回撤 <= 45%）
    if cagr >= 0.05 and vol <= 0.28 and mdd <= 0.45 and pdrop < 0.35:
        return 1

    # 类别 2：适合炒波段股（波段振幅 >= 25% 或波动率 >= 32%，正收益/微利且未崩盘归零）
    if (swing >= 0.25 or vol >= 0.32) and cagr >= 0.015 and pdrop < 0.62:
        return 2

    # 类别 4：完全不考虑股（长期阴跌、收益率跑输无风险利率或回撤后长期半死不活）
    return 4


def ensure_hf_raw_csv(data_dir: str) -> str:
    """确保本地存在 Hugging Face 原始股票 CSV；若不存在则自动从 HF Hub 下载。"""
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "World-Stock-Prices-Dataset.csv")
    if not os.path.exists(csv_path):
        print(f"[HF] 正在从 Hugging Face 下载原始数据集 ({HF_DATASET_ID})...")
        urllib.request.urlretrieve(HF_CSV_URL, csv_path)
    return csv_path


def build_or_load_samples(
    data_dir: Optional[str] = None,
    rebuild: bool = False,
    max_per_class: int = 500,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从 Hugging Face 原始日线提取滑动长周期窗口 5 维特征样本，并缓存到 CSV。"""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    samples_cache = os.path.join(data_dir, "stock_5class_samples.csv")
    showcase_cache = os.path.join(data_dir, "showcase_stocks_10y.csv")

    if not rebuild and os.path.exists(samples_cache) and os.path.exists(showcase_cache):
        return pd.read_csv(samples_cache), pd.read_csv(showcase_cache)

    raw_csv = ensure_hf_raw_csv(data_dir)
    df = pd.read_csv(raw_csv, usecols=["Date", "Ticker", "Brand_Name", "Close"])
    df["Date_str"] = df["Date"].astype(str).str.slice(0, 10)
    df = df.sort_values(["Ticker", "Date_str"]).drop_duplicates(["Ticker", "Date_str"])

    all_rows: List[Dict[str, object]] = []
    showcase_rows: List[Dict[str, object]] = []

    for ticker, group in df.groupby("Ticker", sort=True):
        prices_all = clean_split_artifacts(group["Close"].to_numpy())
        brand = str(group["Brand_Name"].iloc[0])
        if len(prices_all) < 756:
            continue

        # 1) 最新 10 年（2520 个交易日）快照，用于代表性个股展示
        latest_p = prices_all[-2520:]
        feat_latest = compute_5d_features(latest_p)
        label_latest = classify_5d_vector(feat_latest)
        showcase_rows.append(
            {
                "Ticker": ticker,
                "Brand_Name": brand,
                "Years": round(len(latest_p) / 252.0, 1),
                "cagr": feat_latest[0],
                "volatility": feat_latest[1],
                "max_drawdown": feat_latest[2],
                "peak_drop": feat_latest[3],
                "swing_amp": feat_latest[4],
                "label": label_latest,
                "class_name": CLASS_SHORT_ZH[label_latest],
            }
        )

        # 2) 多尺度长周期滑动窗口（10年=2520天, 7.5年=1890天, 5年=1260天, 3.2年=800天）
        win_lengths = [2520, 1890, 1260] if len(prices_all) >= 2520 else [min(len(prices_all), 1260), 800]
        for w in win_lengths:
            if len(prices_all) < w:
                continue
            # 对数据较少的近期新股（如 PTON, ZM, ZI, COIN）加密采样步长以平衡样本数
            stride = 15 if len(prices_all) < 2520 else 45
            for start in range(0, len(prices_all) - w + 1, stride):
                window_p = prices_all[start : start + w]
                feat = compute_5d_features(window_p)
                label = classify_5d_vector(feat)
                all_rows.append(
                    {
                        "Ticker": ticker,
                        "Brand_Name": brand,
                        "window_days": w,
                        "cagr": feat[0],
                        "volatility": feat[1],
                        "max_drawdown": feat[2],
                        "peak_drop": feat[3],
                        "swing_amp": feat[4],
                        "label": label,
                    }
                )

    samples_df = pd.DataFrame(all_rows)
    showcase_df = pd.DataFrame(showcase_rows)

    # 按类别均衡抽样，保证 5 类样本均衡且训练在几秒内完成
    rng = np.random.default_rng(seed)
    balanced_parts = []
    for c in range(5):
        sub = samples_df[samples_df["label"] == c]
        if len(sub) >= max_per_class:
            idx = rng.choice(len(sub), size=max_per_class, replace=False)
            balanced_parts.append(sub.iloc[idx])
        else:
            # 若某类略少于 max_per_class，对已有真实窗口做微小高斯扰动扩充至 max_per_class
            need = max_per_class - len(sub)
            extra_idx = rng.choice(len(sub), size=need, replace=True)
            extra = sub.iloc[extra_idx].copy()
            for col in FEATURE_NAMES:
                noise = rng.normal(0.0, 0.008, size=need)
                extra[col] = np.clip(extra[col].to_numpy() + noise, -0.95, 5.0)
            # 重新校验扩充样本标签一致性
            valid_labels = [
                classify_5d_vector(row)
                for row in extra[FEATURE_NAMES].to_numpy()
            ]
            extra["label"] = valid_labels
            extra = extra[extra["label"] == c]
            balanced_parts.append(pd.concat([sub, extra], ignore_index=True))

    final_df = pd.concat(balanced_parts, ignore_index=True)
    # 打乱顺序
    perm = rng.permutation(len(final_df))
    final_df = final_df.iloc[perm].reset_index(drop=True)

    final_df.to_csv(samples_cache, index=False)
    showcase_df.to_csv(showcase_cache, index=False)
    return final_df, showcase_df


def load_stock_dataset(
    data_dir: Optional[str] = None,
    test_ratio: float = 0.2,
    max_per_class: int = 500,
    seed: int = 42,
    rebuild: bool = False,
) -> StockDataset:
    """加载并切分 80% 训练集 / 20% 测试集，计算 Z-score 标准化参数。"""
    samples_df, showcase_df = build_or_load_samples(
        data_dir=data_dir, rebuild=rebuild, max_per_class=max_per_class, seed=seed
    )

    X_raw = samples_df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = samples_df["label"].to_numpy(dtype=np.int64)
    tickers = samples_df["Ticker"].astype(str).tolist()

    rng = np.random.default_rng(seed)
    train_idx_list, test_idx_list = [], []
    # 分层抽样（Stratified Split），保证训练集和测试集 5 类比例一致
    for c in range(5):
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        n_test = int(round(len(c_idx) * test_ratio))
        test_idx_list.append(c_idx[:n_test])
        train_idx_list.append(c_idx[n_test:])

    train_idx = np.concatenate(train_idx_list)
    test_idx = np.concatenate(test_idx_list)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train_raw = X_raw[train_idx]
    y_train = y[train_idx]
    X_test_raw = X_raw[test_idx]
    y_test = y[test_idx]

    mean = X_train_raw.mean(axis=0)
    std = np.maximum(X_train_raw.std(axis=0), 1e-8)

    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    return StockDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        mean=mean,
        std=std,
        tickers_train=[tickers[i] for i in train_idx],
        tickers_test=[tickers[i] for i in test_idx],
        showcase_stocks=showcase_df,
    )
