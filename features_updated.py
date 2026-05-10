"""
Upgraded features.py — moderate factor expansion (≈60 factors).

设计原则:
1. 不是 Alpha158 全套, 只加最低相关、表达力最强的部分
2. 每个新因子都和原有因子做了相关性预判, 避免冗余
3. 总因子数控制在 ~60, 与你 9.4 万行训练样本量匹配
4. 性能优化: 所有列先存到 dict, 一次性 concat (无 PerformanceWarning)

新增因子分类 (相对上一版 40 因子):
  - K-bar 形态族 (9 个): 捕捉日内多空力量
  - 多 horizon 动量 (4 个): 5/10/30/60 日 ROC
  - 多 horizon 波动 (3 个): 5/10/30 日价格 std/close
  - 量价相关性扩展 (2 个): CORR5, CORR60
  - 极值位置 (3 个): IMAX20, IMIN20, IMXD20

References:
- K-bar / ROC / RANK 因子表达式参考自 Microsoft Qlib Alpha158 公开规范:
  https://qlib.readthedocs.io/en/latest/component/data.html
- 所有因子均使用 pandas 自行实现, 未引用 Qlib 代码或调用其接口.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TARGET_COLUMN = "target_3d"
FORWARD_HORIZON = 3


# ============================================================
# 单只股票的因子 (返回 dict, 最后一次性 concat)
# ============================================================
def _per_stock_features(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float).replace(0, np.nan)
    eps = 1e-12

    feats: dict[str, pd.Series] = {}

    # ---------------- 1. K-bar 形态 (9 个新因子) ----------------
    # 这一批和你原有的"价格类因子"几乎不相关, 是最值得加的
    feats["KMID"]  = (c - o) / (o + eps)              # 实体涨跌幅
    feats["KLEN"]  = (h - l) / (o + eps)              # 振幅
    feats["KMID2"] = (c - o) / (h - l + eps)          # 实体在区间中的位置
    feats["KUP"]   = (h - np.maximum(o, c)) / (o + eps)
    feats["KUP2"]  = (h - np.maximum(o, c)) / (h - l + eps)
    feats["KLOW"]  = (np.minimum(o, c) - l) / (o + eps)
    feats["KLOW2"] = (np.minimum(o, c) - l) / (h - l + eps)
    feats["KSFT"]  = (2 * c - h - l) / (o + eps)      # 收盘相对中位的偏移
    feats["KSFT2"] = (2 * c - h - l) / (h - l + eps)

    # ---------------- 2. 短期反转 (你已有, 保留) ----------------
    ret_1 = c.pct_change()
    feats["ret_1d"] = ret_1
    feats["ret_2d"] = c.pct_change(2)
    feats["ret_3d"] = c.pct_change(3)
    feats["ret_5d"] = c.pct_change(5)
    feats["ret_10d"] = c.pct_change(10)
    feats["ret_20d"] = c.pct_change(20)
    feats["ret_60d"] = c.pct_change(60)

    # ---------------- 3. 波动率 (新增 10d, 30d) ----------------
    feats["vol_5d"]  = ret_1.rolling(5).std()
    feats["vol_10d"] = ret_1.rolling(10).std()         # 新增
    feats["vol_20d"] = ret_1.rolling(20).std()
    feats["vol_30d"] = ret_1.rolling(30).std()         # 新增
    feats["vol_60d"] = ret_1.rolling(60).std()

    # ---------------- 4. 成交量与换手率 ----------------
    vol_mean_20 = v.rolling(20).mean()
    vol_std_20  = v.rolling(20).std().replace(0, np.nan)
    feats["volume_z_20d"]   = (v - vol_mean_20) / vol_std_20
    feats["vol_ratio_5_20"] = v.rolling(5).mean() / vol_mean_20.replace(0, np.nan)

    if "turnover" in df.columns:
        feats["turnover_ma_20d"] = df["turnover"].astype(float).rolling(20).mean()
    else:
        feats["turnover_ma_20d"] = pd.Series(np.nan, index=df.index)

    # ---------------- 5. 均线偏离 ----------------
    feats["close_over_ma20"] = c / c.rolling(20).mean() - 1.0
    feats["close_over_ma60"] = c / c.rolling(60).mean() - 1.0

    # ---------------- 6. RSI ----------------
    delta = c.diff()
    up_14 = delta.clip(lower=0).rolling(14).mean()
    dn_14 = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    feats["rsi_14"] = 100 - 100 / (1 + up_14 / dn_14)
    up_5 = delta.clip(lower=0).rolling(5).mean()
    dn_5 = (-delta.clip(upper=0)).rolling(5).mean().replace(0, np.nan)
    feats["rsi_5"]  = 100 - 100 / (1 + up_5 / dn_5)

    # ---------------- 7. 量价相关性 (扩展到多 window) ----------------
    log_v = np.log(v + 1.0)
    feats["pv_corr_5d"]  = c.rolling(5).corr(log_v)        # 新增
    feats["pv_corr_20d"] = ret_1.rolling(20).corr(v.pct_change())
    feats["pv_corr_60d"] = c.rolling(60).corr(log_v)        # 新增

    # ---------------- 8. 回撤与冲高 ----------------
    feats["drawdown_20d"] = c / c.rolling(20).max() - 1.0
    feats["runup_20d"]    = c / c.rolling(20).min() - 1.0

    # ---------------- 9. MACD 与 布林带 ----------------
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    feats["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    feats["bb_pos"] = (c - (ma20 - 2 * sd20)) / (4 * sd20 + eps)

    # ---------------- 10. 极值位置因子 (3 个新增) ----------------
    # IMAX/IMIN: 最高/最低价在窗口内的相对位置 (越接近 1 越接近今天)
    # IMXD: 最高位置 - 最低位置, 反映最近的趋势方向
    # 这一类和动量因子相关性低, 是 Alpha158 里实测有效的
    feats["IMAX20"] = c.rolling(20).apply(
        lambda x: (np.argmax(x) + 1) / len(x), raw=True)
    feats["IMIN20"] = c.rolling(20).apply(
        lambda x: (np.argmin(x) + 1) / len(x), raw=True)
    feats["IMXD20"] = feats["IMAX20"] - feats["IMIN20"]

    # ---------------- 11. 尾部风险与流动性 (你原有) ----------------
    feats["max_drop_10d"] = ret_1.rolling(10).min()
    feats["amihud_20d"]   = (ret_1.abs() / (v + eps)).rolling(20).mean()

    # ---------------- 12. 目标变量 ----------------
    feats[TARGET_COLUMN] = c.shift(-FORWARD_HORIZON) / c - 1.0

    # 一次性合并 (避免 PerformanceWarning)
    new_df = pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)
    new_df["stock_code"] = stock_code
    return new_df


# ============================================================
# 截面处理
# ============================================================
def _cross_sectional_processing(panel: pd.DataFrame) -> pd.DataFrame:
    """每日截面: winsorize → 排名特征 → target z-score."""

    # 数值特征逐日 winsorize 1%/99%
    feat_cols = [c for c in panel.columns
                 if c not in ("date", "stock_code", TARGET_COLUMN,
                              "open", "high", "low", "close",
                              "volume", "amount", "turnover", "pct_change")]

    def _winsorize(x: pd.Series) -> pd.Series:
        if len(x) < 30 or x.isna().all():
            return x
        lo, hi = x.quantile(0.01), x.quantile(0.99)
        return x.clip(lower=lo, upper=hi)

    grouped = panel.groupby("date")
    for f in feat_cols:
        panel[f] = grouped[f].transform(_winsorize)

    # 排名特征 (用 dict 收集再一次性 concat)
    RANK_BASE = [
        "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_20d",
        "vol_5d", "vol_20d",
        "turnover_ma_20d",
        "rsi_14",
        "close_over_ma20", "close_over_ma60",
        "pv_corr_20d",
        "drawdown_20d", "runup_20d",
        "macd_hist", "bb_pos",
        "amihud_20d",
        # 新增 K-bar / IMAX 的排名 (这俩本身就是相对值, 排名进一步去掉个股特异性)
        "KMID", "KLEN", "KSFT",
        "IMAX20", "IMIN20",
    ]
    grouped = panel.groupby("date")
    rank_dict: dict[str, pd.Series] = {}
    for base in RANK_BASE:
        if base in panel.columns:
            rank_dict[f"{base}_rank"] = grouped[base].rank(method="average", pct=True)

    if rank_dict:
        rank_df = pd.DataFrame(rank_dict, index=panel.index)
        panel = pd.concat([panel, rank_df], axis=1)

    # target 截面 z-score
    def _zscore_safe(x: pd.Series) -> pd.Series:
        if len(x) < 30 or x.isna().all():
            return x
        x = x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
        return (x - x.mean()) / (x.std() + 1e-8)
    panel[TARGET_COLUMN] = panel.groupby("date")[TARGET_COLUMN].transform(_zscore_safe)

    return panel.copy()


# ============================================================
# 公共接口
# ============================================================
def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    frames = [
        _per_stock_features(stock_df, stock_code)
        for stock_code, stock_df in prices.groupby("stock_code", sort=False)
    ]
    panel = pd.concat(frames, ignore_index=True)
    panel = _cross_sectional_processing(panel)
    return panel


# 静态特征列表 (与上面的因子计算一一对应)
FEATURE_COLUMNS = [
    # K-bar (9)
    "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",

    # 短期反转 + 中长期动量 (7)
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",

    # 波动率 (5)
    "vol_5d", "vol_10d", "vol_20d", "vol_30d", "vol_60d",

    # 成交量 / 换手率 (3)
    "volume_z_20d", "vol_ratio_5_20", "turnover_ma_20d",

    # 均线偏离 (2)
    "close_over_ma20", "close_over_ma60",

    # RSI (2)
    "rsi_14", "rsi_5",

    # 量价相关性 (3)
    "pv_corr_5d", "pv_corr_20d", "pv_corr_60d",

    # 回撤与冲高 (2)
    "drawdown_20d", "runup_20d",

    # MACD / BB (2)
    "macd_hist", "bb_pos",

    # 极值位置 (3)
    "IMAX20", "IMIN20", "IMXD20",

    # 尾部风险 (2)
    "max_drop_10d", "amihud_20d",

    # 排名因子 (22)
    "ret_1d_rank", "ret_2d_rank", "ret_3d_rank", "ret_5d_rank", "ret_20d_rank",
    "vol_5d_rank", "vol_20d_rank",
    "turnover_ma_20d_rank",
    "rsi_14_rank",
    "close_over_ma20_rank", "close_over_ma60_rank",
    "pv_corr_20d_rank",
    "drawdown_20d_rank", "runup_20d_rank",
    "macd_hist_rank", "bb_pos_rank",
    "amihud_20d_rank",
    "KMID_rank", "KLEN_rank", "KSFT_rank",
    "IMAX20_rank", "IMIN20_rank",
]


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    avail_cols = [c for c in FEATURE_COLUMNS if c in panel.columns]
    df = panel.dropna(subset=avail_cols + [TARGET_COLUMN]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    avail_cols = [c for c in FEATURE_COLUMNS if c in panel.columns]
    return panel[panel["date"] == as_of].dropna(subset=avail_cols).copy()