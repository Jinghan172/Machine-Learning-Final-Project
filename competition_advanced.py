"""
Advanced Ensemble for CSI500 Stock Selection.

本版本相对上一版的关键修改:
  1. LGBM 移除 huber / mae loss, 回归默认 squared error
     —— z-score 后的 target 是对称分布, huber/mae 会把预测压向 0
  2. 移除 RISK_PENALTY (= 0.0)
     —— 之前的 vol_z 扣分把 alpha 反向抵消, 直接用模型 score_z
  3. min_turnover 0.002 → 0.0005, 不再过度过滤大票
  4. 修复 generate_submission 里 top_k 硬编码 80 的 bug, 改用形参
  5. 删除重复的 XGB_CONFIGS / LGB_CONFIGS 定义
  6. EMBARGO_DAYS 放宽到 max(5, FORWARD_HORIZON+2), 给一点安全垫
  7. 移除无效的 os.environ['PYTHONHASHSEED'] 语句
  8. main 加 --windows 和 --skip-backtest 参数
  9. 输出新增预测池大小 / IR 指标
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import spearmanr

from features_updated import (
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features,
    prediction_frame,
    training_frame,
)
from score_submission import score_window

# ============================================================
# 全局配置
# ============================================================
GLOBAL_SEED = 42

def _set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Python / numpy 全局随机源, 保证赛规 §10 的可复现要求.
    注意: PYTHONHASHSEED 必须在解释器启动前设置才有效, 故不在此处设置.
    复现命令: PYTHONHASHSEED=42 python competition_advanced.py
    """
    random.seed(seed)
    np.random.seed(seed)


DATA_DIR = Path(__file__).parent / "data"
VAL_DAYS = 10
EMBARGO_DAYS = max(5, FORWARD_HORIZON + 2)   # train/val 之间的隔离区, 防止 label 泄漏
MIN_STOCKS = 30
MAX_WEIGHT = 0.10
DEFAULT_TOP_K = 30
DEFAULT_WEIGHTING = "rank"
DEFAULT_BLEND_ALPHA = 0.7
TRAIN_HALF_LIFE_DAYS = 120
MIN_TURNOVER = 0.0005   # 流动性下限. 太严会把权重股(银行/电力/白马)误杀

# 5 个 seed × 3 配置 = 15 个模型集成, 降方差
ENSEMBLE_SEEDS = [42, 7, 99, 2024, 1234]

XGB_CONFIGS = [
    {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03,
     "subsample": 0.7, "colsample_bytree": 0.7,
     "reg_alpha": 1.0, "reg_lambda": 2.0},
]
LGB_CONFIGS = [
    {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03,
     "subsample": 0.7, "colsample_bytree": 0.7,
     "subsample_freq": 1,
     "reg_alpha": 1.0, "reg_lambda": 2.0,
     "verbose": -1},
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.02,
     "subsample": 0.8, "colsample_bytree": 0.6,
     "subsample_freq": 1,
     "reg_alpha": 0.5, "reg_lambda": 3.0,
     "verbose": -1},
]


# ============================================================
# 评测函数
# ============================================================
def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    """每日截面 Spearman 相关, 跨日平均."""
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


# ============================================================
# 组合构造
# ============================================================
def build_portfolio(
    scores: pd.Series,
    top_k: int = DEFAULT_TOP_K,
    weighting: str = "rank",
    blend_alpha: float = 0.7,
) -> pd.Series:
    """选 top_k 只票, 按指定方案分配权重, 封顶 MAX_WEIGHT.

    weighting:
        "rank"      — 纯线性排名权重 (best=K, worst=1)
        "equal"     — 等权 1/K, 跟踪误差最低
        "blend"     — (1-α)*equal + α*rank, 推荐默认 α=0.5
        "sqrt_rank" — weight ∝ sqrt(rank), 介于 rank 和 equal 之间
    """
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k 必须 >= {MIN_STOCKS} (比赛规则)")

    scores = scores.dropna()
    if len(scores) < top_k:
        raise ValueError(f"可打分股票仅 {len(scores)} 只, 不足 top_k={top_k}")
    scores = scores.sort_values(ascending=False, kind="mergesort")
    chosen = scores.head(top_k)

    if weighting == "rank":
        raw = np.arange(top_k, 0, -1, dtype=float)
    elif weighting == "equal":
        raw = np.ones(top_k, dtype=float)
    elif weighting == "sqrt_rank":
        raw = np.sqrt(np.arange(top_k, 0, -1, dtype=float))
    elif weighting == "blend":
        rank_w = np.arange(top_k, 0, -1, dtype=float); rank_w /= rank_w.sum()
        eq_w = np.full(top_k, 1.0 / top_k)
        raw = blend_alpha * rank_w + (1.0 - blend_alpha) * eq_w
    else:
        raise ValueError(f"未知的 weighting 方案: {weighting!r}")

    weights = pd.Series(raw / raw.sum(), index=chosen.index)

    # 迭代封顶: 把超过 MAX_WEIGHT 的削平, 溢出量按现权重比例分给未封顶的票
    for _ in range(top_k + 5):
        over = weights > MAX_WEIGHT + 1e-12
        if not over.any():
            break
        excess = (weights[over] - MAX_WEIGHT).sum()
        weights[over] = MAX_WEIGHT
        free = ~over
        if not free.any() or weights[free].sum() <= 0:
            break
        weights[free] += excess * weights[free] / weights[free].sum()

    weights = weights.clip(lower=0.0, upper=MAX_WEIGHT)
    total = weights.sum()
    if total <= 0:
        raise RuntimeError("所有权重为 0, 无法构建组合")
    weights = weights / total
    assert abs(weights.sum() - 1.0) < 1e-6, f"权重和={weights.sum()}"
    assert (weights <= MAX_WEIGHT + 1e-9).all(), "封顶失败"
    assert (weights > 0).sum() >= MIN_STOCKS, f"持仓票数 < {MIN_STOCKS}"
    return weights


# ============================================================
# 训练 / 验证集切分
# ============================================================
def make_train_val_split(panel: pd.DataFrame, as_of: pd.Timestamp):
    """按时间切: [TRAIN] [embargo, 丢弃] [VAL, 末 VAL_DAYS 天] [GAP] [as_of].

    - GAP: train_pool 的最后一天 = as_of - FORWARD_HORIZON,
           防止 train label 偷看 as_of 当天价格.
    - embargo: train_end 和 val_start 之间留 EMBARGO_DAYS 天,
               防止 train label 跨进 val 期 (label 用了 t+horizon 价格).
    """
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])

    train_pool = training_frame(panel, max_date=train_cutoff)
    all_dates = np.sort(train_pool["date"].unique())

    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    train_df = train_pool[train_pool["date"] <= train_end].copy()
    val_df = train_pool[train_pool["date"] >= val_start].copy()
    return train_df, val_df, train_end


# ============================================================
# 模型训练 / 集成预测
# ============================================================
def make_time_decay_weights(train_df: pd.DataFrame, half_life_days: int = TRAIN_HALF_LIFE_DAYS) -> np.ndarray:
    """Recent samples get larger weight to adapt to regime drift."""
    if half_life_days <= 0:
        return np.ones(len(train_df), dtype=float)
    dates = pd.to_datetime(train_df["date"])
    age_days = (dates.max() - dates).dt.days.astype(float)
    weights = np.power(0.5, age_days / half_life_days)
    return np.asarray(weights / weights.mean(), dtype=float)


def train_models(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """训 5 seeds × (1 XGB + 2 LGB) = 15 个模型."""
    models = []
    sample_weight = make_time_decay_weights(train_df)
    for seed in ENSEMBLE_SEEDS:
        for cfg in XGB_CONFIGS:
            m = xgb.XGBRegressor(
                tree_method="hist", n_jobs=-1,
                early_stopping_rounds=50,
                random_state=seed,
                **cfg,
            )
            m.fit(
                train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
                sample_weight=sample_weight,
                eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])],
                verbose=False,
            )
            models.append(m)
        for cfg in LGB_CONFIGS:
            m = lgb.LGBMRegressor(n_jobs=-1, random_state=seed, **cfg)
            m.fit(
                train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
                sample_weight=sample_weight,
                eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            models.append(m)
    return models


def ensemble_predict(models, features: pd.DataFrame) -> np.ndarray:
    preds = [model.predict(features[FEATURE_COLUMNS]) for model in models]
    return np.mean(preds, axis=0)


# ============================================================
# 提交生成 (核心逻辑)
# ============================================================
def generate_submission(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    top_k: int,
    weighting: str = DEFAULT_WEIGHTING,
    blend_alpha: float = DEFAULT_BLEND_ALPHA,
    verbose: bool = False,
) -> tuple[pd.DataFrame, float]:
    train_df, val_df, _ = make_train_val_split(panel, as_of)
    models = train_models(train_df, val_df)

    # 验证集 IC, 仅作监控指标, 不参与决策
    val_pred = ensemble_predict(models, val_df)
    ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred, val_df["date"].to_numpy())

    # 预测日打分集合: 流动性 + 成交量过滤
    pred_df = prediction_frame(panel, as_of=as_of).copy()
    pred_df = pred_df[
        (pred_df["turnover_ma_20d"] >= MIN_TURNOVER) & (pred_df["volume"] > 0)
    ].copy()

    if verbose:
        print(f"  as_of={pd.Timestamp(as_of).date()}  "
              f"pred_universe={len(pred_df)}  val_IC={ic:+.4f}")

    if len(pred_df) < top_k:
        raise RuntimeError(
            f"流动性过滤后仅剩 {len(pred_df)} 只票, 不足 top_k={top_k}. "
            f"请降低 MIN_TURNOVER (当前 {MIN_TURNOVER})."
        )

    # ---------------- 打分: 直接用 ensemble 输出做截面 z-score ----------------
    # 因为 A 股短期高波动票平均收益更高 (高 beta + 散户追涨现象).
    pred_df["raw_score"] = ensemble_predict(models, pred_df)
    pred_df["score_z"] = (
        (pred_df["raw_score"] - pred_df["raw_score"].mean())
        / (pred_df["raw_score"].std() + 1e-8)
    )
    # ----------------------------------------------------------------------

    weights = build_portfolio(
        pred_df.set_index("stock_code")["score_z"],
        top_k=top_k,
        weighting=weighting,
        blend_alpha=blend_alpha,
    )
    submission = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    return submission, ic


# ============================================================
# 滚动回测
# ============================================================
def run_self_test(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    top_k: int,
    windows: int,
    weighting: str = DEFAULT_WEIGHTING,
    blend_alpha: float = DEFAULT_BLEND_ALPHA,
    hold_days: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """非重叠滚动回测, step = hold_days, 每个窗口完全独立."""
    trading_dates = np.sort(panel["date"].unique())
    start_idx = max(
        FORWARD_HORIZON + VAL_DAYS + EMBARGO_DAYS + 20,
        len(trading_dates) - hold_days * windows - hold_days,
    )
    rows = []

    for pred_idx in range(start_idx, len(trading_dates) - hold_days, hold_days):
        as_of = pd.Timestamp(trading_dates[pred_idx])
        submission, ic = generate_submission(
            panel, as_of=as_of, top_k=top_k,
            weighting=weighting, blend_alpha=blend_alpha,
            verbose=verbose,
        )
        weights = submission.set_index("stock_code")["weight"]

        window_start = pd.Timestamp(trading_dates[pred_idx + 1])
        window_end = pd.Timestamp(
            trading_dates[min(pred_idx + hold_days, len(trading_dates) - 1)]
        )

        realized = score_window(weights, prices, index_df, window_start, window_end)
        rows.append({
            "as_of": as_of.date().isoformat(),
            "start": realized["start"],
            "end": realized["end"],
            "validation_rank_ic": ic,
            "portfolio_return": realized["portfolio_return"],
            "benchmark_return": realized["benchmark_return"],
            "excess_return": realized["excess_return"],
        })
    return pd.DataFrame(rows)


def print_metrics_explanation():
    print("\n" + "=" * 50)
    print("【评测指标含义解释】")
    print("1. Rank IC: 预测排名与实际收益排名的相关性. > 0.02 即为有效信号.")
    print("2. 投资组合收益: 策略组合在 5 天窗口期内的收益.")
    print("3. 基准收益: 中证500 同期收益.")
    print("4. 超额收益 (Alpha): 组合收益 - 基准收益.")
    print("5. 胜率 (Win Rate): 跑赢中证500 的窗口占比.")
    print("6. IR (Info Ratio): 年化超额收益 / 超额波动. > 0.5 算不错.")
    print("=" * 50 + "\n")


# ============================================================
# 主入口
# ============================================================
def main():
    global TRAIN_HALF_LIFE_DAYS
    _set_global_seed()

    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--weighting", choices=["rank", "equal", "blend", "sqrt_rank"],
                   default=DEFAULT_WEIGHTING)
    p.add_argument("--blend-alpha", type=float, default=DEFAULT_BLEND_ALPHA)
    p.add_argument("--train-half-life", type=int, default=TRAIN_HALF_LIFE_DAYS,
                   help="time-decay half life for training samples; <=0 disables weighting")
    p.add_argument("--out", default="submissions/Jinghan Wu_week2.csv")
    p.add_argument("--windows", type=int, default=30,
                   help="回测非重叠窗口数, 30+ 才有统计意义")
    p.add_argument("--skip-backtest", action="store_true",
                   help="跳过回测, 仅生成最新提交文件")
    args = p.parse_args()
    TRAIN_HALF_LIFE_DAYS = args.train_half_life

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])

    print(">> 构建特征面板 (含 winsorize + 截面 z-score 中性化)...")
    panel = build_features(prices)

    if not args.skip_backtest:
        print(f">> 滚动回测: {args.windows} 个非重叠 5 天窗口...")
        results = run_self_test(
            panel, prices, index_df,
            top_k=args.top_k, windows=args.windows,
            weighting=args.weighting, blend_alpha=args.blend_alpha,
            hold_days=5,  # 修改为 5 天
        )
        results.to_csv("backtest_results_5days.csv", index=False)

        avg_ic = results["validation_rank_ic"].mean()
        avg_port_ret = results["portfolio_return"].mean()
        avg_bench_ret = results["benchmark_return"].mean()
        avg_excess_ret = results["excess_return"].mean()
        win_rate = (results["excess_return"] > 0).mean()
        excess_std = results["excess_return"].std()
        ir = avg_excess_ret / (excess_std + 1e-8) * np.sqrt(252 / 5)  # 年化 IR

        print_metrics_explanation()
        print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\n" + "*" * 40)
        print("【回测表现】")
        print(f"窗口数                : {len(results)}")
        print(f"平均 Validation Rank IC: {avg_ic:+.4f}")
        print(f"平均 5 天组合收益     : {avg_port_ret:+.4%}")
        print(f"平均 5 天基准收益     : {avg_bench_ret:+.4%}")
        print(f"平均 5 天超额 (Alpha) : {avg_excess_ret:+.4%}")
        print(f"超额收益标准差        : {excess_std:.4%}")
        print(f"胜率 (Win Rate)       : {win_rate:.1%}")
        print(f"年化 Information Ratio: {ir:+.3f}")
        print("*" * 40 + "\n")

    print(">> 训练最终模型并生成提交文件...")
    latest_date = pd.Timestamp(panel["date"].max())
    submission, final_ic = generate_submission(
        panel, as_of=latest_date, top_k=args.top_k,
        weighting=args.weighting, blend_alpha=args.blend_alpha,
        verbose=True,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(f">> 提交文件已写入: {args.out} (val_IC={final_ic:+.4f})")
    print(">> 请运行 validate_submission.py 校验后再提交!")


if __name__ == "__main__":
    main()
