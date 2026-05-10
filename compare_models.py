"""
compare_models.py — Baseline vs Advanced 历史表现对比

在同一组非重叠回测窗口上分别跑:
  - baseline (baseline_xgboost.py + features.py)
  - advanced (competition_advanced.py + features_updated.py)
然后输出:
  - 上图: 三条累计收益曲线 (baseline / advanced / CSI500)
  - 下图: 每个回测窗口的收益柱状对比 (3 根柱子 / 窗口)
  - 控制台: 累计收益、均值、std、Sharpe 近似、对中证500 胜率

使用:
  python compare_models.py                          # 默认 15 个窗口, hold 5 天
  python compare_models.py --windows 30
  python compare_models.py --windows 20 --hold-days 5 --out-png out/compare.png

注意:
  - 假定项目根目录有 features.py / features_updated.py / baseline_xgboost.py
    / competition_advanced.py / score_submission.py
  - 数据默认在 ./data/prices.parquet 和 ./data/index.parquet
  - 总耗时主要来自 advanced 端 (5 seeds × 3 configs × N 窗口)
"""
from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- baseline 端 ----------
import features as feat_base
from baseline_xgboost import (
    train_model as baseline_train_model,
    build_portfolio as baseline_build_portfolio,
    rank_ic as baseline_rank_ic,
    VAL_DAYS as BASE_VAL_DAYS,
    EMBARGO_DAYS as BASE_EMBARGO_DAYS,
    DEFAULT_TOP_K as BASE_DEFAULT_TOP_K,
)

# ---------- advanced 端 ----------
from competition_advanced import (
    generate_submission as advanced_generate_submission,
    DEFAULT_TOP_K as ADV_DEFAULT_TOP_K,
    DEFAULT_WEIGHTING,
    DEFAULT_BLEND_ALPHA,
    EMBARGO_DAYS as ADV_EMBARGO_DAYS,
    VAL_DAYS as ADV_VAL_DAYS,
    _set_global_seed,
)
from features_updated import (
    FORWARD_HORIZON as ADV_FORWARD_HORIZON,
    build_features as adv_build_features,
)

# 项目里已有的回测打分函数, advanced 也是用这个
from score_submission import score_window

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = Path(__file__).parent / "data"


# ============================================================
# Baseline 端: 把 baseline_xgboost.main 里的逻辑抽成函数
# ============================================================
def baseline_generate_submission(
    panel_base: pd.DataFrame,
    as_of: pd.Timestamp,
    top_k: int,
) -> tuple[pd.DataFrame, float]:
    """复制 baseline_xgboost.main 里训练 + 预测 + 组合的流程."""
    trading_dates = np.sort(panel_base["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    cutoff_idx = max(0, as_of_idx - feat_base.FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool = feat_base.training_frame(panel_base, max_date=train_cutoff)

    all_dates = np.sort(train_pool["date"].unique())
    if len(all_dates) < BASE_VAL_DAYS + BASE_EMBARGO_DAYS + 20:
        raise RuntimeError(f"baseline 训练样本不足: {len(all_dates)} 天")

    val_start = pd.Timestamp(all_dates[-BASE_VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(BASE_VAL_DAYS + BASE_EMBARGO_DAYS + 1)])
    train_df = train_pool[train_pool["date"] <= train_end]
    val_df = train_pool[train_pool["date"] >= val_start]

    model = baseline_train_model(train_df, val_df)
    val_pred = model.predict(val_df[feat_base.FEATURE_COLUMNS])
    ic = baseline_rank_ic(
        val_df[feat_base.TARGET_COLUMN].to_numpy(),
        val_pred,
        val_df["date"].to_numpy(),
    )

    pred_df = feat_base.prediction_frame(panel_base, as_of=as_of)
    if pred_df.empty:
        raise RuntimeError(f"baseline: as_of={as_of.date()} 无可打分股票")
    pred_df = pred_df.assign(score=model.predict(pred_df[feat_base.FEATURE_COLUMNS]))
    scores = pred_df.set_index("stock_code")["score"]
    weights = baseline_build_portfolio(scores, top_k=top_k)
    submission = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    return submission, ic


# ============================================================
# 主回测循环: 在同一组窗口上跑两个模型
# ============================================================
def run_comparison(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    windows: int,
    hold_days: int,
    base_top_k: int,
    adv_top_k: int,
) -> pd.DataFrame:
    print(">> 构建 baseline 特征面板 (features.py)...")
    panel_base = feat_base.build_features(prices)
    print(">> 构建 advanced 特征面板 (features_updated.py)...")
    panel_adv = adv_build_features(prices)

    # 用 advanced 端的预热长度决定回测起点 (它需要的历史更长)
    trading_dates = np.sort(panel_adv["date"].unique())
    min_warmup = ADV_FORWARD_HORIZON + ADV_VAL_DAYS + ADV_EMBARGO_DAYS + 20
    start_idx = max(
        min_warmup,
        len(trading_dates) - hold_days * windows - hold_days,
    )
    pred_indices = list(range(start_idx, len(trading_dates) - hold_days, hold_days))
    print(f">> 计划运行 {len(pred_indices)} 个非重叠窗口, hold_days={hold_days}\n")

    rows = []
    for i, pred_idx in enumerate(pred_indices, 1):
        as_of = pd.Timestamp(trading_dates[pred_idx])
        window_start = pd.Timestamp(trading_dates[pred_idx + 1])
        window_end = pd.Timestamp(
            trading_dates[min(pred_idx + hold_days, len(trading_dates) - 1)]
        )
        print(f"[{i}/{len(pred_indices)}] as_of={as_of.date()}  "
              f"hold {window_start.date()} -> {window_end.date()}")

        # ---- Baseline ----
        try:
            base_sub, base_ic = baseline_generate_submission(panel_base, as_of, base_top_k)
            base_w = base_sub.set_index("stock_code")["weight"]
            base_scored = score_window(base_w, prices, index_df, window_start, window_end)
            base_ret = base_scored["portfolio_return"]
            bench_ret = base_scored["benchmark_return"]
        except Exception as e:
            print(f"   baseline 失败: {e}")
            base_ret, base_ic, bench_ret = np.nan, np.nan, np.nan

        # ---- Advanced ----
        try:
            adv_sub, adv_ic = advanced_generate_submission(
                panel_adv, as_of=as_of, top_k=adv_top_k,
                weighting=DEFAULT_WEIGHTING, blend_alpha=DEFAULT_BLEND_ALPHA,
                verbose=False,
            )
            adv_w = adv_sub.set_index("stock_code")["weight"]
            adv_scored = score_window(adv_w, prices, index_df, window_start, window_end)
            adv_ret = adv_scored["portfolio_return"]
            if np.isnan(bench_ret):  # baseline 失败时还能拿到 bench
                bench_ret = adv_scored["benchmark_return"]
        except Exception as e:
            print(f"   advanced 失败: {e}")
            adv_ret, adv_ic = np.nan, np.nan

        print(f"   baseline={base_ret:+.4%}   advanced={adv_ret:+.4%}   "
              f"csi500={bench_ret:+.4%}   "
              f"(IC base={base_ic:+.3f} adv={adv_ic:+.3f})")

        rows.append({
            "as_of": as_of,
            "start": window_start,
            "end": window_end,
            "baseline_return": base_ret,
            "advanced_return": adv_ret,
            "csi500_return": bench_ret,
            "baseline_ic": base_ic,
            "advanced_ic": adv_ic,
        })
    return pd.DataFrame(rows)


# ============================================================
# 汇总打印
# ============================================================
def print_summary(results: pd.DataFrame) -> None:
    res = results.dropna(subset=["baseline_return", "advanced_return", "csi500_return"])
    if res.empty:
        print(">> 没有有效窗口, 跳过 summary")
        return
    n = len(res)
    avg_days = max((res["end"] - res["start"]).dt.days.mean(), 1.0)
    ann_factor = np.sqrt(252.0 / avg_days)

    def stats(col: str) -> dict:
        r = res[col]
        return dict(
            cum=(1 + r).prod() - 1,
            avg=r.mean(),
            std=r.std(),
            sharpe=r.mean() / (r.std() + 1e-8) * ann_factor,
        )

    print("\n" + "=" * 64)
    print(f"【模型对比汇总】基于 {n} 个非重叠回测窗口")
    print("=" * 64)
    for label, col in [("Baseline", "baseline_return"),
                       ("Advanced", "advanced_return"),
                       ("CSI500  ", "csi500_return")]:
        s = stats(col)
        print(f"{label}  累计={s['cum']:+8.2%}  均值={s['avg']:+7.3%}  "
              f"std={s['std']:6.3%}  ann_Sharpe≈{s['sharpe']:+5.2f}")

    print("-" * 64)
    excess_base = res["baseline_return"] - res["csi500_return"]
    excess_adv = res["advanced_return"] - res["csi500_return"]
    print(f"Baseline alpha 均值={excess_base.mean():+.4%}  "
          f"胜率={(excess_base > 0).mean():.1%}")
    print(f"Advanced alpha 均值={excess_adv.mean():+.4%}  "
          f"胜率={(excess_adv > 0).mean():.1%}")
    print(f"Advanced 跑赢 Baseline 的窗口占比: "
          f"{(res['advanced_return'] > res['baseline_return']).mean():.1%}")
    print("=" * 64)


# ============================================================
# 画图: 上累计 + 下柱状
# ============================================================
def plot_comparison(results: pd.DataFrame, out_path: str = "model_comparison.png") -> None:
    res = results.dropna(
        subset=["baseline_return", "advanced_return", "csi500_return"]
    ).sort_values("as_of").reset_index(drop=True)
    if res.empty:
        print(">> 无有效数据可绘图")
        return

    # 复利累计
    res["base_cum"] = (1 + res["baseline_return"]).cumprod() - 1
    res["adv_cum"] = (1 + res["advanced_return"]).cumprod() - 1
    res["bench_cum"] = (1 + res["csi500_return"]).cumprod() - 1

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 9),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # ---------- 上图: 累计收益 ----------
    x = res["end"]
    ax1.plot(x, res["base_cum"] * 100, label="Baseline (XGBoost)",
             color="#1f77b4", lw=2.0, marker="o", markersize=4.5)
    ax1.plot(x, res["adv_cum"] * 100, label="Advanced (Ensemble)",
             color="#d62728", lw=2.2, marker="s", markersize=4.5)
    ax1.plot(x, res["bench_cum"] * 100, label="CSI500 Benchmark",
             color="#555555", lw=1.6, ls="--", marker="^", markersize=4)
    ax1.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax1.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax1.set_title("Cumulative Performance: Baseline vs Advanced vs CSI500",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="best", fontsize=10, framealpha=0.9)
    ax1.grid(alpha=0.3)

    # 在曲线末端标注最终累计收益
    for cum_col, color in [("base_cum", "#1f77b4"),
                           ("adv_cum", "#d62728"),
                           ("bench_cum", "#555555")]:
        final_x = x.iloc[-1]
        final_y = res[cum_col].iloc[-1] * 100
        ax1.annotate(f"{final_y:+.1f}%", xy=(final_x, final_y),
                     xytext=(6, 0), textcoords="offset points",
                     color=color, fontsize=9, fontweight="bold",
                     va="center")

    # ---------- 下图: 每窗口柱状 ----------
    n = len(res)
    idx = np.arange(n)
    width = 0.27
    ax2.bar(idx - width, res["baseline_return"] * 100, width,
            label="Baseline", color="#1f77b4", alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax2.bar(idx,         res["advanced_return"] * 100, width,
            label="Advanced", color="#d62728", alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax2.bar(idx + width, res["csi500_return"] * 100, width,
            label="CSI500", color="#888888", alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color="black", lw=0.7)
    ax2.set_xlabel("Backtest Window (as_of date)", fontsize=11)
    ax2.set_ylabel("Window Return (%)", fontsize=11)
    ax2.set_title("Per-Window Return Comparison",
                  fontsize=13, fontweight="bold")
    ax2.set_xticks(idx)
    # 窗口数多时只显示部分日期, 避免拥挤
    step = max(1, n // 20)
    labels = [d.strftime("%Y-%m-%d") if i % step == 0 else ""
              for i, d in enumerate(res["as_of"])]
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.legend(loc="best", fontsize=10, framealpha=0.9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f">> 图已保存: {out_path}")
    try:
        plt.show()
    except Exception:
        pass


# ============================================================
# 主入口
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--windows", type=int, default=15,
                   help="非重叠回测窗口数 (默认 15, 越多越慢)")
    p.add_argument("--hold-days", type=int, default=5,
                   help="每个窗口的持仓天数")
    p.add_argument("--base-top-k", type=int, default=BASE_DEFAULT_TOP_K)
    p.add_argument("--adv-top-k", type=int, default=ADV_DEFAULT_TOP_K)
    p.add_argument("--out-csv", default="model_comparison.csv")
    p.add_argument("--out-png", default="model_comparison.png")
    args = p.parse_args()

    # 全局随机源, 让两边都尽量可复现
    _set_global_seed(42)
    random.seed(42)
    np.random.seed(42)

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    print(f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
          f"dates {prices['date'].min().date()} -> {prices['date'].max().date()}")

    print(f">> Loading {args.index}")
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])

    results = run_comparison(
        prices, index_df,
        windows=args.windows,
        hold_days=args.hold_days,
        base_top_k=args.base_top_k,
        adv_top_k=args.adv_top_k,
    )
    results.to_csv(args.out_csv, index=False)
    print(f"\n>> 回测明细已写入: {args.out_csv}")

    print_summary(results)
    plot_comparison(results, out_path=args.out_png)


if __name__ == "__main__":
    main()