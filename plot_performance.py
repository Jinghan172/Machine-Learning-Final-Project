"""
plot_performance.py
Reads the backtest results and plots BOTH Cumulative Return and Per-Window (5-day) Return.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    try:
        results = pd.read_csv("backtest_results_5days.csv")
    except FileNotFoundError:
        print("未找到 backtest_results_5days.csv！请先运行 python competition_advanced.py")
        return

    results["date"] = pd.to_datetime(results["end"])
    results = results.sort_values("date")

    # 计算累计收益率
    results["Cumulative Portfolio"] = (1 + results["portfolio_return"]).cumprod() - 1
    results["Cumulative Benchmark"] = (1 + results["benchmark_return"]).cumprod() - 1

    # 创建上下两部分的图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1.5, 1]})
    
    # --- 上图：累计收益曲线 ---
    ax1.plot(results["date"], results["Cumulative Portfolio"] * 100, 
             label="XGBoost+LGBM Ensemble", color='tab:red', linewidth=2.5)
    ax1.plot(results["date"], results["Cumulative Benchmark"] * 100, 
             label="CSI500 Benchmark", color='tab:blue', linewidth=2.5)
    
    ax1.fill_between(results["date"], 
                     results["Cumulative Benchmark"] * 100, 
                     results["Cumulative Portfolio"] * 100, 
                     where=(results["Cumulative Portfolio"] > results["Cumulative Benchmark"]), 
                     interpolate=True, color='tab:red', alpha=0.1, label="Alpha (Excess Return)")

    ax1.set_title("Cumulative Performance (5-Day Step Walk-Forward Backtest)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 下图：每期（5天）独立收益率柱状图 ---
    x_indices = np.arange(len(results))
    bar_width = 0.35

    # 绘制投资组合和基准的柱子
    ax2.bar(x_indices - bar_width/2, results["portfolio_return"] * 100, 
            bar_width, label='Portfolio 5-Day Return', color='tab:red', alpha=0.8)
    ax2.bar(x_indices + bar_width/2, results["benchmark_return"] * 100, 
            bar_width, label='CSI500 5-Day Return', color='tab:blue', alpha=0.8)

    # 用折线在柱状图上标出“超额收益”
    ax2.plot(x_indices, results["excess_return"] * 100, 
             color='black', marker='o', linestyle='-', linewidth=1.5, markersize=4, label='Excess Return')
    ax2.axhline(0, color='gray', linestyle='-', linewidth=1)

    # 美化X轴标签，只显示部分日期避免拥挤
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([d.strftime('%m-%d') for d in results["date"]], rotation=45, fontsize=9)
    # 如果期数太多，X轴只间隔显示
    for i, label in enumerate(ax2.xaxis.get_ticklabels()):
        if i % 3 != 0: label.set_visible(False)

    ax2.set_title("Per-Window (5-Day) Independent Returns", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Return (%)", fontsize=12)
    ax2.legend(loc="upper left", fontsize=11)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    # 保存并显示
    output_img = "performance_plot_detailed.png"
    plt.savefig(output_img, dpi=300)
    print(f"\n>> 图表已保存为 {output_img}")
    plt.show()

if __name__ == "__main__":
    main()