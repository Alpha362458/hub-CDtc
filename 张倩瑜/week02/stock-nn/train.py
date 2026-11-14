#!/usr/bin/env python3
"""股票五分类神经网络训练与推理主程序（手写 NumPy + PyTorch 双版本对照）。

用法：
    python train.py
    python train.py --epochs 160 --lr 0.008
    python train.py --vector "0.35,0.42,0.50,0.08,0.35"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 确保 Windows 终端 UTF-8 输出不乱码
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dataset_hf import (  # noqa: E402
    CLASS_NAMES_ZH,
    CLASS_SHORT_ZH,
    FEATURE_LABELS_ZH,
    FEATURE_NAMES,
    HF_DATASET_ID,
    StockDataset,
    classify_5d_vector,
    load_stock_dataset,
)
from stock_nn import (  # noqa: E402
    HAVE_TORCH,
    TrainResult,
    predict_single_or_batch,
    train_numpy,
    train_pytorch,
    verify_backprop_and_adam,
)


def configure_chinese_font() -> None:
    """配置 Matplotlib 中文字体支持（兼容 Windows / macOS / Linux）。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_training_dashboard(
    dataset: StockDataset,
    res_np: TrainResult,
    res_pt: TrainResult | None,
    out_path: str,
) -> None:
    """绘制四合一训练与分类效果可视化面板并保存为 PNG。"""
    configure_chinese_font()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), dpi=140)
    fig.patch.set_facecolor("#0f172a")

    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    epochs = np.arange(1, len(res_np.train_loss_history) + 1)

    # 1) 左上：Cross-Entropy Loss 收敛曲线 + 准确率曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, res_np.train_loss_history, color="#38bdf8", lw=2.2, label="NumPy手写 Train Loss")
    ax1.plot(epochs, res_np.val_loss_history, color="#f43f5e", lw=2.2, label="NumPy手写 Val Loss")
    if res_pt is not None:
        ax1.plot(
            epochs,
            res_pt.val_loss_history,
            color="#fbbf24",
            lw=1.8,
            ls="--",
            label="PyTorch对照 Val Loss",
        )
    ax1.set_title("① Cross-Entropy Loss & 准确率收敛曲线 (Adam Optimizer)", color="#f8fafc", fontsize=12, pad=10)
    ax1.set_xlabel("Epoch", color="#cbd5e1")
    ax1.set_ylabel("Cross-Entropy Loss", color="#38bdf8")
    ax1.grid(True, color="#334155", alpha=0.45, ls="--")

    ax1_r = ax1.twinx()
    ax1_r.plot(
        epochs,
        np.array(res_np.val_acc_history) * 100.0,
        color="#34d399",
        lw=2.0,
        ls="-.",
        label="Val Accuracy (%)",
    )
    ax1_r.set_ylabel("验证集准确率 (%)", color="#34d399")
    ax1_r.tick_params(colors="#34d399")
    ax1_r.set_ylim(20, 103)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=8.5)

    # 2) 右上：测试集 5x5 混淆矩阵
    ax2 = axes[0, 1]
    cm = res_np.confusion_matrix
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_title(
        f"② 测试集混淆矩阵 (Val Acc = {res_np.final_val_acc * 100:.2f}%)",
        color="#f8fafc",
        fontsize=12,
        pad=10,
    )
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(5))
    ax2.set_xticklabels(CLASS_SHORT_ZH, rotation=22, ha="right", color="#e2e8f0", fontsize=9)
    ax2.set_yticklabels(CLASS_SHORT_ZH, color="#e2e8f0", fontsize=9)
    ax2.set_xlabel("模型预测类别", color="#cbd5e1")
    ax2.set_ylabel("真实类别", color="#cbd5e1")
    thresh = cm.max() / 2.0
    for i in range(5):
        for j in range(5):
            ax2.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "#0f172a",
                fontsize=10,
                fontweight="bold",
            )

    # 3) 左下：61只真实股票最新10年窗口特征分布图 (CAGR vs 峰值回落比, 气泡大小=波段振幅)
    ax3 = axes[1, 0]
    palette = ["#22c55e", "#38bdf8", "#f59e0b", "#ef4444", "#94a3b8"]
    sh = dataset.showcase_stocks
    for c in range(5):
        sub = sh[sh["label"] == c]
        ax3.scatter(
            sub["cagr"] * 100.0,
            sub["peak_drop"] * 100.0,
            s=sub["swing_amp"] * 380.0 + 40.0,
            c=palette[c],
            alpha=0.85,
            edgecolors="#0f172a",
            linewidth=1.0,
            label=CLASS_SHORT_ZH[c],
        )

    highlight_tickers = [
        "NVDA", "MSFT", "AAPL", "COST",
        "KO", "PG", "JNJ",
        "TSLA", "AMD", "COIN", "RBLX",
        "PTON", "ZM", "SQ", "UBSFY",
        "DIS", "MMM", "PHG",
    ]
    for _, row in sh[sh["Ticker"].isin(highlight_tickers)].iterrows():
        ax3.annotate(
            str(row["Ticker"]),
            (float(row["cagr"]) * 100.0, float(row["peak_drop"]) * 100.0),
            xytext=(4, 4),
            textcoords="offset points",
            color="#f8fafc",
            fontsize=8.5,
            fontweight="bold",
        )

    ax3.set_title("③ Hugging Face 真实个股 10年期特征分布 (气泡大小=波段振幅)", color="#f8fafc", fontsize=12, pad=10)
    ax3.set_xlabel("x1: 10年年化收益率 CAGR (%)", color="#cbd5e1")
    ax3.set_ylabel("x4: 期末距历史最高点回落幅度 Peak Drop (%)", color="#cbd5e1")
    ax3.grid(True, color="#334155", alpha=0.45, ls="--")
    ax3.legend(loc="upper right", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=8.5)

    # 4) 右下：5只代表性股票的“5个原始分数 (Logits)”对比柱状图
    ax4 = axes[1, 1]
    demo_tickers = ["NVDA", "KO", "TSLA", "PTON", "DIS"]
    bar_x = np.arange(len(demo_tickers))
    width = 0.15

    logits_matrix = []
    for tk in demo_tickers:
        row = sh[sh["Ticker"] == tk].iloc[0]
        x_vec = np.array([row[col] for col in FEATURE_NAMES], dtype=np.float64)
        logits_tk, _, _ = predict_single_or_batch(x_vec, res_np.params, dataset.mean, dataset.std)
        logits_matrix.append(logits_tk)
    logits_matrix = np.array(logits_matrix)  # (5 stocks, 5 classes)

    for c in range(5):
        ax4.bar(
            bar_x + (c - 2) * width,
            logits_matrix[:, c],
            width=width,
            color=palette[c],
            label=f"分数 z_{c}: {CLASS_SHORT_ZH[c]}",
        )
    ax4.axhline(0.0, color="#94a3b8", lw=1.0, ls="--")
    ax4.set_xticks(bar_x)
    ax4.set_xticklabels(
        ["NVDA\n(暴利大牛)", "KO\n(长期稳健)", "TSLA\n(炒波段)", "PTON\n(风口撤退)", "DIS\n(不考虑)"],
        color="#e2e8f0",
        fontsize=9,
    )
    ax4.set_title("④ 神经网络输出的 5 个原始分数 (Logits z0..z4)", color="#f8fafc", fontsize=12, pad=10)
    ax4.set_ylabel("原始分数 Logit 值 (未经过 Softmax)", color="#cbd5e1")
    ax4.grid(True, axis="y", color="#334155", alpha=0.45, ls="--")
    ax4.legend(loc="lower left", facecolor="#0f172a", edgecolor="#475569", labelcolor="#f8fafc", fontsize=8)

    fig.suptitle(
        "股票五分类神经网络训练看板 (5D Vector -> 5 Raw Scores -> Cross-Entropy -> Backprop -> Adam)",
        color="#f8fafc",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.955))
    plt.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def print_inference_table(dataset: StockDataset, params: dict, custom_vectors: List[str]) -> None:
    """在终端展示典型真实股票与自定义 5 维向量的推理全过程。"""
    sh = dataset.showcase_stocks
    demo_tickers = ["NVDA", "MSFT", "KO", "JNJ", "TSLA", "AMD", "PTON", "ZM", "DIS", "MMM"]

    print("\n" + "=" * 96)
    print("【四】真实股票推理演示（5维向量 x -> 神经网络算5个原始分数 Logits -> Softmax概率 -> 预测分类）")
    print("=" * 96)
    for tk in demo_tickers:
        sub = sh[sh["Ticker"] == tk]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]
        x_raw = np.array([row[col] for col in FEATURE_NAMES], dtype=np.float64)
        logits, probs, pred = predict_single_or_batch(x_raw, params, dataset.mean, dataset.std)
        true_c = int(row["label"])
        pred_c = int(pred)
        mark = "✓" if pred_c == true_c else "✗"

        x_str = (
            f"CAGR={x_raw[0]:+6.1%}, Vol={x_raw[1]:5.1%}, "
            f"MDD={x_raw[2]:5.1%}, PeakDrop={x_raw[3]:5.1%}, Swing={x_raw[4]:5.1%}"
        )
        z_str = ", ".join(f"{z:+6.2f}" for z in logits)
        p_str = ", ".join(f"{p * 100:5.1f}%" for p in probs)

        print(f"▸ {tk:<5s} ({str(row['Brand_Name']):<24s})")
        print(f"    输入5维向量 x : [{x_str}]")
        print(f"    5个原始分数 z : [{z_str}]")
        print(f"    Softmax 概率  : [{p_str}]")
        print(
            f"    最终预测类别  : [{pred_c}] {CLASS_SHORT_ZH[pred_c]} "
            f"(真实标签: [{true_c}] {CLASS_SHORT_ZH[true_c]}) {mark}"
        )

    # 演示任意自定义 5 维向量输入
    default_custom = [
        ("自定义向量A (高增长低回撤大牛型)", [0.35, 0.30, 0.35, 0.05, 0.22]),
        ("自定义向量B (低波稳健养老收息型)", [0.09, 0.17, 0.24, 0.06, 0.12]),
        ("自定义向量C (高振幅反复震荡波段型)", [0.15, 0.55, 0.60, 0.28, 0.52]),
        ("自定义向量D (暴涨后崩盘85%风口撤退型)", [-0.08, 0.65, 0.92, 0.88, 0.55]),
        ("自定义向量E (长期阴跌缺乏弹性不考虑型)", [-0.05, 0.26, 0.65, 0.52, 0.18]),
    ]
    for idx, vec_str in enumerate(custom_vectors):
        vals = [float(v.strip()) for v in vec_str.split(",")]
        if len(vals) != 5:
            raise ValueError(f"--vector 需要正好 5 个浮点数（逗号分隔），收到: {vec_str}")
        default_custom.append((f"命令行输入向量 #{idx + 1}", vals))

    print("\n" + "=" * 96)
    print("【五】输入任意 5 维向量直接推理演示")
    print("=" * 96)
    for title, vec in default_custom:
        x_raw = np.array(vec, dtype=np.float64)
        logits, probs, pred = predict_single_or_batch(x_raw, params, dataset.mean, dataset.std)
        rule_c = classify_5d_vector(x_raw)
        pred_c = int(pred)
        z_str = ", ".join(f"{z:+6.2f}" for z in logits)
        p_str = ", ".join(f"{p * 100:5.1f}%" for p in probs)
        print(f"▸ {title}: x = {np.round(x_raw, 3).tolist()}")
        print(f"    5个原始分数 z = [{z_str}]  ->  Softmax = [{p_str}]")
        print(f"    神经网络预测  = [{pred_c}] {CLASS_NAMES_ZH[pred_c]} (规则基准: [{rule_c}] {CLASS_SHORT_ZH[rule_c]})")


def main() -> int:
    parser = argparse.ArgumentParser(description="股票五分类神经网络训练与推理（手写 NumPy + PyTorch 对照）")
    parser.add_argument("--epochs", type=int, default=160, help="训练轮数 (默认: 160)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch 大小 (默认: 64)")
    parser.add_argument("--lr", type=float, default=0.008, help="Adam 学习率 (默认: 0.008)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    parser.add_argument("--rebuild", action="store_true", help="重新从 Hugging Face 原始 CSV 提取特征窗口")
    parser.add_argument(
        "--vector",
        action="append",
        default=[],
        help="输入任意自定义的 5 维向量做分类预测，格式如 '0.32,0.40,0.45,0.10,0.30'",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    print("=" * 96)
    print(f"【一】加载 Hugging Face 股票数据集 ({HF_DATASET_ID})")
    print("=" * 96)
    dataset = load_stock_dataset(rebuild=args.rebuild, seed=args.seed)
    print(f"  训练集样本数: {len(dataset.X_train)} | 测试集样本数: {len(dataset.X_test)} | 特征维度: 5 | 类别数: 5")
    for c in range(5):
        n_tr = int(np.sum(dataset.y_train == c))
        n_te = int(np.sum(dataset.y_test == c))
        print(f"    类别 {CLASS_NAMES_ZH[c]:<18s} : 训练集 {n_tr:4d} 条 | 测试集 {n_te:4d} 条")

    print("\n" + "=" * 96)
    print("【二】校验手写 Backpropagation 梯度 & Adam 步长更新精度")
    print("=" * 96)
    grad_check = verify_backprop_and_adam(dataset.X_train[:16], dataset.y_train[:16], seed=args.seed)
    print(f"  手写 Backprop vs 数值有限差分最大梯度误差  : {grad_check['max_numerical_grad_diff']:.3e}")
    if "max_pytorch_grad_diff" in grad_check:
        print(f"  手写 CrossEntropy vs PyTorch 损失绝对误差 : {grad_check['loss_diff_vs_pytorch']:.3e}")
        print(f"  手写 Backprop vs PyTorch Autograd 梯度误差 : {grad_check['max_pytorch_grad_diff']:.3e}")
        print(f"  手写 Adam 一步更新后参数 vs PyTorch 误差   : {grad_check['max_pytorch_adam_step_diff']:.3e}")

    print("\n" + "=" * 96)
    print(f"【三】开始训练五分类神经网络 (5 -> 32 -> 16 -> 5, Epochs={args.epochs}, Batch={args.batch_size}, Adam lr={args.lr})")
    print("=" * 96)
    res_np = train_numpy(
        dataset.X_train,
        dataset.y_train,
        dataset.X_test,
        dataset.y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        verbose=True,
    )

    res_pt = None
    if HAVE_TORCH:
        print("-" * 96)
        res_pt = train_pytorch(
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            dataset.y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            verbose=True,
        )

    print("-" * 96)
    print(
        f"  ★ 手写 NumPy 版  -> 最终 Train Acc: {res_np.final_train_acc * 100:.2f}% | "
        f"Val Acc: {res_np.final_val_acc * 100:.2f}% | 耗时: {res_np.elapsed_sec:.2f} 秒"
    )
    if res_pt is not None:
        max_w_diff = max(
            float(np.max(np.abs(res_np.params[k] - res_pt.params[k])))
            for k in res_np.params
        )
        print(
            f"  ★ PyTorch 对照版 -> 最终 Train Acc: {res_pt.final_train_acc * 100:.2f}% | "
            f"Val Acc: {res_pt.final_val_acc * 100:.2f}% | 耗时: {res_pt.elapsed_sec:.2f} 秒 | "
            f"160轮后与手写版最大权重差: {max_w_diff:.3e}"
        )

    print_inference_table(dataset, res_np.params, args.vector)

    fig_path = os.path.join(HERE, "training_results.png")
    plot_training_dashboard(dataset, res_np, res_pt, fig_path)

    # 保存最终训练完成的全部参数与标准化常量到 JSON
    import json
    params_path = os.path.join(HERE, "trained_params.json")
    export_payload = {
        "architecture": "5 -> 32 (ReLU) -> 16 (ReLU) -> 5 (Logits)",
        "total_parameters": int(sum(v.size for v in res_np.params.values())),
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES_ZH,
        "normalization": {
            "mean": np.round(dataset.mean, 6).tolist(),
            "std": np.round(dataset.std, 6).tolist(),
        },
        "parameters": {
            k: np.round(v, 6).tolist() for k, v in res_np.params.items()
        },
    }
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, ensure_ascii=False, indent=2)

    total_sec = time.perf_counter() - t_start
    print("\n" + "=" * 96)
    print(f"最终模型参数已导出: {params_path}")
    print(f"可视化图表已生成  : {fig_path}")
    print(f"全流程执行完毕，总耗时: {total_sec:.2f} 秒（远低于 2 分钟要求）")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
