"""绘制 SFT 的 history.json；直接点击运行也可以，不需要重新训练。"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
DEFAULT_HISTORY = HERE / "sft_output" / "history.json"


def render_history(history_path=DEFAULT_HISTORY, output_dir=None):
    """输出 PNG、SVG 和原始指标 CSV，不修改训练记录或模型。"""
    # 无图形界面的服务器也能生成图片；字体缓存放在临时目录。
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "week5_sft_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter, ScalarFormatter

    history_path = Path(history_path)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    records = history.get("epochs", [])
    if not records:
        raise ValueError("记录中还没有完成的训练轮次，请至少训练一轮后再绘图")
    epochs = [r["epoch"] for r in records]
    if any(a >= b for a, b in zip(epochs, epochs[1:])):
        raise ValueError("history.json 中的 epoch 必须严格递增")
    for record in records:
        for key in ("answer_loss", "train_answer_loss"):
            if not math.isfinite(record[key]):
                raise ValueError(f"发现非有限指标：epoch={record['epoch']} {key}")
    baseline = history.get("baseline", {})
    output_dir = Path(output_dir) if output_dir else history_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 优先使用跨平台的中文字体；没有中文字体时自动改用英文标签。
    chinese = False
    font = "DejaVu Sans"
    for candidate in ("PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=candidate), fallback_to_default=False)
        except ValueError:
            continue
        font, chinese = candidate, True
        break
    def label(zh, en):
        return zh if chinese else en

    colors = {"train": "#2563EB", "val": "#E16A2D", "prefix": "#0D9488", "time": "#64748B"}
    style = {"font.family": font, "font.size": 10, "axes.titlesize": 13,
             "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
             "axes.edgecolor": "#CBD5E1", "text.color": "#172B4D",
             "axes.labelcolor": "#334155", "xtick.color": "#475569", "ytick.color": "#475569",
             "axes.unicode_minus": False, "svg.fonttype": "path"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.6))
        fig.subplots_adjust(left=0.075, right=0.96, bottom=0.15, top=0.84, wspace=0.27, hspace=0.44)
        fig.suptitle(label("SFT 训练效果", "SFT training metrics"), fontsize=23, fontweight="bold", y=0.97)
        subtitle = label("字符级 Transformer  |  第 0 轮代表微调前的模型", "Character-level Transformer  |  Epoch 0 is the pre-SFT baseline")
        fig.text(0.5, 0.91, subtitle, ha="center", fontsize=11, color="#64748B")
        best = min(records, key=lambda r: r["answer_loss"])
        for axis in axes.flat:
            axis.set_axisbelow(True)
            axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
            axis.set_xlabel(label("训练轮次（Epoch）", "Epoch"))
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

        # 训练 loss 没有第 0 轮数值，不把验证基线伪装成训练 loss。
        axis = axes[0, 0]
        val_x, val_y = list(epochs), [r["answer_loss"] for r in records]
        if baseline.get("answer_loss") is not None:
            val_x.insert(0, 0)
            val_y.insert(0, baseline["answer_loss"])
        axis.plot(epochs, [r["train_answer_loss"] for r in records], "o-", color=colors["train"],
                  label=label("训练答案 Loss", "Training answer loss"), markersize=4)
        axis.plot(val_x, val_y, "o-", color=colors["val"],
                  label=label("验证答案 Loss", "Validation answer loss"), markersize=4)
        axis.scatter([best["epoch"]], [best["answer_loss"]], s=90, facecolors="none",
                     edgecolors=colors["val"], linewidths=1.5, zorder=5)
        axis.annotate(f"{best['answer_loss']:.3f}", (best["epoch"], best["answer_loss"]),
                      xytext=(-8, 13), textcoords="offset points", ha="right", color=colors["val"])
        axis.set_title(label("答案损失：越低越好", "Answer loss: lower is better"), loc="left", pad=12)
        axis.set_ylabel("Cross-entropy")
        axis.set_ylim(bottom=0, top=max(val_y + [r["train_answer_loss"] for r in records]) * 1.14)
        axis.legend(frameon=False, fontsize=9)

        axis = axes[0, 1]
        ppl_rows = [(r["epoch"], r.get("answer_ppl")) for r in records]
        if baseline.get("answer_ppl") is not None:
            ppl_rows.insert(0, (0, baseline["answer_ppl"]))
        ppl_rows = [(x, y) for x, y in ppl_rows if y is not None and math.isfinite(y) and y > 0]
        if ppl_rows:
            x, y = zip(*ppl_rows)
            axis.plot(x, y, "o-", color=colors["val"], markersize=4)
            axis.set_yscale("log")
            axis.yaxis.set_major_formatter(ScalarFormatter())
            for index, offset in ((0, (8, -17)), (-1, (-8, 12))):
                axis.annotate(f"{y[index]:.2f}", (x[index], y[index]), xytext=offset,
                              textcoords="offset points", ha="left" if index == 0 else "right")
        axis.set_title(label("验证集 PPL：对数坐标，越低越好", "Validation PPL: log scale, lower is better"), loc="left", pad=12)
        axis.set_ylabel("Perplexity (log scale)")

        axis = axes[1, 0]
        prefix_rows = [(r["epoch"], r.get("prefix_accuracy")) for r in records]
        if baseline.get("prefix_accuracy") is not None:
            prefix_rows.insert(0, (0, baseline["prefix_accuracy"]))
        prefix_rows = [(x, p) for x, p in prefix_rows if p is not None]
        if prefix_rows:
            axis.plot([x for x, _ in prefix_rows], [p["rate"] * 100 for _, p in prefix_rows],
                      "o-", color=colors["prefix"], markersize=4)
            last = prefix_rows[-1][1]
            axis.text(0.97, 0.14, label(f"最后一轮：{last['correct']}/{last['total']} 条符合",
                                       f"Final epoch: {last['correct']}/{last['total']} match"),
                      transform=axis.transAxes, ha="right", color=colors["prefix"], fontsize=11)
        else:
            axis.text(0.5, 0.5, label("未启用前缀评估", "Prefix evaluation disabled"),
                      ha="center", transform=axis.transAxes)
        axis.set_title(label("回答前缀符合率：不代表内容正确率", "Answer prefix compliance: not answer correctness"), loc="left", pad=12)
        axis.set_ylim(-5, 110)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        axis.set_ylabel(label("前缀符合率", "Prefix compliance"))

        axis = axes[1, 1]
        durations = [r.get("elapsed_seconds", 0) for r in records]
        axis.bar(epochs, durations, width=0.65, color=colors["time"], alpha=0.85)
        axis.set_title(label("每轮耗时：训练与轮末验证", "Time per epoch: training and validation"), loc="left", pad=12)
        axis.set_ylabel(label("秒", "Seconds"))
        axis.set_ylim(0, max(max(durations), 1) * 1.23)
        axis.text(0.97, 0.9, label(f"记录总计 {sum(durations):.1f} 秒", f"Recorded total: {sum(durations):.1f} s"),
                  transform=axis.transAxes, ha="right", fontsize=10, color=colors["time"])

        note = label("Loss / PPL 仅统计答案与 EOS；训练 Loss 是轮内各批次的平均，验证 Loss 在轮末计算。\n"
                     "前缀符合率使用 history.json 中每轮记录的抽查范围；PPL 不是回答错误率。",
                     "Loss / PPL cover answer tokens and EOS. Training loss averages batches during updates; validation is at epoch end.\n"
                     "Prefix compliance uses the sample counts recorded in history.json. PPL is not an answer error rate.")
        fig.text(0.075, 0.045, note, ha="left", va="bottom", color="#64748B", fontsize=9, linespacing=1.6)
        png, svg = output_dir / "training_metrics.png", output_dir / "training_metrics.svg"
        fig.savefig(png, dpi=180, facecolor="white")
        fig.savefig(svg, facecolor="white")
        plt.close(fig)

    csv_path = output_dir / "training_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["epoch", "global_step", "train_answer_loss", "answer_loss", "answer_ppl",
                  "prefix_correct", "prefix_total", "prefix_rate", "elapsed_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        entries = ([{"epoch": 0, "global_step": 0, **baseline}] if baseline else []) + records
        for record in entries:
            row = {key: record.get(key) for key in fields}
            prefix = record.get("prefix_accuracy") or {}
            row.update({"prefix_correct": prefix.get("correct"), "prefix_total": prefix.get("total"),
                        "prefix_rate": prefix.get("rate")})
            writer.writerow(row)
    return [png, svg, csv_path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        paths = render_history(args.history, args.output_dir)
    except ModuleNotFoundError as error:
        raise SystemExit("缺少绘图依赖，请运行：python -m pip install matplotlib") from error
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"绘图失败：{error}") from error
    for path in paths:
        print(f"已生成：{path.resolve()}")


if __name__ == "__main__":
    main()
