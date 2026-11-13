"""由实际结果生成作业 README 和其中引用的训练曲线。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

from common import ROOT, read_json


def build_readme():
    data = read_json(ROOT / "results" / "data_stats.json")
    training = read_json(ROOT / "results" / "training.json")
    history = read_json(ROOT / "results" / "history.json")
    metrics = read_json(ROOT / "results" / "metrics.json")
    generations = read_json(ROOT / "results" / "generations.json")["samples"]
    cfg = training["config"]
    test = metrics["transformer"]["test"]
    baseline = metrics["baselines"]["splits"]["test"]
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#FAFAF7", "axes.facecolor": "white", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    epochs = [h["epoch"] for h in history]
    axes[0].plot(epochs, [h["train_nll"] for h in history], marker="o", color="#167D8D", label="训练（Dropout 开启）")
    axes[0].plot(epochs, [h["val_nll"] for h in history], marker="s", color="#D98336", label="验证（Dropout 关闭）")
    axes[0].axvline(training["best_epoch"], color="#56616B", linestyle="--", label=f"保留第 {training['best_epoch']} 轮")
    axes[0].set(xlabel="Epoch", ylabel="平均负对数似然（NLL）", title="Transformer 训练与验证损失")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].grid(alpha=.25)
    axes[0].legend(frameon=False)
    values = [baseline["bigram"]["perplexity"], test["perplexity"]]
    bars = axes[1].bar(["Bigram", "Transformer"], values, width=.5, color=["#D98336", "#167D8D"])
    axes[1].bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=7, fontsize=12)
    axes[1].set(ylim=(0, max(values)*1.22), ylabel="困惑度（PPL）", title="独立测试集 PPL：越低越好")
    axes[1].set_axisbelow(True)
    axes[1].grid(axis="y", alpha=.25)
    fig.savefig(ROOT / "results" / "training_curve.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    split_rows = "\n".join(f"| {name} | {data['splits'][key]['sentences']:,} | {data['splits'][key]['characters']:,} | {data['splits'][key]['predicted_tokens']:,} | {data['splits'][key]['unknown_rate']:.3%} |" for key, name in (("train","训练"),("val","验证"),("test","测试")))
    score_rows = "\n".join(f"| {name} | {metrics['transformer'][key]['nll']:.4f} | {metrics['transformer'][key]['perplexity']:.2f} | {metrics['transformer'][key]['accuracy']:.2%} |" for key,name in (("train","训练（评估模式）"),("val","验证"),("test","测试")))
    baseline_rows = "\n".join(f"| {name} | {baseline[key]['perplexity']:.2f} | {baseline[key]['accuracy']:.2%} |" for key,name in (("unigram","Unigram + 加性平滑"),("bigram","Bigram + 加性平滑")))
    examples = "\n".join(f"| {s['prompt']} | {s['continuation'].replace('|', '｜')} |" for s in generations[:-1])
    reduction = 1 - test["perplexity"] / baseline["bigram"]["perplexity"]
    readme = f'''# 第四周作业：训练基于 Transformer 的语言模型

姓名：周树晖　｜　周次：week04

使用第四周随课中文语料，从随机初始化开始训练字符级自回归 Transformer，学习根据前文预测下一个字符。最终模型在独立测试集上的 **PPL 为 {test['perplexity']:.2f}，下一字符准确率为 {test['accuracy']:.2%}**；同字表、同划分的 Bigram 基线 PPL 为 {baseline['bigram']['perplexity']:.2f}。

## 一、课程内容与任务

课件讲解了 N-gram、RNN/LSTM、BERT，以及 Transformer 的位置编码、多头自注意力、前馈网络、残差连接和 LayerNorm。本作业沿用课程 `language_model.py` 的字符级下一步预测任务，将循环网络替换为 Transformer。

模型使用单向因果注意力，预测位置 t 的下一个字符时只能访问 0…t 的输入。BERT 的 MLM 使用双向上下文和随机掩码；本项目训练的是用于续写的自回归语言模型，未加载现成 BERT/GPT 权重，也未调用外部生成 API。

## 二、语料与数据设计

语料来自第四周文件夹中的 `循环神经网络语言模型/corpus_general.txt`，原文件复制为 [data/corpus.txt](data/corpus.txt)。按通用换行读取后共 {data['raw_characters']:,} 个字符、{data['raw_lines']:,} 行，内容主要是中文问题、口语及业务对话短句。

先逐行 NFKC 归一化、合并行内空白、去除首尾空白，再去掉空行和完全相同的句子。共去除 {data['duplicates_removed']:,} 条重复记录，保留 **{data['unique_lines']:,} 条唯一句子**。用随机种子 {cfg['seed']} 打乱，按 90% / 5% / 5% 划分。三个划分的完整归一化句子交集为 0；此检查不排除近似句或同义改写。

| 划分 | 句子数 | 编码字符数 | 计分目标数 | UNK 比例 |
| --- | ---: | ---: | ---: | ---: |
{split_rows}

字表只由训练集统计，包含 `{cfg['vocab_size']:,}` 项：`<PAD>=0`、`<UNK>=1`，以及按频次选取的 {cfg['vocab_size']-2:,} 个字符。保留标点、数字、空格和换行；低频字、训练外字符映射到 UNK，不直接删除原文中的字。训练集实际出现 {data['observed_train_characters']:,} 种普通字符。

各划分分别用换行连接句子，并在首尾补一个换行，再按 {cfg['block_size']} 个输入位置切成独立块。输入 `x=stream[s:s+128]`，目标 `y=stream[s+1:s+129]`，两者错开一位。每个目标字符只计分一次；末块补 PAD，其无效目标标记为 -100 并从交叉熵与准确率中排除。换行、UNK 都参与计分，PAD 不参与输出分布。块之间不传递隐藏状态，块内最多利用 128 字符上下文，也可能读到前一条句子。

分词结果保存在 [data/tokens.npz](data/tokens.npz)，字表在 [data/vocab.json](data/vocab.json)。[results/data_stats.json](results/data_stats.json) 记录处理规则、划分数量和原始语料 SHA-256，运行 `prepare_data.py` 可由原始文件重新构建。

## 三、模型设计

```text
输入字符 ID [B, T]，T ≤ 128
  → Token Embedding(4000, 192) + 可学习 Position Embedding(128, 192)
  → Dropout(0.1)
  → Transformer Block × 4
      ├─ x = x + CausalMultiHeadAttention(LayerNorm(x))
      └─ x = x + FFN(LayerNorm(x))
  → LayerNorm
  → 共享字符 Embedding 权重的输出层
  → logits [B, T, 4000]
```

| 参数 | 设置 |
| --- | ---: |
| Transformer 层数 | {cfg['n_layers']} |
| 隐藏维度 d_model | {cfg['d_model']} |
| 注意力头数 / 单头维度 | {cfg['n_heads']} / {cfg['d_model']//cfg['n_heads']} |
| FFN 中间维度 | {cfg['d_ff']} |
| 最大上下文 | {cfg['block_size']} |
| Dropout | {cfg['dropout']} |
| 总参数量（共享权重不重复计数） | **{training['parameters']:,}** |

注意力计算为 `softmax(QKᵀ / sqrt(d_head) + causal_mask) V`，通过 PyTorch 的 `scaled_dot_product_attention(..., is_causal=True)` 实现，右侧未来位置被屏蔽。FFN 为 `192 → 768 → GELU → 192`；使用 Pre-LN 和残差连接稳定训练。模型定义见 [model.py](model.py)。

## 四、训练方法

训练目标是有效目标位置上的平均交叉熵。损失直接接收 logits，训练时不先做 Softmax；采样生成时才将最后一个位置的 logits 转为概率。

| 训练项 | 设置 |
| --- | --- |
| 优化器 | AdamW，betas=(0.9, 0.95) |
| 学习率 | {cfg['learning_rate']}，预热 {cfg['warmup_steps']} 步后余弦衰减至 {cfg['min_learning_rate']} |
| 权重衰减 | {cfg['weight_decay']} |
| Batch size | {cfg['batch_size']}，保留最后不足一批的数据 |
| 梯度裁剪 | L2 范数上限 {cfg['gradient_clip']} |
| 随机种子 | {cfg['seed']} |
| 最大训练轮数 / 早停耐心 | {cfg['epochs']} / {cfg['patience']} 轮 |
| 选型依据 | 验证集 token NLL 最低 |
| 本次设备 / 精度 | {training['environment']['gpu']} / BF16 autocast |

实际训练 {training['epochs_run']} 轮，保存第 **{training['best_epoch']}** 轮；本次未触发早停。训练耗时约 **{training['training_seconds']:.1f} 秒**，不含导入、数据处理和最终评测。初始随机模型验证 PPL 为 {training['initial_val']['perplexity']:.2f}，训练后降到 {metrics['transformer']['val']['perplexity']:.2f}。训练预算内验证损失持续改善，不据此声称已经达到最优训练规模。

配置见 [config.json](config.json)，逐轮记录见 [results/history.json](results/history.json)，环境、权重哈希及选型信息见 [results/training.json](results/training.json)。字表、权重和训练轮次均未使用测试集选择。

![训练损失和测试集困惑度对比](results/training_curve.png)

左图训练损失在开启 Dropout、参数不断更新时统计，验证损失在关闭 Dropout 的固定模型上统计，因此验证损失可以低于训练损失。右图仅比较已锁定模型的独立测试结果。

## 五、最终评估

`NLL = -Σ log P(y_t | x_≤t) / N`，`PPL = exp(NLL)`，N 为有效目标字符数。先累加所有目标的损失，再除以总字符数，不直接平均各批次 PPL。准确率是下一字符 Top-1 命中率。

| 划分 | NLL | PPL ↓ | 下一字符准确率 ↑ |
| --- | ---: | ---: | ---: |
{score_rows}

训练集这一行是选中权重在关闭 Dropout 后重新评估的结果，与训练过程中记录的在线训练损失含义不同。测试集共计分 **{test['tokens']:,}** 个目标字符，命中 {test['correct']:,} 个。

| 测试模型 | PPL ↓ | 下一字符准确率 ↑ |
| --- | ---: | ---: |
| 均匀分布（不含 PAD 的 3999 项） | {baseline['uniform']['perplexity']:.2f} | — |
{baseline_rows}
| Transformer | **{test['perplexity']:.2f}** | **{test['accuracy']:.2%}** |

统计基线只从训练集计数，平滑系数固定为 0.1，采用完全相同的字表、划分、UNK 和换行计分规则。Transformer 测试 PPL 比 Bigram 低 **{reduction:.2%}**。没有直接沿用课件中的 RNN/LSTM 指标，因为本项目重新去重、划分并保留 UNK，评测条件不同。

详细评测见 [results/metrics.json](results/metrics.json)。PPL 衡量对真实目标字符分配的概率，并不是“从 PPL 个字里猜一个”的严格计数。字符级 PPL 不能直接与采用其他字表或分词器的模型比较。

## 六、文本生成结果与局限

以下使用同一个保存模型，提示固定，`temperature=0.8`、`top_k=40`、随机种子 42，最多生成 60 个新字符，遇到换行停止。全部六个提示的结果如下，未根据好坏筛选。

| 输入提示 | 实际续写 |
| --- | --- |
{examples}

同样以“我想学习”为提示，贪心解码（temperature=0）得到：**{generations[-1]['full_text']}**。完整生成配置和结果保存在 [results/generations.json](results/generations.json)。生成时屏蔽 PAD/UNK；正式 PPL 评估仍把真实 UNK 纳入计分，未使用 Top-k 或温度修改评估分布。

模型已能生成部分常见问句及短语，但也出现搭配不合理、重复和主体不清的文本，例如“人工智能手机号码怎么设置”。这与短句语料、约 257 万参数及有限训练预算有关。它没有指令微调或事实核查能力；测试分数反映同分布字符预测，不代表开放域问答质量。上下文窗口为 128 字符，续写时滑动截取最近窗口，无法保留更早的完整历史。

## 七、运行方式

进入本 README 所在目录后执行：

```powershell
python -m pip install -r requirements.txt

# 从零完成数据处理、训练、评估和 README 更新
python -B -X utf8 run_all.py

# 直接用现有权重续写；不重新训练
python -B -X utf8 generate.py --prompt "我想学习" --max-new-tokens 60
python -B -X utf8 generate.py --prompt "今天天气" --temperature 0

# 独立运行关键正确性测试
python -B -X utf8 test_model.py

# 使用现有权重重新评估并更新 README
python -B -X utf8 run_all.py --evaluate-only
```

`device=auto` 优先使用 CUDA，无 GPU 时使用 CPU；也可在 `config.json` 中指定 `cpu`。CPU 或不同 GPU 的数值可能与本次 BF16 结果略有差异。重新训练会覆盖本目录的模型和结果。

本次环境：Python {training['environment']['python']}，PyTorch {training['environment']['torch']}，NumPy {training['environment']['numpy']}。测试覆盖因果掩码、前缀与补齐不变性、输入目标错位、尾块目标完整性、有限梯度、权重共享、去重划分及上下文越界处理。

## 八、文件说明

| 文件或目录 | 用途 |
| --- | --- |
| `README.md` | 课程对应、数据与模型设计、训练方法、指标、曲线和生成结果 |
| `config.json`、`requirements.txt` | 实验配置与依赖 |
| `prepare_data.py`、`model.py`、`train.py` | 数据处理、模型与训练 |
| `evaluate.py`、`generate.py` | 独立评估与续写 |
| `common.py`、`run_all.py`、`build_readme.py` | 公共读写、一键运行和报告更新 |
| `test_model.py` | 模型和数据边界测试 |
| `data/` | 课程原始语料、可复现的编码数据和字表 |
| `models/best_model.pt` | 最低验证 NLL 对应的权重及模型配置 |
| `results/` | 数据统计、逐轮历史、最终指标、实际续写和 README 引用的曲线 |
'''
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print("README.md updated.", flush=True)


if __name__ == "__main__":
    build_readme()
