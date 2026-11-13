# 第五周作业：在第四周 Transformer 上做 SFT

代码直接加载本项目中的 `pretrained/tf_mask.pt`，模型定义来自
`pretrained/transformer_lm.py` 中的 `TransformerLM`。它们是第四周文件的原样副本，
这是约 256 万参数的字符级教学模型。第五周文件夹现在可以独立运行，
无需第四周目录，无需下载其他模型或重新预训练，不使用 Hugging Face Trainer 或 LoRA。
默认对全部参数进行微调；脚本按自身位置寻找权重和数据。

上传 GitHub 时，把本文件夹作为仓库目录即可。以下命令均在本文件夹（仓库根目录）执行。
原有 `week5-大模型初探/` 是其他课程材料，不是运行 SFT 所需的依赖。

```text
.
├── sft_lm.py                  # SFT 训练与生成对比
├── plot_sft.py                # 训练指标绘图，可直接点击运行
├── test_sft_lm.py             # 自动检查
├── requirements_sft.txt       # PyTorch 与 Matplotlib 依赖
├── README_SFT.md
├── .gitignore
├── pretrained/
│   ├── __init__.py
│   ├── transformer_lm.py      # 第四周模型代码的副本
│   └── tf_mask.pt             # 第四周预训练权重（约 9.9 MiB）
└── sft_train/
    ├── sft_train.jsonl
    └── sft_val.jsonl
```

`pretrained/tf_mask.pt` 应随仓库提交；`.gitignore` 只排除缓存、虚拟环境和默认训练输出，
不会忽略此权重。没有复制无因果掩码的 `tf_nomask.pt`，因为本实验不使用它。
复制的模型代码还保留原来的预训练入口，但重新预训练需要另行提供语料；
本项目已附带预训练权重，执行 SFT 不需要原始预训练语料。

## 1. 环境与数据检查

本机已安装 PyTorch 的解释器是 `/opt/anaconda3/bin/python`；系统的
`/usr/bin/python3` 没有 PyTorch。以下命令使用终端中的 `python`。
如果更换环境，先安装依赖：

```bash
python -m pip install -r "requirements_sft.txt"
```

在本文件夹（仓库根目录）执行：

```bash
python "sft_lm.py" inspect
```

程序报告训练/验证样本数、词表扩展、长度、超长样本、未知字符和完全重复的问答对。
默认读取 `sft_train/sft_train.jsonl`（3000 条）和 `sft_train/sft_val.jsonl`（200 条）。
每行必须包含非空字符串 `instruction` 和 `output`。

## 2. 开始完整训练

```bash
python "sft_lm.py" train
```

默认设置：最多 10 轮，batch size 16，AdamW 学习率 0.0003，梯度裁剪 1.0，
验证 loss 连续 3 轮未改善则早停；自动选择 CUDA、MPS、CPU。
这些是起始超参数，不保证此模型能准确回答各种问题，请结合验证结果调整。
CPU 训练所需时间取决于硬件；每轮会打印进度。

可手动调整，并用独立目录保留不同实验：

```bash
python "sft_lm.py" train \
  --epochs 15 --batch-size 16 --lr 0.0001 \
  --output-dir "sft_output_lr1e4"
```

默认输出到 `sft_output/`：

| 文件 | 含义 |
| --- | --- |
| `pre_sft.pt` | 扩展词表和位置编码后、第一次更新前的模型，作为精确的训练前基线 |
| `best.pt` | 验证答案 loss 最低的微调模型 |
| `last.pt` | 最后执行完的一轮模型 |
| `history.json` | 基线与每轮的训练/验证损失、答案 PPL、前缀符合率和时间 |
| `run_config.json` | 运行参数 |
| `plots/training_metrics.png` | Loss、PPL、前缀符合率及每轮耗时总览图 |
| `plots/training_metrics.svg` | 可缩放的矢量图 |
| `plots/training_metrics.csv` | 用于表格软件分析的原始指标，epoch 0 为微调前基线 |

输出目录非空时程序会拒绝覆盖，请用新的 `--output-dir`。
检查点保存模型、词表、模板及结构配置，不包含优化器状态；当前脚本不提供断点续训。

### 将已有训练记录画成图

训练完成后会自动绘图。已有模型无需重新训练，可以直接点击运行 `plot_sft.py`，或执行：

```bash
python plot_sft.py
```

默认读取本项目 `sft_output/history.json`，输出到 `sft_output/plots/`。
指定其他实验或将图保存到仓库中未被忽略的目录：

```bash
python plot_sft.py --history sft_output_lr1e4/history.json --output-dir figures/lr1e4
```

如果提示缺少依赖，先运行 `python -m pip install matplotlib`。
PPL 图使用对数坐标以同时看清预训练基线和后续变化；数值来自记录，不做平滑。
前缀符合率默认是每轮抽查的 20 条验证问题，不是全部 200 条，也不代表内容正确率。
CSV 同时保留每轮的抽查分子与分母。训练 loss 是轮内模型持续更新时的批次平均，
验证 loss 是轮末模型的评估结果，因此两者并非在完全相同的模型权重下测量。

## 3. 对比 SFT 前后的生成

```bash
python "sft_lm.py" compare \
  --instruction "给定两个数字，选择较大的数。数字1：15，数字2：28" \
                "解释什么是机器学习。"
```

默认加载 `sft_output/best.pt` 及同目录的 `pre_sft.pt`。
两个模型使用完全相同的输入模板、词表、上下文长度和解码参数。
这里的“训练前”指扩展后的起始模型；原始 `tf_mask.pt` 的全部已有权重都被保留，
但它本身没有新增字符和新增位置的参数，因而不直接把原模型作为不同配置的对照。

指定其他实验并保存生成结果：

```bash
python "sft_lm.py" compare \
  --checkpoint "sft_output_lr1e4/best.pt" \
  --instruction "什么是深度学习？" \
  --save-json "sft_output_lr1e4/comparison.json"
```

只使用微调模型：

```bash
python "sft_lm.py" generate \
  --instruction "解释什么是机器学习。"
```

默认 `--temperature 0`，使用贪心解码便于比较。可添加
`--temperature 0.7 --top-k 40 --max-new-tokens 80` 观察采样效果。
遇到 EOS 会停止；达到生成上限也会停止，并报告是否生成 EOS。
上下文超过模型窗口时，生成过程仅使用最近的 128 个字符。
输入提示本身超过窗口则报错，要求缩短问题。

## 4. 代码中的关键设计

### 问答模板与损失

例如原始数据是：

```json
{"instruction": "选择较大的数：15和28", "output": "较大的数字是28。"}
```

输入提示固定为：

```text
问题：选择较大的数：15和28
回答：
```

模型需要自主生成：

```text
结论：较大的数字是28。<eos>
```

`<eos>` 是单独的结束 token。推理代码不会预先填入“结论：”，
因此可以观察模型是否真的学会了这个前缀。可用 `train --answer-prefix "答案："`
进行另一种格式实验；模板会保存在检查点里，推理自动读取。

训练仍然预测下一个字符：`x = full[:-1]`、`y = full[1:]`。
只计算答案（含“结论：”）和 EOS 的交叉熵；问题和右侧 PAD 的标签使用 `-100`，
通过 `ignore_index=-100` 排除。问题仍然参与注意力计算。
注意错位后的边界：最后一个提示字符的位置负责预测答案的第一个字符。

### 词表与上下文

原模型有 4000 个字符和 64 个位置。默认方案为：

1. 保持全部旧字符编号不变，只追加训练数据和模板中的缺失字符，以及 PAD/EOS/UNK。
2. 扩展输入嵌入和共享的输出头，复制全部旧权重；新增字符向量随机初始化。
3. 将位置表扩展至 128，复制前 64 行，新增位置向量随机初始化。
4. 复制其余全部 Transformer 权重，并继续保持因果掩码。

这不是仅仅修改 `seq_len` 参数。新增位置和字符没有经过预训练，需要在 SFT 中学习，
可能影响效果；本实验重点是掌握迁移权重、答案监督和格式学习的完整流程。
词表仅从训练集建立。验证或推理时出现的新字符映射为 UNK，并报告数量，避免默默删字。
超长训练样本整条跳过并报告数量，不强行截断问题或答案；可用 `inspect` 提前检查。
默认 128 应以你当前数据的实际检查输出为准；更换数据或前缀后请重新检查。

### 指标与数据质量

`answer_loss` 和 `answer_ppl` 只统计答案及 EOS，不等同于第四周的全语料 PPL。
`prefix_accuracy` 默认对前 20 条可容纳的验证问题做贪心生成，检查是否自主输出“结论：”。
它只是前缀格式符合率，不代表答案正确、完整或能准确结束。
使用 `--format-eval-samples 200` 可以扩大检测范围。

模型按验证 loss 保存最佳轮次，因此最佳 loss 与最高格式符合率未必出现在同一轮。
写作业时同时记录损失、前缀符合率、是否正常结束，并人工核对未见过问题的答案。
可以对比 `best.pt` 和 `last.pt`，但不要把验证集用于继续训练。

原始数据存在内容错误。例如验证样本把 12 和 25 判断为不互质，实际二者互质。
代码不擅自改动原始标签；正式评价前应清理数据，小模型也可能学习错误答案。

## 5. 快速验证和自动检查

以下命令只跑两个更新步骤，用于验证加载、反向传播、保存和推理流程；
它不构成完成训练，也不应期待输出已经符合格式：

```bash
python "sft_lm.py" train \
  --epochs 1 --train-limit 8 --val-limit 4 --batch-size 2 \
  --max-steps 2 --format-eval-samples 2 \
  --output-dir "sft_smoke"

python "sft_lm.py" compare \
  --checkpoint "sft_smoke/best.pt" \
  --instruction "什么是机器学习？" --max-new-tokens 12

python "test_sft_lm.py"
```

自动检查覆盖监督边界、PAD 屏蔽、超长样本、未知字符、因果性、旧权重保留与检查点重载。

最初的 SFT 流程验证（2026-11-11）：8 项自动检查全部通过；默认模板下，训练集 3000 条、
验证集 200 条全部可用，扩展词表为 4106，验证集有 5 个字符映射为 UNK。
使用全部数据建表和验证、实际执行 2 次参数更新后，验证答案 loss 从约 5.131 降至 4.754，
并成功保存、重载和运行训练前后生成对比。测试文件保存在系统临时目录，未占用默认输出目录。
这只是流程验证：两条抽查问题的前缀符合率仍为 0，不代表已经完成 SFT 或获得可用的问答模型。

独立打包验证：将上述项目文件复制到没有第四周相邻目录的临时文件夹后，
9 项自动检查全部通过，并成功执行数据检查、一次参数更新、检查点保存和生成对比。
同时检查了 Git 忽略规则：预训练代码、权重和数据可提交，训练输出与 Python 缓存被忽略。
