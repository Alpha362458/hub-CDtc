# 基于 LoRA 的语言模型监督微调（SFT）

本目录交付物演示如何对一个开源语言模型进行**监督微调（Supervised Fine-Tuning, SFT）**。
基座模型选用 `Qwen/Qwen2.5-0.5B-Instruct`，采用 **LoRA** 参数高效微调，
在自带的指令数据集上完成 2 个 epoch 的训练，并产出可推理的合并模型。

## 目录结构

```
week5/
├── train_sft.py        # 训练脚本（LoRA SFT）
├── inference.py        # 推理 / 效果演示脚本
├── README.md           # 本说明文档
└── outputs/            # 训练产物（运行后生成）
    ├── lora/           # LoRA adapter 权重 + tokenizer
    ├── merged/         # 合并 LoRA 后的完整模型权重
    ├── checkpoint-376/ # 训练中间检查点
    └── metrics.json    # 训练 / 验证指标
```

> 训练数据来自上层 `sft_train/sft_train.jsonl`（3000 条）与 `sft_train/sft_val.jsonl`（200 条），
> 每条样例包含 `instruction` 与 `output` 两个字段。

## 环境依赖

| 组件        | 版本     |
| ----------- | -------- |
| Python      | 3.12     |
| torch       | 2.14.1 (CUDA 12.6) |
| transformers| 5.22.2   |
| trl         | 1.17.1   |
| peft        | 0.15.0   |
| accelerate  | 1.15.0   |
| datasets    | 5.2.0    |

运行设备：NVIDIA GeForce RTX 4090 Laptop GPU（16GB）。

## 快速开始

### 1. 训练

```bash
python train_sft.py --merge_and_save
```

常用参数：

| 参数                | 默认值                          | 说明                         |
| ------------------- | ------------------------------- | ---------------------------- |
| `--model`           | `Qwen/Qwen2.5-0.5B-Instruct`    | 基座模型名或本地路径         |
| `--train_file`      | `../sft_train/sft_train.jsonl`  | 训练集                       |
| `--val_file`        | `../sft_train/sft_val.jsonl`     | 验证集                       |
| `--output_dir`      | `./outputs`                     | 输出目录                     |
| `--epochs`          | `2.0`                           | 训练轮数                     |
| `--batch_size`      | `8`                             | 单卡 batch size               |
| `--grad_accum`      | `2`                             | 梯度累积步数                 |
| `--lr`              | `1e-4`                          | 学习率                       |
| `--max_seq_length`  | `512`                           | 最大序列长度                 |
| `--lora_r`          | `16`                            | LoRA 秩                      |
| `--lora_alpha`      | `32`                            | LoRA 缩放系数                |
| `--merge_and_save`  | 关                              | 训练后合并并保存完整模型     |

### 2. 推理

```bash
# 使用合并后的完整模型
python inference.py --merged outputs/merged

# 或仅加载 LoRA adapter
python inference.py --adapter outputs/lora
```

## 训练结果

| 指标                    | 值      |
| ----------------------- | ------- |
| train_loss              | 1.0637  |
| eval_loss               | 1.0863  |
| eval_mean_token_accuracy| 0.7245  |
| train_runtime           | 227.5 s |
| 可训练参数占比          | 1.75%   |

可训练参数为 8.8M / 502.8M，LoRA 仅微调注意力与 MLP 投影矩阵
（`q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj`）。

## 推理示例

```
[user] 给定两个数字，请选择其中的较大数并将其打印出来。 数字1: 128 数字2: 96
[assistant] 较大的数是96，因此打印出的结果为：96。

[user] 计算一组数字的平均值和标准差。 数字：10, 20, 30, 40, 50
[assistant] 这组数字的平均值为 30，标准差为 10.87。

[user] 请把下面这句话翻译成英文：今天天气真好。
[assistant] The weather is really nice today.
```

## 实现要点

1. **数据格式化**：将 `instruction / output` 转写为 chat 模板的 `prompt / completion`，
   利用 `completion_only_loss=True` 仅对回答部分计算损失，避免在指令上浪费学习容量。
2. **参数高效微调**：使用 LoRA（r=16, alpha=32）冻结基座，仅训练少量适配器参数，
   单卡 16GB 即可完成训练，且产物体积小（adapter ≈ 35MB）。
3. **混合精度**：在 CUDA 可用时启用 `bf16`，兼顾速度与数值稳定性。
4. **模型合并**：训练结束后通过 `merge_and_unload` 将 LoRA 权重合并回基座，
   产出可独立部署的完整模型（`outputs/merged/`），便于脱离 PEFT 框架直接加载推理。
