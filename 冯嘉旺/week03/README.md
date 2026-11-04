# 12 类语种识别作业

## 目录说明

- `train_langid_12cls.py`：12 类语种分类训练脚本（自包含，不依赖课堂示例）
- `eval_langid_12cls.py`：12 类评估脚本（复用 `train_langid_12cls.py` 的模型）
- `demo_mixed_text_12cls.py`：12 类混排演示（复用 `eval_langid_12cls.py`）
- `langid_rnn_v12.pt`：已训练好的 12 类模型
- `data/langid/vocab/vocab_v12.txt`：由 `v12/train.csv` 生成的字符表
- `data/langid/v12/`：12 类训练、验证、测试数据
- `data/langid/test_hard.csv`：短文本/去重音/去假名难度集
- `data/langid/test_openset.csv`：训练中未出现的语种

目录只保留 12 类流程所需的文件；课堂 8 类示例文件已清理。

## 运行环境

需要 Python 3.12 和 PyTorch。当前机器可直接使用：

```powershell
C:\Users\fengj\.conda\envs\py312\python.exe
```

如果当前终端已经激活了该环境，也可以直接把命令里的解释器换成 `python`。

## 训练

在**本文件夹内**运行：

```powershell
python train_langid_12cls.py
```

默认训练 8 个 epoch，完成后覆盖 `langid_rnn_v12.pt`。字符表已存在时会直接复用；需要重新生成时加 `--rebuild-vocab`：

```powershell
python train_langid_12cls.py --rebuild-vocab
```

## 评估

```powershell
python eval_langid_12cls.py --split test
python eval_langid_12cls.py --split test_hard
python eval_langid_12cls.py --openset
```

默认读取 `langid_rnn_v12.pt` 和 `data/langid/v12`；需要换模型或数据时再加
`--model` / `--data-dir`。

当前模型结果：

| 项目 | 结果 |
|---|---|
| 测试集 accuracy | 95.83% |
| 测试集 macro-F1 | 95.85% |
| 短文本切片 | 85.5% |
| 去重音切片 | 73.3% |
| 去假名切片 | 20.0% |
| 混排句子（模型） | 18/20 |
| 混排句子（纯规则） | 9/20 |
| 开放集拒识率 | 32.2% |

表中短文本、去重音、去假名三项来自独立的 `test_hard.csv`，不是 v12 test 的子集。
主要混淆是 `es ↔ pt`，两类都使用拉丁字母且词形相近。`nokana` 切片中，日语去掉假名后只剩汉字，与中文高度重合，准确率下降属于预期现象。

## 混排演示

```powershell
python demo_mixed_text_12cls.py
```

演示脚本默认加载 `langid_rnn_v12.pt` 和 `data/langid/v12`；它不再依赖课堂 8 类示例。
