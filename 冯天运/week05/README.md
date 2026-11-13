# week5 作业：基于语言模型的 SFT

只需阅读和运行 **`train.py`**。数据处理、模型、训练、推理和自检已合并，每条非空代码行保留中文注释。基座、数据和全部历史实验保留；精简前的权重仍可加载。

## 文件用途

| 文件或目录 | 用途 |
| --- | --- |
| `train.py` | 唯一 Python 入口：训练、推理、自检 |
| `README.md` | 操作说明、原理和结果 |
| `requirements.txt` | 唯一依赖 PyTorch；当前机器已安装 |
| `data/` | 3000 条训练、200 条验证数据副本 |
| `base/` | week4 的预训练 Transformer 权重副本 |
| `outputs/` | 已有及新训练的模型、指标，禁止覆盖旧实验 |
| `AGENTS.md`、`ROADMAP.md`、`ENVIRONMENT.md` | 按协作规则保留的项目约定、进度及环境记录，日常学习无需逐一阅读 |

精简后的代码分为四段：数据与标签 → 模型与生成 → 自检 → 训练及命令行。旧 `data.py`、`model.py`、`infer.py`、`verify.py` 及独立 `REPORT.md` 的功能或关键内容均已合并，冗余文件和旧缓存已按用户批准删除。

## 三条常用命令

先在 PowerShell 进入本目录：

```powershell
# 进入作业目录；引号保护路径中的 &。
Set-Location -LiteralPath 'F:\冯天运\AI\AI大模型&Agent\八斗\主课程\week5-大模型初探\sft_coursework'
# 设置中文输出编码。
$env:PYTHONIOENCODING = 'utf-8'
# 1. 检查参数迁移、因果性、标签错位、padding、梯度和保存加载。
python train.py --verify
# 2. 加载你训练 50 轮的模型；必须明确指定权重，避免误用最初的三轮模型。
python train.py --checkpoint outputs/verify_02/best_sft.pt --instruction '将以下句子转换为被动语态。 句子： Tom 发现了一份工作。'
# 3. 从 week4 基座开始新训练，保持旧实验不变。
python train.py --epochs 3 --output outputs/new_run_01
```

`new_run_01` 若已存在，请改用新名字。推理不传 `--checkpoint` 时默认使用 `outputs/course_run/best_sft.pt`，即最初的三轮模型。已取消独立推理和自检入口的必要性，统一使用上面命令。

```powershell
# 仅用少量样本体验完整流程。
python train.py --epochs 1 --limit-train 64 --limit-val 32 --output outputs/quick_new_01
# 查看实际参数、逐轮指标、最佳轮次和三条生成前后对照。
Get-Content -LiteralPath outputs/new_run_01/metrics.json -Encoding UTF8
# 查看全部参数。
python train.py --help
```

默认参数：CPU、4 线程、batch_size=32、lr=0.0001、max_length=128、seed=42、3 轮。`--base`、`--train`、`--val` 可指定其他输入。默认输入路径相对脚本目录，相对输出路径相对当前终端目录。所有实验从基座重新开始，不支持中断续训。

换电脑且未安装 PyTorch 时，可在本目录创建虚拟环境；本次未安装依赖或改系统配置：

```powershell
# 创建本目录专用环境。
python -m venv .venv
# 在隔离环境安装依赖。
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# 后续命令中的 python 均可换成此环境解释器。
.\.venv\Scripts\python.exe train.py --verify
```

## 与 week4、week5 的联系

week4 的字符级 Transformer 学习下一字符预测。本作业加载其真实权重，保留 2 层、128 维、4 个注意力头及因果掩码。虽然使用 `TransformerEncoderLayer`，严格因果掩码使其执行单向自回归建模，不是 BERT 双向 MLM。

week5 第 27–29 页强调对话模板、回答 loss mask、EOS 和自回归生成。因此把数据拼为 `<user>问题<assistant>答案<eos>`，只对答案与 EOS 算交叉熵。角色标记各占一个 token；问题和右侧 padding 的标签是 `-100`。输入 `ids[:-1]`、目标 `labels[1:]`，只错位一次。

以“问题：你好；答案：好”为例：

| 输入 | `<user>` | 你 | 好 | `<assistant>` | 好 |
| --- | --- | --- | --- | --- | --- |
| 预测目标 | 忽略 | 忽略 | 忽略 | 好 | `<eos>` |

因果 mask 负责不能看未来，loss mask 负责只学回答；两者作用不同。训练时有正确答案历史，生成时只能回填自己预测的字符，所以验证 loss 好不代表自由回答正确。

原 4000 个字符 ID 和预训练参数原样保留；仅从训练集追加角色和缺失字符，扩展到 4108 个 token，参数量为 1,386,252。新增词表行随机初始化，主干结构不变。验证集未知问题字符用 UNK，未知答案字符不计 loss。完整实验验证中分别有 1 和 4 个未知字符。

所有批次均右侧补齐，因果 mask 阻止有效位置看到右侧 PAD，PAD 目标又被忽略，因此此实现无需额外 padding attention mask。改为左侧补齐或双向模型时不能照搬。完整序列超出上限直接报错，不静默裁掉答案。位置表虽有 512 行，week4 实际训练长度为 64，较长位置需要在 SFT 继续学习。

## 实验结果与能力边界

最初完整实验：3000/200 条数据，三轮 CPU，约 42.58 秒；环境 Python 3.12.8、PyTorch 2.5.1+cpu。

| 阶段 | 验证回答 loss | 验证回答 PPL |
| --- | ---: | ---: |
| 微调前，同一扩展字表与模板 | 5.4823 | 240.39 |
| 第 1 轮 | 4.6663 | 106.30 |
| 第 2 轮 | 4.2881 | 72.83 |
| 第 3 轮 | 4.1024 | 60.49 |

精确记录、输入 SHA-256 和生成示例见 `outputs/course_run/metrics.json`；你后续的结果保存在 `outputs/verify_02/`。训练持续记录验证指标，并保存验证 loss 最低的微调轮次；若所有轮次都比基线差，也会保存其中最好的微调模型，是否改善仍须比较基线。

模型已学到部分回答形式与 EOS，但仍有重复、答非所问。约 139 万参数的小基座与有限预训练语料难以支撑多任务问答，继续增加轮数不能保证解决问题。PPL 按已知答案字符/EOS 加权，不能直接与 week4 普通文本 PPL 比较；也不是答案正确率。生成用贪心解码，屏蔽 PAD、UNK 和角色标记；`eos` 表示主动结束，其他停止原因表示达到上限。

原始数据未改：例如第二条的标准差标为 33.41，但实际总体/样本标准差约 28.17/31.50。尚未全面审核标注。未验证 GPU、LoRA、全部答案正确率和遗忘程度。

## 精简验证

已通过逐行注释检查、关键自检；与原分文件实现相比，编码张量、迁移参数、前向 logits 和 50 轮模型同题生成完全一致。另用 8 条训练、4 条验证做一轮临时实验，验证训练保存加载正常，过程产物自动清理。保留原课程和全部历史实验；需要切回旧模型时只需指定对应 `--checkpoint`。
