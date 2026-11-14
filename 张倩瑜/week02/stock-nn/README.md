# 股票五分类神经网络（手写 NumPy + PyTorch 双版本对照）

本项目实现了一个完整的**股票投资属性五分类神经网络模型**，从 Hugging Face 公开股票日线数据集中提取 5 维量化指标向量 $\mathbf{x} \in \mathbb{R}^5$，并通过**手写 NumPy 底层实现**与 **PyTorch 对照实现**完成：

1. **前向传播（Forward）**：接收任意 5 维向量 $\mathbf{x} \in \mathbb{R}^5$，计算出 5 个类别的原始分数（Logits $\mathbf{z} \in \mathbb{R}^5$）
2. **交叉熵损失（Cross-Entropy Loss）**：经过数值稳定 Softmax 计算概率分布 $\hat{\mathbf{y}}$ 与多分类交叉熵损失 $\mathcal{L}$
3. **反向传播（Backpropagation）**：依据链式法则逐层计算每个参数（$\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2, \mathbf{W}_3, \mathbf{b}_3$）的精确梯度，并与中心有限差分数值梯度、PyTorch `autograd` 梯度进行逐位校验（误差 $< 10^{-16}$）
4. **Adam 优化器（Adam Optimizer）**：手写一阶动量 $m_t$、二阶自适应动量 $v_t$ 与偏差修正 $\hat{m}_t, \hat{v}_t$，动态调整每个参数的更新步长（160 轮训练后与 PyTorch `optim.Adam` 最大权重差异仅 $9.58 \times 10^{-13}$）

---

## 一、 数据集与五维特征工程

- **Hugging Face 数据集**：[`no-ry/world-stock-prices-daily-updating`](https://huggingface.co/datasets/no-ry/world-stock-prices-daily-updating)（包含 2000–2025 年全球 60+ 家知名上市公司的 `310,122` 条日线记录）。
- **输入 5 维向量 $\mathbf{x} = [x_1, x_2, x_3, x_4, x_5]^T$**：
  - $x_1$ (`cagr`)：10 年期（或长周期窗口）年化收益率 $\left(\frac{P_{\text{end}}}{P_{\text{start}}}\right)^{1/Y} - 1$
  - $x_2$ (`volatility`)：年化波动率 $\text{std}(r_{\text{daily}}) \times \sqrt{252}$
  - $x_3$ (`max_drawdown`)：历史最大回撤 $\max_t \frac{\text{Peak}_t - P_t}{\text{Peak}_t}$
  - $x_4$ (`peak_drop`)：期末距历史最高点回落比 $\frac{P_{\max} - P_{\text{end}}}{P_{\max}}$
  - $x_5$ (`swing_amp`)：60 日滚动高低点波段振幅均值 $\text{mean}\left(\frac{P_{\max,60} - P_{\min,60}}{P_{\min,60}}\right)$

- **输出 5 大股票分类标签 $y \in \{0, 1, 2, 3, 4\}$**：
  - **`0` — 10 年期暴利大牛股 (`ten_bagger`)**：高年化收益且长期创新高（代表：`NVDA`, `MSFT`, `AAPL`, `COST`, `AMZN`）
  - **`1` — 长期稳健持有股 (`steady_hold`)**：稳健正收益、低波动、浅回撤（代表：`KO`, `PG`, `JNJ`, `MCD`, `UL`）
  - **`2` — 适合炒波段股 (`swing_trade`)**：高波动率、高波段振幅、反复拉升（代表：`TSLA`, `AMD`, `SHOP`, `COIN`, `RBLX`）
  - **`3` — 风口暴涨及时撤退 (`bubble_retreat`)**：风口期剧烈冲高后崩盘回落逾 62%（代表：`PTON`, `ZM`, `SQ`, `ZI`, `UBSFY`）
  - **`4` — 完全不考虑股 (`avoid`)**：长期阴跌或年化收益跑输通胀且缺乏波段弹性（代表：`DIS`, `MMM`, `HMC`, `LUV`, `PHG`）

---

## 二、 核心数学公式与代码对应

### 1. 前向传播计算 5 个原始分数 (`stock_nn.forward`)
$$\mathbf{h}_1 = \text{ReLU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1), \quad \mathbf{h}_2 = \text{ReLU}(\mathbf{h}_1\mathbf{W}_2 + \mathbf{b}_2), \quad \mathbf{z} = \mathbf{h}_2\mathbf{W}_3 + \mathbf{b}_3 \in \mathbb{R}^5$$

### 2. Softmax 与交叉熵损失 (`stock_nn.softmax`, `stock_nn.cross_entropy_loss`)
$$\hat{y}_c = \frac{e^{z_c - \max_k z_k}}{\sum_{k=0}^{4} e^{z_k - \max_k z_k}}, \qquad \mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log\left(\hat{y}_{i, y_i}\right)$$

### 3. 反向传播计算每个参数的梯度 (`stock_nn.backward`)
$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \frac{1}{B}(\hat{\mathbf{y}} - \mathbf{Y}_{\text{onehot}})$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_3} = \mathbf{h}_2^T \frac{\partial \mathcal{L}}{\partial \mathbf{z}}, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{b}_3} = \sum_{i=1}^{B} \left(\frac{\partial \mathcal{L}}{\partial \mathbf{z}}\right)_i$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{2,\text{pre}}} = \left(\frac{\partial \mathcal{L}}{\partial \mathbf{z}} \mathbf{W}_3^T\right) \odot \mathbb{I}(\mathbf{h}_{2,\text{pre}} > 0), \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = \mathbf{h}_1^T \frac{\partial \mathcal{L}}{\partial \mathbf{h}_{2,\text{pre}}}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{1,\text{pre}}} = \left(\frac{\partial \mathcal{L}}{\partial \mathbf{h}_{2,\text{pre}}} \mathbf{W}_2^T\right) \odot \mathbb{I}(\mathbf{h}_{1,\text{pre}} > 0), \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \mathbf{X}^T \frac{\partial \mathcal{L}}{\partial \mathbf{h}_{1,\text{pre}}}$$

### 4. Adam 优化器自适应更新步长 (`stock_nn.AdamOptimizer.step`)
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \qquad \theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

---

## 三、 快速运行方式

```bash
# 1. 默认运行（加载 HF 数据 -> 梯度校验 -> NumPy & PyTorch 双版训练 -> 真实个股推理 -> 生成图表）
python 张倩瑜/week02/stock-nn/train.py

# 2. 传入任意自定义的 5 维向量做预测（格式: cagr,volatility,max_drawdown,peak_drop,swing_amp）
python 张倩瑜/week02/stock-nn/train.py --vector "0.40,0.35,0.45,0.05,0.28" --vector "-0.15,0.75,0.95,0.92,0.65"

# 3. 强制从 Hugging Face 原始 CSV 重新滑动窗口提取特征
python 张倩瑜/week02/stock-nn/train.py --rebuild
```

## 四、 目录结构

- `dataset_hf.py`：Hugging Face 股票数据集自动下载、拆股断层清洗、5 维特征计算与缓存
- `stock_nn.py`：手写 NumPy 神经网络（Forward / Softmax / CrossEntropy / Backprop / Adam）+ PyTorch 等价对照实现
- `train.py`：训练评估主入口、命令行任意 5 维向量推理演示、可视化看板生成
- `training_results.png`：训练收敛曲线、测试集混淆矩阵、Hugging Face 真实个股分布与 5 个原始分数（Logits）对比图
