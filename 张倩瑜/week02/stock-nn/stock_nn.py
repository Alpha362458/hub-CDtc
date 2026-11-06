"""手写神经网络五分类模型（NumPy 从零推导版 + PyTorch 对照版）。

任务链路严格对应：
    1) 输入任意 5 维向量 x in R^5（或批量 X in R^{B x 5}）
    2) 神经网络前向传播 (Forward) 计算出 5 个原始分数 logits z in R^5
    3) 用 Cross-Entropy Loss (交叉熵损失) 计算损失 L
    4) 用 Backpropagation (反向传播链式法则) 计算每个参数的梯度 dW1, db1, dW2, db2, dW3, db3
    5) 用 Adam Optimizer (一阶动量 m_t + 二阶动量 v_t + 偏差修正) 调整更新步长

每个手写函数下方均附有对应的数学公式与 PyTorch 等价调用说明。
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    HAVE_TORCH = False


# =============================================================================
#  一、手写 NumPy 版：参数初始化、前向传播、Cross-Entropy、Backprop、Adam
# =============================================================================
def init_params(
    input_dim: int = 5,
    hidden1: int = 32,
    hidden2: int = 16,
    num_classes: int = 5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """He (Kaiming) 初始化神经网络参数。

    网络结构: 5 (输入) -> 32 (ReLU) -> 16 (ReLU) -> 5 (原始分数 Logits)
    等价 PyTorch:
        nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 5))
    """
    rng = np.random.default_rng(seed)
    params = {
        "W1": rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden1)),
        "b1": np.zeros(hidden1, dtype=np.float64),
        "W2": rng.normal(0.0, np.sqrt(2.0 / hidden1), size=(hidden1, hidden2)),
        "b2": np.zeros(hidden2, dtype=np.float64),
        "W3": rng.normal(0.0, np.sqrt(2.0 / hidden2), size=(hidden2, num_classes)),
        "b3": np.zeros(num_classes, dtype=np.float64),
    }
    return params


def forward(
    X: np.ndarray, params: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """前向传播：输入任意 5 维向量（单条 (5,) 或批量 (B, 5)），计算出 5 个原始分数 logits。

    数学公式：
        h1_pre = X @ W1 + b1,        h1 = ReLU(h1_pre) = max(0, h1_pre)
        h2_pre = h1 @ W2 + b2,       h2 = ReLU(h2_pre) = max(0, h2_pre)
        z      = h2 @ W3 + b3        (z 即 5 个类别的原始分数 logits)

    等价 PyTorch：
        logits = model(X_tensor)
    """
    x_arr = np.asarray(X, dtype=np.float64)
    single_input = x_arr.ndim == 1
    if single_input:
        x_arr = x_arr[None, :]  # (1, 5)

    if x_arr.shape[1] != 5:
        raise ValueError(f"输入向量必须是 5 维，当前形状为 {X.shape}")

    # 第一层: 5 -> H1 + ReLU
    h1_pre = x_arr @ params["W1"] + params["b1"]
    h1 = np.maximum(0.0, h1_pre)

    # 第二层: H1 -> H2 + ReLU
    h2_pre = h1 @ params["W2"] + params["b2"]
    h2 = np.maximum(0.0, h2_pre)

    # 输出层: H2 -> 5 个原始分数 (Logits)
    logits = h2 @ params["W3"] + params["b3"]

    cache = {
        "X": x_arr,
        "h1_pre": h1_pre,
        "h1": h1,
        "h2_pre": h2_pre,
        "h2": h2,
        "logits": logits,
    }
    if single_input:
        return logits[0], cache
    return logits, cache


def softmax(logits: np.ndarray) -> np.ndarray:
    """数值稳定的 Softmax：把 5 个原始分数转化为概率分布。

    数学公式：
        p_c = exp(z_c - max_k z_k) / sum_{k=0}^4 exp(z_k - max_k z_k)

    等价 PyTorch：
        torch.softmax(logits, dim=-1)
    """
    z = np.asarray(logits, dtype=np.float64)
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def cross_entropy_loss(
    logits: np.ndarray, y: np.ndarray
) -> Tuple[float, np.ndarray]:
    """计算多分类交叉熵损失 (Cross-Entropy Loss) 并返回 Softmax 概率。

    数学公式：
        p = Softmax(z)
        L = - (1 / B) * sum_{i=1}^B log(p_{i, y_i})

    等价 PyTorch：
        nn.CrossEntropyLoss()(logits, y)
    """
    probs = softmax(logits)
    batch_size = logits.shape[0]
    # 选取每个样本真实类别 y_i 对应的预测概率 p_{i, y_i}
    correct_Class_probs = np.clip(probs[np.arange(batch_size), y], 1e-15, 1.0)
    loss = float(-np.mean(np.log(correct_Class_probs)))
    return loss, probs


def backward(
    cache: Dict[str, np.ndarray],
    probs: np.ndarray,
    y: np.ndarray,
    params: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """反向传播 (Backpropagation)：用链式法则计算每个参数的精确梯度。

    数学公式（设批量大小为 B）：
        1. 输出层原始分数梯度:
               dz = (probs - one_hot(y)) / B                in R^{B x 5}
        2. 第三层参数梯度:
               dW3 = h2^T @ dz                              in R^{H2 x 5}
               db3 = sum_i dz_i                             in R^5
        3. 反传至第二隐藏层 + ReLU 导数门控:
               dh2     = dz @ W3^T                          in R^{B x H2}
               dh2_pre = dh2 * I(h2_pre > 0)                in R^{B x H2}
               dW2     = h1^T @ dh2_pre                     in R^{H1 x H2}
               db2     = sum_i dh2_pre_i                    in R^{H2}
        4. 反传至第一隐藏层 + ReLU 导数门控:
               dh1     = dh2_pre @ W2^T                     in R^{B x H1}
               dh1_pre = dh1 * I(h1_pre > 0)                in R^{B x H1}
               dW1     = X^T @ dh1_pre                      in R^{5 x H1}
               db1     = sum_i dh1_pre_i                    in R^{H1}

    等价 PyTorch：
        loss.backward()  # 各参数梯度存入 param.grad
    """
    X = cache["X"]
    h1_pre, h1 = cache["h1_pre"], cache["h1"]
    h2_pre, h2 = cache["h2_pre"], cache["h2"]
    batch_size = X.shape[0]

    # Step 1: dL / dz = (p - y_onehot) / B
    dz = probs.copy()
    dz[np.arange(batch_size), y] -= 1.0
    dz /= batch_size

    # Step 2: 输出层 (W3, b3) 梯度
    dW3 = h2.T @ dz
    db3 = np.sum(dz, axis=0)

    # Step 3: 第二隐藏层 (W2, b2) 梯度
    dh2 = dz @ params["W3"].T
    dh2_pre = dh2 * (h2_pre > 0.0)
    dW2 = h1.T @ dh2_pre
    db2 = np.sum(dh2_pre, axis=0)

    # Step 4: 第一隐藏层 (W1, b1) 梯度
    dh1 = dh2_pre @ params["W2"].T
    dh1_pre = dh1 * (h1_pre > 0.0)
    dW1 = X.T @ dh1_pre
    db1 = np.sum(dh1_pre, axis=0)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "W3": dW3,
        "b3": db3,
    }


class AdamOptimizer:
    """手写 Adam 优化器：结合一阶动量与二阶自适应步长更新所有网络参数。

    数学公式（第 t 步，对每个参数 theta 及其梯度 g_t）：
        m_t     = beta1 * m_{t-1} + (1 - beta1) * g_t          (一阶动量：平滑方向)
        v_t     = beta2 * v_{t-1} + (1 - beta2) * (g_t ^ 2)    (二阶动量：自适应尺度)
        m_hat_t = m_t / (1 - beta1 ^ t)                        (一阶偏差修正)
        v_hat_t = v_t / (1 - beta2 ^ t)                        (二阶偏差修正)
        theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)

    等价 PyTorch：
        torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)
    """

    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params.items()}
        self.v: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in params.items()}

    def step(
        self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]
    ) -> None:
        """执行一步 Adam 参数更新。"""
        self.t += 1
        bias_corr1 = 1.0 - (self.beta1 ** self.t)
        bias_corr2 = 1.0 - (self.beta2 ** self.t)

        for name in params:
            g = grads[name]
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (g * g)

            m_hat = self.m[name] / bias_corr1
            v_hat = self.v[name] / bias_corr2

            step_size = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            params[name] -= step_size


# =============================================================================
#  二、梯度校验：对比手写 Backprop vs 有限差分数值梯度 vs PyTorch Autograd
# =============================================================================
def verify_backprop_and_adam(
    X_sample: np.ndarray, y_sample: np.ndarray, seed: int = 42
) -> Dict[str, float]:
    """验证手写反向传播与 Adam 更新的数学精度（对比中心差分数值梯度与 PyTorch Autograd）。"""
    params = init_params(seed=seed)
    logits, cache = forward(X_sample, params)
    loss, probs = cross_entropy_loss(logits, y_sample)
    grads = backward(cache, probs, y_sample, params)

    # 1) 对比中心有限差分 (f(theta + eps) - f(theta - eps)) / (2 * eps)
    eps = 1e-6
    max_num_diff = 0.0
    for name, param in params.items():
        it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = float(param[idx])
            param[idx] = orig + eps
            loss_plus, _ = cross_entropy_loss(forward(X_sample, params)[0], y_sample)
            param[idx] = orig - eps
            loss_minus, _ = cross_entropy_loss(forward(X_sample, params)[0], y_sample)
            param[idx] = orig
            num_grad = (loss_plus - loss_minus) / (2.0 * eps)
            diff = abs(float(grads[name][idx]) - num_grad)
            if diff > max_num_diff:
                max_num_diff = diff
            it.iternext()

    result = {"max_numerical_grad_diff": max_num_diff}

    # 2) 如果安装了 PyTorch，对比手写梯度 & Adam 一步更新后参数 vs PyTorch
    if HAVE_TORCH:
        params_np = copy.deepcopy(params)
        opt_np = AdamOptimizer(params_np, lr=0.01)

        # 构造相同权重的 PyTorch 模型 (float64)
        l1 = nn.Linear(5, 32, dtype=torch.float64)
        l2 = nn.Linear(32, 16, dtype=torch.float64)
        l3 = nn.Linear(16, 5, dtype=torch.float64)
        with torch.no_grad():
            l1.weight.copy_(torch.from_numpy(params["W1"].T))
            l1.bias.copy_(torch.from_numpy(params["b1"]))
            l2.weight.copy_(torch.from_numpy(params["W2"].T))
            l2.bias.copy_(torch.from_numpy(params["b2"]))
            l3.weight.copy_(torch.from_numpy(params["W3"].T))
            l3.bias.copy_(torch.from_numpy(params["b3"]))

        model_pt = nn.Sequential(l1, nn.ReLU(), l2, nn.ReLU(), l3)
        opt_pt = torch.optim.Adam(model_pt.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        criterion = nn.CrossEntropyLoss()

        x_t = torch.from_numpy(X_sample)
        y_t = torch.from_numpy(y_sample).long()
        logits_pt = model_pt(x_t)
        loss_pt = criterion(logits_pt, y_t)
        loss_pt.backward()

        grad_diffs = [
            np.max(np.abs(grads["W1"] - l1.weight.grad.numpy().T)),
            np.max(np.abs(grads["b1"] - l1.bias.grad.numpy())),
            np.max(np.abs(grads["W2"] - l2.weight.grad.numpy().T)),
            np.max(np.abs(grads["b2"] - l2.bias.grad.numpy())),
            np.max(np.abs(grads["W3"] - l3.weight.grad.numpy().T)),
            np.max(np.abs(grads["b3"] - l3.bias.grad.numpy())),
        ]
        result["max_pytorch_grad_diff"] = float(max(grad_diffs))
        result["loss_diff_vs_pytorch"] = float(abs(loss - float(loss_pt.item())))

        # 各自执行一步 Adam 更新并对比更新后的参数
        opt_np.step(params_np, grads)
        opt_pt.step()
        param_diffs = [
            np.max(np.abs(params_np["W1"] - l1.weight.detach().numpy().T)),
            np.max(np.abs(params_np["b1"] - l1.bias.detach().numpy())),
            np.max(np.abs(params_np["W2"] - l2.weight.detach().numpy().T)),
            np.max(np.abs(params_np["b2"] - l2.bias.detach().numpy())),
            np.max(np.abs(params_np["W3"] - l3.weight.detach().numpy().T)),
            np.max(np.abs(params_np["b3"] - l3.bias.detach().numpy())),
        ]
        result["max_pytorch_adam_step_diff"] = float(max(param_diffs))

    return result


# =============================================================================
#  三、训练循环：手写 NumPy 版 vs PyTorch 对照版
# =============================================================================
@dataclass
class TrainResult:
    """训练结果记录（供打印、推理与可视化使用）。"""

    impl: str                        # "numpy" 或 "pytorch"
    params: Dict[str, np.ndarray]    # 训练完成后的模型参数
    train_loss_history: List[float]
    val_loss_history: List[float]
    train_acc_history: List[float]
    val_acc_history: List[float]
    final_train_acc: float
    final_val_acc: float
    confusion_matrix: np.ndarray     # (5, 5) 测试集混淆矩阵
    elapsed_sec: float


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """计算 (5, 5) 混淆矩阵，行表示真实类别，列表示预测类别。"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def predict_single_or_batch(
    x_raw: np.ndarray,
    params: Dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对任意原始 5 维向量做完整推理：标准化 -> 5个原始分数 Logits -> Softmax 概率 -> 预测类别。"""
    x_norm = (np.asarray(x_raw, dtype=np.float64) - mean) / std
    logits, _ = forward(x_norm, params)
    probs = softmax(logits)
    pred = np.argmax(probs, axis=-1)
    return logits, probs, pred


def train_numpy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 160,
    batch_size: int = 64,
    lr: float = 0.008,
    seed: int = 42,
    verbose: bool = True,
) -> TrainResult:
    """使用纯手写 NumPy (Forward + CrossEntropy + Backprop + Adam) 训练五分类神经网络。"""
    t0 = time.perf_counter()
    params = init_params(seed=seed)
    optimizer = AdamOptimizer(params, lr=lr, beta1=0.9, beta2=0.999, eps=1e-8)

    rng = np.random.default_rng(seed)
    n_train = X_train.shape[0]

    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []
    train_acc_hist: List[float] = []
    val_acc_hist: List[float] = []

    for epoch in range(1, epochs + 1):
        # 打乱训练集切分 mini-batch
        perm = rng.permutation(n_train)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        for start in range(0, n_train, batch_size):
            xb = X_shuf[start : start + batch_size]
            yb = y_shuf[start : start + batch_size]

            # 1. 前向传播计算 5 个原始分数 logits
            logits_b, cache = forward(xb, params)
            # 2. Cross-Entropy 计算损失与 Softmax 概率
            _, probs_b = cross_entropy_loss(logits_b, yb)
            # 3. Backpropagation 计算每个参数的梯度
            grads = backward(cache, probs_b, yb, params)
            # 4. Adam Optimizer 调整更新步长
            optimizer.step(params, grads)

        # 记录当前 Epoch 全量训练集与验证集指标
        tr_logits, _ = forward(X_train, params)
        tr_loss, tr_probs = cross_entropy_loss(tr_logits, y_train)
        tr_acc = float(np.mean(np.argmax(tr_probs, axis=1) == y_train))

        va_logits, _ = forward(X_val, params)
        va_loss, va_probs = cross_entropy_loss(va_logits, y_val)
        va_acc = float(np.mean(np.argmax(va_probs, axis=1) == y_val))

        train_loss_hist.append(tr_loss)
        val_loss_hist.append(va_loss)
        train_acc_hist.append(tr_acc)
        val_acc_hist.append(va_acc)

        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            print(
                f"  [NumPy 手写版] Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc * 100:5.1f}% | "
                f"Val Loss: {va_loss:.4f} | Val Acc: {va_acc * 100:5.1f}%"
            )

    elapsed = time.perf_counter() - t0
    val_logits, _ = forward(X_val, params)
    val_preds = np.argmax(val_logits, axis=1)
    cm = compute_confusion_matrix(y_val, val_preds)

    return TrainResult(
        impl="numpy",
        params=params,
        train_loss_history=train_loss_hist,
        val_loss_history=val_loss_hist,
        train_acc_history=train_acc_hist,
        val_acc_history=val_acc_hist,
        final_train_acc=train_acc_hist[-1],
        final_val_acc=val_acc_hist[-1],
        confusion_matrix=cm,
        elapsed_sec=elapsed,
    )


def train_pytorch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 160,
    batch_size: int = 64,
    lr: float = 0.008,
    seed: int = 42,
    verbose: bool = True,
) -> TrainResult:
    """使用 PyTorch (nn.Sequential + CrossEntropyLoss + autograd + optim.Adam) 训练对照模型。

    为了直观验证手写 NumPy 实现的正确性，这里使用与 train_numpy 完全相同的初始权重与 mini-batch 序列。
    """
    if not HAVE_TORCH:
        raise RuntimeError("当前环境未安装 PyTorch")

    t0 = time.perf_counter()
    init_p = init_params(seed=seed)

    l1 = nn.Linear(5, 32, dtype=torch.float64)
    l2 = nn.Linear(32, 16, dtype=torch.float64)
    l3 = nn.Linear(16, 5, dtype=torch.float64)
    with torch.no_grad():
        l1.weight.copy_(torch.from_numpy(init_p["W1"].T))
        l1.bias.copy_(torch.from_numpy(init_p["b1"]))
        l2.weight.copy_(torch.from_numpy(init_p["W2"].T))
        l2.bias.copy_(torch.from_numpy(init_p["b2"]))
        l3.weight.copy_(torch.from_numpy(init_p["W3"].T))
        l3.bias.copy_(torch.from_numpy(init_p["b3"]))

    model = nn.Sequential(l1, nn.ReLU(), l2, nn.ReLU(), l3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    rng = np.random.default_rng(seed)
    n_train = X_train.shape[0]

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val).long()

    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []
    train_acc_hist: List[float] = []
    val_acc_hist: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n_train)
        X_shuf = X_train_t[perm]
        y_shuf = y_train_t[perm]

        for start in range(0, n_train, batch_size):
            xb = X_shuf[start : start + batch_size]
            yb = y_shuf[start : start + batch_size]

            optimizer.zero_grad()
            logits_b = model(xb)               # 1. 前向传播得 5 个原始分数
            loss_b = criterion(logits_b, yb)   # 2. Cross-Entropy Loss
            loss_b.backward()                  # 3. Backpropagation 计算梯度
            optimizer.step()                   # 4. Adam 更新参数步长

        model.eval()
        with torch.no_grad():
            tr_logits = model(X_train_t)
            tr_loss = float(criterion(tr_logits, y_train_t).item())
            tr_acc = float((tr_logits.argmax(dim=1) == y_train_t).float().mean().item())

            va_logits = model(X_val_t)
            va_loss = float(criterion(va_logits, y_val_t).item())
            va_acc = float((va_logits.argmax(dim=1) == y_val_t).float().mean().item())

        train_loss_hist.append(tr_loss)
        val_loss_hist.append(va_loss)
        train_acc_hist.append(tr_acc)
        val_acc_hist.append(va_acc)

        if verbose and (epoch == 1 or epoch % 40 == 0 or epoch == epochs):
            print(
                f"  [PyTorch对照版] Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc * 100:5.1f}% | "
                f"Val Loss: {va_loss:.4f} | Val Acc: {va_acc * 100:5.1f}%"
            )

    elapsed = time.perf_counter() - t0
    with torch.no_grad():
        val_preds = model(X_val_t).argmax(dim=1).numpy()
    cm = compute_confusion_matrix(y_val, val_preds)

    trained_params = {
        "W1": l1.weight.detach().numpy().T.copy(),
        "b1": l1.bias.detach().numpy().copy(),
        "W2": l2.weight.detach().numpy().T.copy(),
        "b2": l2.bias.detach().numpy().copy(),
        "W3": l3.weight.detach().numpy().T.copy(),
        "b3": l3.bias.detach().numpy().copy(),
    }

    return TrainResult(
        impl="pytorch",
        params=trained_params,
        train_loss_history=train_loss_hist,
        val_loss_history=val_loss_hist,
        train_acc_history=train_acc_hist,
        val_acc_history=val_acc_hist,
        final_train_acc=train_acc_hist[-1],
        final_val_acc=val_acc_hist[-1],
        confusion_matrix=cm,
        elapsed_sec=elapsed,
    )
