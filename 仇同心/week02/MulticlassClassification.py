# coding:utf8
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
多分类任务：5维向量，最大值出现在第几维，就属于第几类（类别：0,1,2,3,4）
"""


class TorchModel(nn.Module):
    # 构造函数：输入维度、隐藏层维度、类别数
    def __init__(self, input_size, hidden_size, class_num):
        # 调用父类构造方法，完成参数注册等内部初始化
        super(TorchModel, self).__init__()
        # 隐藏层+ReLU引入非线性，加快收敛；
        # argmax规律理论上单层线性(w_j=e_j)也能表示，但拟合慢、效果不稳
        # 第一层全连接：5 -> 32，自动创建并注册 weight/bias 参数
        self.layer1 = nn.Linear(input_size, hidden_size)
        # ReLU 激活函数：max(0, x)，引入非线性变换
        self.activation = nn.ReLU()
        # 第二层全连接：32 -> 5，输出各类别的 logits（未归一化得分）
        self.layer2 = nn.Linear(hidden_size, class_num)
        # 多分类交叉熵损失（内部包含softmax）
        # 交叉熵损失，内部 = LogSoftmax + NLLLoss，要求输入为原始 logits
        self.loss = nn.CrossEntropyLoss()

    # forward：有标签返回loss；无标签返回预测logits
    # 前向传播，y 默认为 None，用于区分训练与预测两种模式
    def forward(self, x, y=None):
        # 第一层线性变换：[batch_size, 5] -> [batch_size, 32]
        x = self.layer1(x)
        # ReLU 非线性变换，形状不变，仍为 [batch_size, 32]
        x = self.activation(x)
        # 第二层线性变换：[batch_size, 32] -> [batch_size, 5]，得到各类得分
        logits = self.layer2(x)
        # 若传入了真实标签，说明处于训练阶段
        if y is not None:
            # 用 logits 与标签计算交叉熵损失并返回
            return self.loss(logits, y)
        # 未传标签，说明处于预测阶段
        else:
            # 直接返回各类别的 logits 得分
            return logits


# 生成单个样本
def build_sample():
    # 随机生成 5 个 [0,1) 之间的数，构成一个 5 维向量
    x = np.random.random(5)
    # 最大值下标就是类别
    # 取最大值所在的下标(0~4)，作为该样本的类别标签
    label = np.argmax(x)
    # 返回特征向量和对应的类别标签
    return x, label


# 批量生成数据集
# 参数：需要生成的样本总数
def build_dataset(total_sample_num):
    # 存放所有样本特征的列表
    X = []
    # 存放所有样本标签的列表
    Y = []
    # 循环生成指定数量的样本
    for i in range(total_sample_num):
        # 调用单样本生成函数，得到一个特征向量和标签
        x, y = build_sample()
        # 把特征向量追加进 X
        X.append(x)
        # 把标签追加进 Y
        Y.append(y)
    # X: [N,5] float；Y: [N] long（交叉熵要求标签是long型一维）
    # 转成张量：特征用 float 型，标签用 long 型一维
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 评估函数，计算准确率
# 参数：待评估的模型
def evaluate(model):
    # 切换为评估模式（本模型无 Dropout/BN，主要起规范作用）
    model.eval()
    # 评估集样本数量
    test_sample_num = 500
    # 随机新生成 500 个样本作为测试集
    x, y = build_dataset(test_sample_num)
    # 记录预测正确的个数
    correct = 0
    # 关闭梯度计算，节省显存并加快推理
    with torch.no_grad():
        # 只传 x，forward 返回各类 logits：[500, 5]
        logits = model(x)
        # logits最大值下标 = 预测类别
        # 沿类别维取最大值下标，得到每个样本的预测类别：[500]
        y_pred = torch.argmax(logits, dim=1)
        # 预测与真实标签逐位比较，求和得到正确个数
        correct = int(torch.sum(y_pred == y))
    # 正确数除以总数得到准确率
    acc = correct / test_sample_num
    # 打印本轮评估结果
    print(f"正确预测个数：{correct}, 正确率：{acc:.4f}")
    # 返回准确率供主流程记录
    return acc


def main():
    # 训练总轮数
    epoch_num = 20
    # 每个 batch 的样本数
    batch_size = 100
    # 训练集样本总数
    train_sample = 50000
    # 输入向量维度
    input_size = 5
    # 隐藏层神经元个数
    hidden_size = 32
    # 类别数量（最大值可能出现在 5 个位置）
    class_num = 5
    # 学习率，控制参数更新的步长
    learning_rate = 0.01

    # 实例化模型
    model = TorchModel(input_size, hidden_size, class_num)
    # 用 Adam 优化器管理模型全部可训练参数
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 记录每轮的 [准确率, 平均loss]，用于最后绘图
    log = []

    # 一次性生成固定的训练集
    train_x, train_y = build_dataset(train_sample)

    # 外层循环：遍历每一轮训练
    for epoch in range(epoch_num):
        # 切换为训练模式
        model.train()
        # 收集本轮所有 batch 的 loss
        watch_loss = []
        # 分批
        # 内层循环：把训练集切成若干 batch 依次训练
        for batch_index in range(train_sample // batch_size):
            # 当前 batch 的起始下标
            start = batch_index * batch_size
            # 当前 batch 的结束下标（不含）
            end = start + batch_size
            # 取出当前 batch 的特征 [100, 5]
            x = train_x[start:end]
            # 取出当前 batch 的标签 [100]
            y = train_y[start:end]

            # 前向传播，传入标签返回损失值
            loss = model(x, y)
            # 反向传播，自动计算各参数的梯度
            loss.backward()
            # 优化器按梯度更新模型参数
            optim.step()
            # 清空梯度，防止下一轮累加
            optim.zero_grad()
            # 取出 loss 的数值存入列表
            watch_loss.append(loss.item())

        # 计算本轮所有 batch 的平均损失
        mean_loss = np.mean(watch_loss)
        # 打印本轮平均损失
        print(f"=========\n第{epoch + 1}轮平均loss:{mean_loss:.4f}")
        # 用新生成的样本测试本轮模型准确率
        acc = evaluate(model)
        # 把本轮准确率和损失记入日志
        log.append([acc, mean_loss])

    # 保存模型权重
    # 只保存参数字典，不含网络结构
    torch.save(model.state_dict(), "multi_class_model.bin")
    # 绘图
    # 横轴为轮数，纵轴取每轮准确率画曲线
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    # 同一张图上叠加每轮平均损失曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    # 显示图例，区分 acc 与 loss 两条线
    plt.legend()
    # 弹出窗口显示图像
    plt.show()
    # 主函数结束
    return


# 参数：模型权重路径、待预测向量列表
def predict(model_path, input_vec):
    # 必须与训练时保持一致的输入维度
    input_size = 5
    # 必须与训练时保持一致的隐藏层维度
    hidden_size = 32
    # 必须与训练时保持一致的类别数
    class_num = 5
    # 先构建与训练时结构完全相同的模型
    model = TorchModel(input_size, hidden_size, class_num)
    # 加载训练好的权重，weights_only=True 更安全
    model.load_state_dict(torch.load(model_path, weights_only=True))
    # 切换为评估模式
    model.eval()

    # 推理阶段关闭梯度计算
    with torch.no_grad():
        # 把输入转成 float 张量并前向传播得到 logits
        logits = model(torch.FloatTensor(input_vec))
        # 预测类别
        # 取最大值下标作为预测类别
        pred_class = torch.argmax(logits, dim=1)
        # softmax得到各类概率
        # 对 logits 做 softmax 得到归一化后的各类概率
        prob = torch.softmax(logits, dim=1)

    # 同时遍历输入向量、预测类别、概率分布
    for vec, pc, p in zip(input_vec, pred_class, prob):
        # 打印输入向量
        print(f"输入向量：{vec}")
        # 打印预测类别及 5 个类别的概率
        print(f"预测类别：{int(pc)}，各类概率：{p.numpy()}\n")


# 仅当直接运行本文件时才执行，被 import 时不执行
if __name__ == "__main__":
    # 执行训练流程并保存模型
    main()
    test_vec = [
        [0.1, 0.8, 0.3, 0.5, 0.2],
        [0.9, 0.2, 0.4, 0.1, 0.7],
        [0.2, 0.1, 0.7, 0.4, 0.3],
        [0.1, 0.2, 0.3, 0.4, 0.6]
    ]
    # 加载训练好的模型对测试向量做预测
    predict("multi_class_model.bin", test_vec)
