# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，哪一维的数字最大，就属于第几类（类别号 = 最大值所在维度的下标，共5类）

"""


class TorchModel(nn.Module):
    """五分类模型：输入5维向量，输出5个类别的得分，得分最高的维度下标即预测类别"""

    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层，每个类别对应一个输出得分
        self.loss = nn.functional.cross_entropy  # 多分类交叉熵损失，内部自带 log_softmax

    # 当输入真实标签，返回loss值；无真实标签，返回预测得分
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(y_pred, y)  # 预测得分和真实类别计算损失
        else:
            return y_pred  # 输出每个类别的得分


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，标签为最大值所在维度的下标（0~4）
def build_sample():
    x = np.random.random(5)
    return x, int(np.argmax(x))


# 随机生成一批样本，5个类别天然均匀（每一维成为最大值的概率相同）
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 交叉熵要求标签是长整型类别下标


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)，形状为(样本数, 5)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):  # 得分最高的维度即预测类别
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 计数训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数，等于最大值可能落进的维度个数
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入  train_x[0:20]  train_y[0:20]  train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size, 5)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 每行是5个类别的得分
        prob = torch.softmax(result, dim=1)  # 得分转成概率分布，只为展示用
    for vec, res, p in zip(input_vec, result, prob):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, int(torch.argmax(res)), float(p.max())))


if __name__ == "__main__":
    main()
    # test_vec = [[0.81889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.39349776,0.59416669,0.92579291,0.41567412,0.8358894]]
    # predict("model.bin", test_vec)
