# -*- coding: utf-8 -*-
"""
任务：输入一个五维向量，预测最大值所在的位置，共五个类别。
程序内部使用编号0～4；显示“第几维”时加1，得到1～5。
并列最大时，约定选择最先出现的位置，与torch.argmax的规则一致。
"""

from pathlib import Path  # 处理文件路径，让保存位置由脚本位置决定。
import torch  # 提供张量计算和自动求导。
from torch import nn  # 提供Linear、ReLU和损失函数。


def main():  
    # 第1步：设置实验参数。它们是人为选择的设置，不是训练得到的权重。
    torch.manual_seed(7)  # 在创建随机数据和模型前设种子，方便重复实验。
    train_count = 2000  # 用2000条样本训练。
    test_count = 500  # 另用500条新样本做最终测试。
    max_epochs = 1000  # 最多执行1000轮，防止一直达不到损失目标。
    target_loss = 0.1  # 本练习尝试把训练交叉熵降到0.1或以下。
    learning_rate = 0.1  # SGD学习率，用来缩放梯度更新的幅度。

    # 第2步：准备训练数据。每行是一道题，每行五个数。
    train_x = torch.randn(train_count, 5)  # [2000,5]，生成包含正负数的随机输入。
    train_y = train_x.argmax(dim=1)  # [2000]，每行最大值的位置就是正确类别。
    # argmax返回long整数标签，取值为0、1、2、3、4，正好符合交叉熵要求。

    # 第3步：准备独立测试数据。训练循环不会使用test_x和test_y更新参数。
    test_x = torch.randn(test_count, 5)  # [500,5]，重新生成的输入样本。
    test_y = test_x.argmax(dim=1)  # [500]，只用来在最后核对测试预测。

    # 第4步：建立网络。B代表当前一起处理的样本数量。
    model = nn.Sequential(  # 按下面的先后顺序执行计算。
        nn.Linear(5, 32),  # [B,5] -> [B,32]，用32套权重计算中间值。
        nn.ReLU(),  # [B,32] -> [B,32]，把负值改为0，保留32个位置。
        nn.Linear(32, 5),  # [B,32] -> [B,5]，输出五个类别的原始分数logits。
    )  # 末尾保留原始分数，CrossEntropyLoss直接接收它们。

    # 第5步：准备计算损失和更新参数的两个工具。
    criterion = nn.CrossEntropyLoss()  # 根据原始分数与正确编号计算平均交叉熵。
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 管理并更新模型参数。
    print("训练样本数：", train_count, "测试样本数：", test_count)

    # 第6步：循环训练。本例每轮使用全部训练样本，只更新一次参数。
    for epoch in range(1, max_epochs + 1):  # epoch依次为1～1000。
        model.train()  # 告诉模型当前处于训练阶段。
        optimizer.zero_grad()  # 清除旧梯度记录，保留已有权重和偏置。
        logits = model(train_x)  # 用当前参数计算2000条样本的五类分数：[2000,5]。
        loss = criterion(logits, train_y)  # 用[2000]的正确类别计算一个标量损失。
        loss.backward()  # 用链式法则求出各个权重和偏置的梯度，尚未改参数。
        optimizer.step()  # 执行“参数 -= 学习率*梯度”，真正更新参数。

        # 参数已经改变，重新计算一次，取得这次更新之后的训练损失。
        with torch.no_grad():  # 这次计算只做监控，不需要记录求导关系。
            updated_logits = model(train_x)  # 使用刚更新好的参数重新计算分数。
            updated_loss = criterion(updated_logits, train_y).item()  # 读取普通数字。

        if epoch == 1 or epoch % 100 == 0:  # 第1轮及之后每100轮打印一次。
            print(f"第{epoch}轮，更新后的训练损失：{updated_loss:.6f}")

        if updated_loss <= target_loss:  # 检查更新后的参数是否达到本次练习的目标。
            print(f"达到目标损失，于第{epoch}轮停止，loss={updated_loss:.6f}")
            break  # 立即结束for循环，不必训练满最大轮数。
    else:  # 只有循环正常跑满、没有执行break时，才进入这个else。
        print(f"已达到最大轮数{max_epochs}，结束训练，loss={updated_loss:.6f}")

    # 第7步：用从未参与参数更新的测试集评价模型。
    model.eval()  # 切换评估模式；本例Linear/ReLU没有模式差异，保留规范写法。
    with torch.no_grad():  # 评估只计算结果，不记录梯度。with表示在下面的代码段中临时关闭梯度记录，离开with后恢复原状。
        test_logits = model(test_x)  # [500,5]，每个新样本得到五个分数。
        test_loss = criterion(test_logits, test_y).item()  # 测试集平均交叉熵。
        test_pred = test_logits.argmax(dim=1)  # [500]，每行选最高分类分数的位置。
    correct = (test_pred == test_y).sum().item()  # 对应位置比较，统计预测正确数量。
    accuracy = correct / test_count  # 正确数量除以测试样本总数。
    print(f"测试损失：{test_loss:.6f}")
    print(f"测试正确数：{correct}/{test_count}，测试准确率：{accuracy:.2%}")

    # 第8步：观察各类分别被预测成什么，避免只看一个总准确率。
    confusion = torch.zeros(5, 5, dtype=torch.long)  # 五行五列的计数表，初始全为0。
    for actual, predicted in zip(test_y.tolist(), test_pred.tolist()):  # 对齐真实与预测编号。
        confusion[actual, predicted] += 1  # 行=真实类别，列=预测类别，对应格子加1。
    print("混淆矩阵：行是真实类别0～4，列是预测类别0～4。")
    print(confusion)  # 对角线是各类别预测正确的数量，整张表总和应为500。

    # 第9步：保存权重。使用独立目录和文件名，避免覆盖课程自带的model.bin。
    output_dir = Path(__file__).resolve().parent / "max5_outputs"  # 脚本旁的输出文件夹。
    output_dir.mkdir(parents=True, exist_ok=True)  # 不存在则创建，已存在则继续使用。
    model_path = output_dir / "max5_model.pt"  # 定义本练习的模型权重文件路径。
    torch.save(model.state_dict(), model_path)  # state_dict保存各层权重和偏置数值。
    print("模型权重已保存到：", model_path)

    # 第10步：重新建立相同结构，并从文件加载权重，验证保存结果能使用。
    loaded_model = nn.Sequential(  # 网络的层顺序与尺寸必须和训练时一致。
        nn.Linear(5, 32),  # 第一层结构相同。
        nn.ReLU(),  # 激活方式相同。
        nn.Linear(32, 5),  # 输出类别数相同。
    )
    weights = torch.load(model_path, map_location="cpu", weights_only=True)  # 读取本脚本保存的权重。
    loaded_model.load_state_dict(weights)  # 用保存的参数替换新模型刚初始化的参数。
    loaded_model.eval()  # 加载后切换到评估模式。

    # 第11步：自己输入几道新题，使用“从文件恢复的模型”作答。
    new_x = torch.tensor([
        [1.8, 0.2, -0.5, 0.7, -1.0],  # 正确编号0，对外显示第1维。
        [0.2, 1.8, -0.5, 0.7, -1.0],  # 正确编号1，对外显示第2维。
        [0.2, -0.5, 1.8, 0.7, -1.0],  # 正确编号2，对外显示第3维。
        [0.2, -0.5, 0.7, 1.8, -1.0],  # 正确编号3，对外显示第4维。
        [0.2, -0.5, 0.7, -1.0, 1.8],  # 正确编号4，对外显示第5维。
    ], dtype=torch.float32)  # 一批五个样本，每个样本五个浮点数。
    with torch.no_grad():  # 预测期间只计算结果。
        new_scores = loaded_model(new_x)  # 模型生成五个类别分数，形状[5,5]。
        new_pred = new_scores.argmax(dim=1)  # 根据模型分数选择预测编号。
    new_true = new_x.argmax(dim=1)  # 直接检查输入，生成标准答案，仅用于核对。
    for vector, predicted, actual in zip(new_x.tolist(), new_pred.tolist(), new_true.tolist()):  # 逐样本显示结果。
        shown_vector = [round(value, 3) for value in vector]  # 只简化小数显示，不改预测使用的数据。
        print(f"输入：{shown_vector}，模型预测：第{predicted + 1}维，正确答案：第{actual + 1}维")


if __name__ == "__main__":  # 直接运行这个文件时才训练；被import时不自动执行main。
    main()  # 从第1步开始运行整个实验。
