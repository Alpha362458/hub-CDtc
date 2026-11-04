第三周作业总结：
任务：一句话 → 它属于 12 种语言中的哪一种
模型：Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear
损失：CrossEntropyLoss        优化器：Adam
数据：data/v12（在 v8 的 8 语种基础上增加 kor / tha / vie / tur 四个语种）
