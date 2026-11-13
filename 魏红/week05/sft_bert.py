# coding: utf-8
"""
Week05 作业（BERT 预训练版）：同一套 SFT 流程，把「从零训练的小模型」换成「BERT 预训练模型」

和 sft_lm.py 的关系：
    数据（SftDataset）、补齐（BatchCollator）、损失与训练（Trainer，手动错位 + ignore_index=-100）、
    对齐表（LabelInspector）、回答（TextPredictor）、因果性验证（CausalityChecker）——全部直接 import 过来，一行没改。
    只换了两样东西：
        BertVocabulary  字表：从「3000 条数据里的字」换成 bert-base-chinese 自带的字表（21128 个）
        BertCausalLm    模型：从「手写的 TransformerLm」换成「加载好预训练权重的 BERT」

BERT 里面是什么（对照 Week04 学的）：
    Embedding   字表 + 位置表 + Segment 表，三张表逐位相加           → 手写版只有前两张
    12 个 Block  每个 = 多头注意力(12 头 × 64 维) + FFN(768→3072→768)，Post-LN（先残差再 LayerNorm），GELU
                                                                  → 就是 sft_lm.py 里手写的 TransformerBlock，只是 D=768、12 层
    LM 头       Linear(768 → 21128)，每个位置给 21128 个字打分       → 手写版的 _output_layer
    is_decoder=True  给注意力加因果掩码（下三角）                   → 手写版的 causal_mask
    区别只有一个：这些权重是 bert-base-chinese 在大量中文上预训练好的（做的是完形填空），不是随机的。

注意：它预训练的目标是「完形填空」，不是「预测下一个字」，所以仍然要靠 SFT 教它接龙和回答。

运行（在本文件所在目录；不要加 python -I，否则找不到 sft_lm.py）：
    python sft_bert.py --check               # 看 BERT 的结构、对齐表、验证因果掩码
    python sft_bert.py --overfit             # 先在 32 条上过拟合
    python sft_bert.py                       # 正式训练 + 独立验证集评估
    python sft_bert.py --generate "问题"     # 用训练好的模型回答
"""

import argparse
import os
import time
from dataclasses import replace
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertLMHeadModel, BertTokenizerFast

from sft_lm import (BatchCollator, CausalityChecker, Config, LabelInspector, SftDataset, SftRepository,
                    SpecialTokens, TextPredictor, Trainer)


def find_model_dir() -> str:
    """优先用本机已缓存的 bert-base-chinese，找不到就用模型名（由 HuggingFace 去下载）。"""
    root = "D:/app/CourseNotes/models/huggingface/hub/models--bert-base-chinese/snapshots"
    if os.path.isdir(root) and os.listdir(root):
        return os.path.join(root, os.listdir(root)[0])
    return "bert-base-chinese"


class BertVocabulary:
    """
    <summary>让 BERT 的字表，长得和 CharVocabulary 一样（encode / to_id / to_char / size），这样 sft_lm.py 里的类不用改。</summary>
    四个特殊符号借 BERT 词表里已有的：<pad>→[PAD]，<user>→[unused1]，<assistant>→[unused2]，<eos>→[SEP]。
    [unusedN] 是 BERT 词表里预留的空位，预训练时没用过，拿来当新标记最合适。
    """
    _ALIAS: Dict[str, str] = {SpecialTokens.PAD: "[PAD]", SpecialTokens.USER: "[unused1]",
                              SpecialTokens.ASSISTANT: "[unused2]", SpecialTokens.EOS: "[SEP]"}

    def __init__(self, tokenizer: BertTokenizerFast) -> None:
        self._tokenizer = tokenizer

    @property
    def size(self) -> int:
        return len(self._tokenizer)

    def encode(self, text: str) -> List[int]:
        # add_special_tokens=False：不自动加首尾的 [CLS]/[SEP]，我们自己控制。中文基本一字一个 token
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def to_id(self, token: str) -> int:
        return self._tokenizer.convert_tokens_to_ids(self._ALIAS.get(token, token))

    def to_char(self, index: int) -> str:
        token = self._tokenizer.convert_ids_to_tokens(index)
        return token[2:] if token.startswith("##") else token      # "##4" 是接在前一块后面的子词，去掉标记


class BertCausalLm(nn.Module):
    """
    <summary>加载预训练好的 BERT，加因果掩码，输出每个位置的字表分数。接口和手写的 TransformerLm 一样：model(x) → logits (B, L, V)。</summary>
    """

    def __init__(self, model_dir: str) -> None:
        super().__init__()
        # from_pretrained：把预训练好的权重装进来。is_decoder=True：注意力自动带上下三角的因果掩码。
        self._bert = BertLMHeadModel.from_pretrained(model_dir, is_decoder=True)

    def describe(self) -> str:
        c = self._bert.config
        return (f"层数 {c.num_hidden_layers}，隐藏维度 {c.hidden_size}，头数 {c.num_attention_heads}（每头 {c.hidden_size // c.num_attention_heads}），"
                f"FFN 中间层 {c.intermediate_size}，字表 {c.vocab_size}，最长位置 {c.max_position_embeddings}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 右侧补 <pad> 时不用传 attention_mask：有因果掩码，前面的真实字本来就看不到后面的补齐位
        return self._bert(input_ids=x).logits


class Program:
    @staticmethod
    def show_replies(predictor: TextPredictor, rows: List[Dict[str, str]], count: int) -> None:
        for row in rows[:count]:
            answer, eos = predictor.reply(row["instruction"])
            print(f"  问题：{row['instruction']}")
            print(f"  标准：{row['output']}")
            print(f"  模型：{answer}   [{'遇到 <eos> 停止' if eos else '到最长长度才停'}]\n")

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description="BERT 预训练模型 + SFT")
        parser.add_argument("--check", action="store_true")
        parser.add_argument("--overfit", action="store_true")
        parser.add_argument("--generate", default=None)
        args = parser.parse_args()

        here = os.path.dirname(os.path.abspath(__file__))
        # BERT 微调：学习率要小得多（3e-5，不是从零训练的 1e-3），轮数少（4 轮）
        config = Config(train_path=os.path.join(here, "data", "sft_train.jsonl"),
                        val_path=os.path.join(here, "data", "sft_val.jsonl"),
                        save_path=os.path.join(here, "checkpoints", "sft_bert.pt"),
                        epochs=4, batch_size=16, learning_rate=3e-5)
        torch.manual_seed(config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = find_model_dir()
        vocab = BertVocabulary(BertTokenizerFast.from_pretrained(model_dir))
        model = BertCausalLm(model_dir).to(device)

        if args.generate:
            model.load_state_dict(torch.load(config.save_path, map_location=device))
            print(TextPredictor(model, vocab, config.seq_len, device).reply(args.generate)[0])
            return

        train_rows = SftRepository.load(config.train_path)
        val_rows = SftRepository.load(config.val_path)
        if args.overfit:
            train_rows = train_rows[:32]
            val_rows = train_rows
            config = replace(config, epochs=40, batch_size=8, learning_rate=1e-4)
        print(f"device: {device}  训练 {len(train_rows)} 条 / 验证 {len(val_rows)} 条")
        print(f"BERT：{model.describe()}")
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}（手写版 2.68M）")

        train_set = SftDataset(train_rows, vocab, config.seq_len)
        val_set = SftDataset(val_rows, vocab, config.seq_len)
        collator = BatchCollator(vocab.to_id(SpecialTokens.PAD))
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, collate_fn=collator)

        print("\n── labels 对齐表（第 1 条训练样本）──")
        LabelInspector.show(vocab, *train_set[0])
        change = CausalityChecker.max_change_before_last(model, vocab.size, config.seq_len, device)
        print(f"\n── 因果性验证 ──\n改最后一个字后，前 {config.seq_len - 1} 个位置输出的最大变化 = {change:.3e}（应为 0）")
        if args.check:
            return

        trainer = Trainer(model, config, device)
        predictor = TextPredictor(model, vocab, config.seq_len, device)
        start_loss, start_ppl = trainer.run_epoch(val_loader, train=False)
        print(f"\n训练前验证集  Loss {start_loss:.4f}  PPL {start_ppl:.2f}")
        Program.show_replies(predictor, val_rows, 2)

        os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
        best = float("inf")
        print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  {'Val PPL':>9}  {'用时':>5}")
        for epoch in range(1, config.epochs + 1):
            start = time.time()
            train_loss, _ = trainer.run_epoch(train_loader, train=True)
            val_loss, val_ppl = trainer.run_epoch(val_loader, train=False)
            marker = ""
            if val_loss < best:
                best = val_loss
                marker = "  *"
                torch.save(model.state_dict(), config.save_path)
            if not args.overfit or epoch % 10 == 0:
                print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>9.4f}  {val_ppl:>9.2f}  {time.time() - start:>4.0f}s{marker}")

        model.load_state_dict(torch.load(config.save_path, map_location=device))
        print("\n── 验证集里的问题（模型没见过）──" if not args.overfit else "\n── 这 32 条（训练时见过）──")
        Program.show_replies(predictor, val_rows, 5)
        if not args.overfit:
            print("── 训练集里的问题（开卷）──")
            Program.show_replies(predictor, train_rows, 3)
        exact = sum(predictor.reply(r["instruction"])[0] == "".join(vocab.to_char(i) for i in vocab.encode(r["output"]))
                    for r in val_rows)
        print(f"回答与标准答案逐字一致：{exact}/{len(val_rows)}")


if __name__ == "__main__":
    Program.main()
