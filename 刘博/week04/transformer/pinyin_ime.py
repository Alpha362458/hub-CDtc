"""
基于 Transformer 语言模型的拼音输入法。
用法:
    python pinyin_ime.py
    python pinyin_ime.py --topk 8 --beam 10

拼音转换 = 拼音表给出"哪些字可能"，语言模型判断"哪句话更像人话"，
束搜索把两者组合起来找全局最优的汉字序列。
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from data import build_unigram_logprob
from model import build_lm


def load_pinyin_map(path):
    """从 JSON 文件加载 {音节: [候选字, ...]} 映射表。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到拼音映射表文件: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def all_segmentations(token, syllables, limit=32):
    """
    枚举 token 的所有合法音节切分，返回切分列表（每项为音节元组）。

    同一个拼音串常有多种切法："fangan" → fang+an(反感) / fan+gan(翻擀)，
    "xian" → 先/县 与 "xi+an" 西安 同理。单路径贪心（永远匹配最长音节）
    只看局部、切错后面全错；这里枚举全部合法切法交给语言模型统一打分，
    歧义由数据决定。limit 防止长输入的切分数指数爆炸。
    """
    results = []

    def dfs(i, parts):
        if len(results) >= limit:
            return
        if i == len(token):
            results.append(tuple(parts))
            return
        # 按长度降序尝试，保证优先产出"长音节优先"的切法
        for s in syllables:
            if token.startswith(s, i):
                parts.append(s)
                dfs(i + len(s), parts)
                parts.pop()

    dfs(0, [])
    return results


def segment(pinyin_str, syllables, limit=32):
    """把拼音串按空格分 token，各 token 枚举切分后做笛卡尔积，
    返回所有「整体切分方案」的列表，如 [["fang","an"], ["fan","gan"]]。
    连续输入（"huangjin"）与带空格（"huang jin"）都支持；
    某个 token 完全切不开时整体跳过该 token。"""
    token_options = []
    for token in pinyin_str.strip().lower().split():
        segs = all_segmentations(token, syllables, limit)
        if segs:
            token_options.append(segs)

    plans = [[]]
    for segs in token_options:
        plans = [p + list(s) for p in plans for s in segs]
    return plans


class PinyinIME:
    """拼音 → 候选汉字的转换引擎：多切分枚举 + 逐字束搜索，供交互 CLI 与评估脚本共用。"""

    def __init__(self, model, char2idx, idx2char, pinyin_map,
                 uni_logprob, device, beam_size=10, max_context=128):
        self.model = model
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.pinyin_map = pinyin_map
        self.uni_logprob = uni_logprob   # 无上文时第一字的字频先验 (V,) log 概率
        self.device = device
        self.beam_size = beam_size
        # Transformer 注意力计算量随长度平方增长，多轮输入后已确认文字无界
        # 增长会让打分越来越慢，只取尾部最近的上文即可
        self.max_context = max_context
        self.syllables = sorted(pinyin_map.keys(), key=len, reverse=True)

    def _next_char_logprobs(self, context):
        """语言模型在 context 下对下一字的 log 概率；空 context 退回字频先验
        （第一字没有上文，模型无法打分，不退回则所有候选同分只能按表序瞎挑）。"""
        if not context:
            return self.uni_logprob
        ids = [self.char2idx[c] for c in context if c in self.char2idx]
        if not ids:
            return self.uni_logprob
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(x)   # (1, T, V)
        return F.log_softmax(logits[0, -1, :], dim=-1)

    def _beam_search(self, syllables, prefix):
        """单个切分方案内的逐字束搜索，返回 [(累计log_prob, 文字), ...]。

        每处理一个音节，把所有存活 beam 与该音节的候选汉字做笛卡尔积展开，
        得分增量为语言模型在「prefix + 已生成部分」下对下一字的 log 概率，
        按总得分截断保留 beam_size 条路径继续。
        """
        beams = [(0.0, "")]  # 初始只有一条空路径，得分为 0

        for syllable in syllables:
            # 只留词表内的候选字（表外字模型打不了分）并去重 —— 拼音表同一字可能出现两次
            candidates = list(dict.fromkeys(
                c for c in self.pinyin_map.get(syllable, []) if c in self.char2idx))
            if not candidates:
                continue  # 该音节无可用候选，跳过（不终止整个搜索）

            new_beams = []
            for score, partial in beams:
                log_probs = self._next_char_logprobs((prefix + partial)[-self.max_context:])
                for char in candidates:
                    new_beams.append((score + log_probs[self.char2idx[char]].item(),
                                      partial + char))

            new_beams.sort(reverse=True)
            beams = new_beams[:self.beam_size]

        return beams

    def convert(self, pinyin_str, prefix=""):
        """转换拼音串为候选列表：枚举所有切分方案，各方案束搜索后合并、
        按文字去重、统一按得分排序。prefix 为已确认上文，参与语言模型打分。"""
        plans = segment(pinyin_str, self.syllables)
        merged = {}
        for plan in plans:
            for score, text in self._beam_search(plan, prefix):
                if text not in merged or score > merged[text]:
                    merged[text] = score
        return [(score, text) for text, score in
                sorted(merged.items(), key=lambda kv: kv[1], reverse=True)]

    def convert_detail(self, pinyin_str):
        """评估用：同时返回切分方案与候选，便于展示「一句话的多种读法」。"""
        plans = segment(pinyin_str, self.syllables)
        return {"pinyin": pinyin_str, "segmentations": plans, "results": self.convert(pinyin_str)}

    def interactive(self, topk):
        """交互主循环：输入拼音 → 展示 topk 候选 → 选中者追加到已确认文字，作为下一轮的上文。"""
        print("=" * 52)
        print("  拼音输入法（Transformer 字符级语言模型）")
        print("  输入拼音回车 → 选候选编号追加到已输入文字")
        print("  r = 重置  q = 退出")
        print("=" * 52)

        confirmed = ""  # 已确认的文字，累积作为语言模型上文

        while True:
            print(f"\n已输入: 「{confirmed}」" if confirmed else "\n已输入: （空）")
            raw = input("拼音> ").strip()

            if not raw:
                continue
            if raw == "q":
                print("退出。")
                break
            if raw == "r":
                confirmed = ""
                continue

            plans = segment(raw, self.syllables)
            if not plans:
                print("无法识别任何音节，请检查拼音拼写。")
                continue

            if len(plans) > 1:
                print(f"识别到 {len(plans)} 种切分: " + " / ".join(
                    " ".join(p) for p in plans[:4]) + ("..." if len(plans) > 4 else ""))

            results = self.convert(raw, prefix=confirmed)
            if not results:
                print("无候选结果。")
                continue

            print("候选:")
            for i, (score, text) in enumerate(results[:topk]):
                print(f"  [{i}] {text}  ({score:.2f})")

            choice = input("选择编号 (回车跳过): ").strip()
            if choice.isdigit():
                idx = int(choice)
                if 0 <= idx < len(results):
                    confirmed += results[idx][1]
                else:
                    print("编号超出范围。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="best_transformer.pt")
    parser.add_argument("--pinyin_map", default="pinyin_map.json", help="拼音映射表 JSON 文件")
    parser.add_argument("--topk",        type=int, default=5,  help="展示候选数")
    parser.add_argument("--beam",        type=int, default=10, help="束搜索宽度")
    parser.add_argument("--max_context", type=int, default=128, help="送入模型的上文截断长度（防长上下文打分变慢）")
    parser.add_argument("--unigram",     default="*.txt", help="统计字频先验用的语料（无上文的第一个字）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt     = torch.load(args.model_path, map_location=device)
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    cfg      = ckpt["args"]

    model = build_lm(
        vocab_size=len(char2idx),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        dropout=0.0,   # 推理阶段关闭 dropout，保证输出确定性
        impl=cfg.get("impl", "official"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"拼音表: {args.pinyin_map}  ({len(load_pinyin_map(args.pinyin_map))} 个音节)")
    print(f"字频先验: {args.unigram}")
    print(f"模型: {args.model_path}  (Transformer, 词表 {len(char2idx)} 字)")

    ime = PinyinIME(
        model=model,
        char2idx=char2idx,
        idx2char=idx2char,
        pinyin_map=load_pinyin_map(args.pinyin_map),
        uni_logprob=build_unigram_logprob(args.unigram, char2idx, device),
        device=device,
        beam_size=args.beam,
        max_context=args.max_context,
    )
    ime.interactive(args.topk)


if __name__ == "__main__":
    main()
