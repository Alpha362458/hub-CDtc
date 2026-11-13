"""终端与网页共用的推理逻辑：设备选择、检查点加载、文本续写和候选概率。"""

import math
from pathlib import Path

import torch

from language_model import DEFAULT_CHECKPOINT, LM


def select_device(name="auto"):
    """自动优先选择 CUDA，其次 MPS，最后 CPU；也可手动指定。"""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path, device):
    """使用检查点中的结构配置和词表恢复模型，预测时无需重新读取语料。"""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"找不到模型文件：{path}；请先训练或用 --checkpoint 指定路径。")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "causal_transformer_encoder":
        raise ValueError("该检查点不是当前 TransformerEncoder 模型，不能加载旧 LSTM 或完整 Transformer 权重。")
    config = checkpoint["args"]
    model = LM(
        vocab_size=len(checkpoint["char2idx"]),
        embed_dim=config["embed_dim"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        dropout=config["dropout"],
        max_seq_len=config["seq_len"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    print(f"已加载模型：{path.resolve()}")
    return model, checkpoint["char2idx"], checkpoint["idx2char"]


@torch.inference_mode()
def generate_text(model, char2idx, idx2char, prompt, length=100, temperature=0.8, top_k=20):
    """逐字续写，超过上下文长度时仅保留最近的字符作为输入。"""
    if not prompt.strip():
        raise ValueError("输入文字不能为空。")
    unknown = sorted(set(prompt) - set(char2idx))
    if unknown:
        raise ValueError(f"输入包含词表外字符：{unknown}；请更换文字，本模型没有 UNK 字符。")
    if length < 0 or temperature < 0 or not math.isfinite(temperature) or top_k < 0:
        raise ValueError("length、temperature、top_k 必须非负，temperature 必须为有限数值。")
    model.eval()  # 关闭所有 Dropout；inference_mode 关闭梯度计算。
    device = next(model.parameters()).device
    ids = [char2idx[char] for char in prompt]
    max_len = model.position_embed.num_embeddings
    for _ in range(length):
        x = torch.tensor([ids[-max_len:]], dtype=torch.long, device=device)
        scores = model(x)[0, -1]  # 最后一个位置预测下一个字符。
        if temperature == 0:
            next_id = scores.argmax().item()
        else:
            scores = scores / temperature
            if top_k:
                values, indices = torch.topk(scores, min(top_k, scores.numel()))
                selected = torch.multinomial(torch.softmax(values, dim=-1), 1)
                next_id = indices[selected].item()
            else:
                next_id = torch.multinomial(torch.softmax(scores, dim=-1), 1).item()
        ids.append(next_id)
    return "".join(idx2char[index] for index in ids)


@torch.inference_mode()
def next_character_candidates(model, char2idx, idx2char, prompt, count=5):
    """返回下一个字符的原始概率，不应用温度或 Top-K 采样过滤。"""
    if not prompt.strip():
        raise ValueError("输入文字不能为空。")
    unknown = sorted(set(prompt) - set(char2idx))
    if unknown:
        raise ValueError(f"输入包含词表外字符：{unknown}；请更换文字，本模型没有 UNK 字符。")
    if type(count) is not int or count < 1:
        raise ValueError("候选数量必须为正整数。")
    model.eval()
    context = prompt[-model.position_embed.num_embeddings:]
    device = next(model.parameters()).device
    x = torch.tensor([[char2idx[char] for char in context]], device=device)
    probs = torch.softmax(model(x)[0, -1], dim=-1)
    values, indices = probs.topk(min(count, len(char2idx)))
    return [{"char": idx2char[index], "probability": value}
            for index, value in zip(indices.tolist(), values.tolist())]
