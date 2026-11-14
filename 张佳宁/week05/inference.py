"""终端与网页共用的推理逻辑：设备选择、检查点加载、文本续写和候选概率。"""

import math
from pathlib import Path

import torch

from language_model import DEFAULT_CHECKPOINT, LM

DEFAULT_SFT_CHECKPOINT = (
    Path(__file__).resolve().parent / "sft_transformer_encoder_model.pt"
)
PROMPT_TEMPLATE = "问：{instruction}\n答："


def format_prompt(instruction):
    """训练和问答推理共用一个模板，避免格式不一致。"""
    return PROMPT_TEMPLATE.format(instruction=instruction)


def encode_text(text, char2idx):
    """SFT 字表使用 UNK 兜底；原始字表没有 UNK 时保留明确报错。"""
    unk_id = char2idx.get("<UNK>")
    unknown = sorted(set(text) - set(char2idx))
    if unknown and unk_id is None:
        raise ValueError(f"输入包含词表外字符：{unknown}；当前检查点没有 UNK 字符。")
    return [char2idx.get(char, unk_id) for char in text]


def select_device(name="auto"):
    """自动优先选择 CUDA，其次 MPS，最后 CPU；也可手动指定。"""
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境没有可用 CUDA，请选择 auto、cpu 或可用的 mps。")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("当前环境没有可用 MPS，请选择 auto 或 cpu。")
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
        raise ValueError(
            f"找不到模型文件：{path}；请先训练或用 --checkpoint 指定路径。"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "causal_transformer_encoder":
        raise ValueError(
            "该检查点不是当前 TransformerEncoder 模型，不能加载旧 LSTM 或完整 Transformer 权重。"
        )
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
def generate_text(
    model,
    char2idx,
    idx2char,
    prompt,
    length=100,
    temperature=0.8,
    top_k=20,
    *,
    stop_at_newline=False,
    include_prompt=True,
):
    """逐字续写，超过上下文长度时仅保留最近的字符作为输入。"""
    if not prompt.strip():
        raise ValueError("输入文字不能为空。")
    if length < 0 or temperature < 0 or not math.isfinite(temperature) or top_k < 0:
        raise ValueError(
            "length、temperature、top_k 必须非负，temperature 必须为有限数值。"
        )
    model.eval()  # 关闭所有 Dropout；inference_mode 关闭梯度计算。
    device = next(model.parameters()).device
    ids = encode_text(prompt, char2idx)
    generated_ids = []
    max_len = model.position_embed.num_embeddings
    for _ in range(length):
        x = torch.tensor([ids[-max_len:]], dtype=torch.long, device=device)
        scores = model(x)[0, -1]  # 最后一个位置预测下一个字符。
        # 输入仍然可以使用 UNK，但生成答案时禁止选择它。
        unk_id = char2idx.get("<UNK>")
        if unk_id is not None:
            scores[unk_id] = float("-inf")
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
        # 只判断新生成的字符，提示文本本身的换行不会导致提前停止。
        if stop_at_newline and next_id == char2idx.get("\n"):
            break
        ids.append(next_id)
        generated_ids.append(next_id)
    answer = "".join(idx2char[index] for index in generated_ids)
    return (prompt if include_prompt else "") + answer


def generate_answer(
    model, char2idx, idx2char, question, length=100, temperature=0.8, top_k=20
):
    """输入普通问题，自动添加 SFT 提示格式，只返回答案。"""
    if not question.strip():
        raise ValueError("问题不能为空。")
    prompt = format_prompt(question)
    if len(prompt) > model.position_embed.num_embeddings:
        raise ValueError("问题超过模型上下文长度，请缩短问题。")
    return generate_text(
        model,
        char2idx,
        idx2char,
        prompt,
        length,
        temperature,
        top_k,
        stop_at_newline=True,
        include_prompt=False,
    )


@torch.inference_mode()
def next_character_candidates(model, char2idx, idx2char, prompt, count=5):
    """返回下一个字符的原始概率，不应用温度或 Top-K 采样过滤。"""
    if not prompt.strip():
        raise ValueError("输入文字不能为空。")
    if type(count) is not int or count < 1:
        raise ValueError("候选数量必须为正整数。")
    model.eval()
    ids = encode_text(prompt, char2idx)
    context = ids[-model.position_embed.num_embeddings :]
    device = next(model.parameters()).device
    x = torch.tensor([context], dtype=torch.long, device=device)
    probs = torch.softmax(model(x)[0, -1], dim=-1)
    values, indices = probs.topk(min(count, len(char2idx)))
    return [
        {"char": idx2char[index], "probability": value}
        for index, value in zip(indices.tolist(), values.tolist())
    ]
