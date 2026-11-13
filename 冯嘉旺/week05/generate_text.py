"""使用 week05 的 SFT 检查点进行单轮指令回答。"""

import argparse
import math

import torch

from sft_model import EOS, LM, PAD, ROOT, UNK, encode, format_prompt, get_device


@torch.no_grad()
def generate(model, vocab, instruction, device, max_new_tokens=80, temperature=0.7):
    model.eval()
    ids = encode(format_prompt(instruction), vocab)
    context_length = model.position_embed.num_embeddings
    if len(ids) > context_length:
        raise ValueError(f"指令模板长度 {len(ids)} 超过上下文 {context_length}，请缩短指令")
    idx2char = {index: char for char, index in vocab.items()}
    answer = []
    for _ in range(max_new_tokens):
        if len(ids) > context_length:
            return "".join(answer), "达到上下文长度"
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x)[0, -1].clone()
        # PAD/UNK 只用于数据处理，不作为可见回答输出。
        logits[vocab[PAD]] = float("-inf")
        logits[vocab[UNK]] = float("-inf")
        if temperature == 0:
            next_id = logits.argmax().item()
        else:
            next_id = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).item()
        if next_id == vocab[EOS]:
            return "".join(answer), "EOS"
        ids.append(next_id)
        answer.append(idx2char[next_id])
    return "".join(answer), "达到生成字符上限"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "best_model_sft.pt"))
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7, help="0=贪婪解码")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if not args.instruction.strip():
        parser.error("指令不能为空")
    if args.max_new_tokens <= 0 or not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("生成字符上限必须为正整数，温度必须是非负有限数")
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "char_transformer_sft_v1":
        parser.error("请加载 week05 的 SFT 检查点，而不是 week04 的预训练权重")
    vocab = checkpoint["char2idx"]
    device = get_device(args.device)
    model = LM(vocab_size=len(vocab), **checkpoint["args"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    unknown = sum(char not in vocab for char in format_prompt(args.instruction))
    if unknown:
        print(f"提示：指令中有 {unknown} 个词表外字符，将映射为 UNK。")
    answer, reason = generate(
        model, vocab, args.instruction, device, args.max_new_tokens, args.temperature,
    )
    print(f"用户：{args.instruction}\n助手：{answer}\n[停止原因：{reason}]")


if __name__ == "__main__":
    main()
