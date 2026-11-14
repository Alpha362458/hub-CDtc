import os
import torch
import torch.nn as nn
import argparse
from language_model_train import DIY_Transformer_Network

def auto_generate(model, input_ids, topp, eos_id, device):
    """
    使用 top-p sampling 生成后续文本。
    input_ids: 输入文本的 token id 列表，形状为 (1, L)
    topp: 累积概率阈值
    beam: 束搜索宽度
    """
    total_ids = input_ids.copy()  # 保存完整的输入 + 生成序列
    with torch.no_grad():
        for _ in range(100):  # 最多生成 100 个字符
            x = torch.tensor([total_ids], dtype=torch.long, device=device)  # (1, L)
            logits = model(x)  # (1, L, vocab_size)
            logits = logits[0, -1, :]  # 取最后一个时间步的输出，(vocab_size,)
            probs = torch.softmax(logits, dim=-1)

            # top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            topp_tensor = torch.tensor(topp, device=probs.device)
            cutoff_index = torch.searchsorted(cumulative_probs, topp_tensor).item()
            top_indices = sorted_indices[:cutoff_index + 1]
            top_probs = sorted_probs[:cutoff_index + 1]
            top_probs /= top_probs.sum()  # 归一化

            next_token = torch.multinomial(top_probs, num_samples=1).item()
            total_ids.append(top_indices[next_token].item())

            if next_token == eos_id:  # 遇到 EOS token 停止生成
                break

    return total_ids[len(input_ids):]  # 返回新生成的部分


#根据输入文本，一直输入模型预测输出文本，直到达到指定长度或遇到 EOS token
def run(model, char2idx, idx2char, topp, device):
    print("=" * 52)
    print("  语言模型文本生成")
    print("  输入一段文字回车 → 模型继续生成文本")
    print("  q = 退出")
    print("=" * 52)

    while True:
        raw = input("文字> ").strip()

        if not raw:
            continue
        if raw == "q":
            print("退出。")
            break

        # 将已确认文字 + 新输入文字编码为 token id
        input_ids = [char2idx[c] for c in raw if c in char2idx]

        # 使用 top-p sampling 生成后续文本
        output_ids = auto_generate(model, input_ids, topp, char2idx["<eos>"], device)

        output_text = "".join(idx2char[i] for i in output_ids)
        print(f"生成结果: 「{output_text}」")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                "--model_path",
                type=str,
                default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt"),
            )
    parser.add_argument("--topp", type=float, default=0.9,  help="累积概率")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    # 从 checkpoint 中恢复词表和模型超参
    ckpt     = torch.load(args.model_path, map_location=device)
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    cfg      = ckpt["args"]

    model = DIY_Transformer_Network(
        vocab_size = len(char2idx),
        embed_dim  = cfg["embed_dim"],
        num_heads = cfg["num_heads"],
        num_layers = cfg["num_layers"],
        dropout    = 0.0,   # 推理阶段关闭 dropout，保证输出确定性
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"模型: {args.model_path}  (词表 {len(char2idx)} 字)")
    run(model, char2idx, idx2char, args.topp, device)


if __name__ == "__main__":
    main()