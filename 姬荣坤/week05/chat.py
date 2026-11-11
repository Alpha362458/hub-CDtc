import torch
import torch.nn as nn
import torch.nn.functional as F

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout, nhead=8):
        super().__init__()
        self.model_type = model_type
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        
        hidden_dim = hidden_dim * 4  
        self.pos_embedding = nn.Embedding(512, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim, 
            dropout=dropout, batch_first=True, norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz: int, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        e = self.drop(self.embed(x))
        B, T, E = e.size()
        positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        e = e + self.pos_embedding(positions)
        attn_mask = self.generate_square_subsequent_mask(T, x.device)
        out = self.transformer(e, mask=attn_mask)
        return self.fc(self.drop(out))

@torch.no_grad()
def generate_text_top_p(model, prompt, char2idx, idx2char, device, max_new_tokens=50, temperature=0.7, top_p=0.9):
    """
    使用 Top-P 采样生成文本
    """
    model.eval()
    bos_id = char2idx["<bos>"]
    eos_id = char2idx["<eos>"]
    pad_id = char2idx["<pad>"]
    sep_id = char2idx.get("。", pad_id)
    
    # 构造输入： <bos> + prompt + 。
    # 注意：这里不再随便删除未知词，而是映射为 <unk> 或 <pad>
    input_ids = [bos_id] + [char2idx.get(c, pad_id) for c in prompt] + [sep_id]
    generated_ids = input_ids.copy()

    for _ in range(max_new_tokens):
        current_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)
        logits = model(current_tensor)
        
        # 取最后一个时间步的 logits
        next_token_logits = logits[0, -1, :].clone()
        
        # 屏蔽特殊 token 的输出
        next_token_logits[pad_id] = -float('inf')
        next_token_logits[bos_id] = -float('inf')
        
        # 1. 应用温度系数 (Temperature)
        next_token_logits = next_token_logits / temperature
        
        # 2. 计算概率分布
        probs = F.softmax(next_token_logits, dim=-1)
        
        # 3. Top-P 截断 (Nucleus Sampling)
        # 对概率进行降序排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过 top_p 的位置（保留这些位置，后面的全部丢弃）
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将第一个超过 top_p 的位置也保留（保证至少有一个词被选中）
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 把不要的词概率设为 0
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0.0
        
        # 4. 重新归一化概率并进行采样
        if probs.sum() == 0:
            break # 防止极端情况
            
        probs = probs / probs.sum()
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        
        if next_token_id == eos_id:
            break
            
        generated_ids.append(next_token_id)

    # 提取生成的部分
    output_ids = generated_ids[len(input_ids):]
    output_text = "".join([idx2char.get(i, "") for i in output_ids if idx2char.get(i, "") not in ["<pad>", "<bos>", "<eos>"]])
    return output_text

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        ckpt = torch.load("best_model_sft.pt", map_location=device)
    except FileNotFoundError:
        print("未找到 best_model_sft.pt，请先运行 sft_train.py")
        return

    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    state_dict = ckpt["model_state"]
    
    # 自动推断模型维度
    vocab_size = state_dict["embed.weight"].shape[0]
    embed_dim = state_dict["embed.weight"].shape[1]
    num_layers = max(int(k.split('.')[2]) for k in state_dict if "transformer.layers" in k and "weight" in k) + 1
    dim_feedforward = state_dict["transformer.layers.0.linear1.weight"].shape[0]
    hidden_dim = dim_feedforward // 4
    
    model = LM(
        vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, 
        num_layers=num_layers, model_type="transformer", dropout=0.0
    ).to(device)
    
    model.load_state_dict(state_dict)
    
    print("========================================")
    print("模型加载成功！(Top-P 采样版)")
    print("你可以通过在输入时调整参数，例如：")
    print("输入 'set temp 0.5' 修改温度 (越低越保守)")
    print("输入 'set top_p 0.8' 修改采样阈值")
    print("========================================")
    
    # 默认参数
    temperature = 0.7
    top_p = 0.9

    while True:
        try:
            user_input = input("\n👨‍💻 你: ")
            if user_input.lower() in ["quit", "exit"]: 
                break
            if not user_input.strip(): 
                continue
                
            # 允许在对话中动态调整参数
            if user_input.startswith("set temp "):
                temperature = float(user_input.split()[-1])
                print(f"✅ 温度已调整为: {temperature}")
                continue
            if user_input.startswith("set top_p "):
                top_p = float(user_input.split()[-1])
                print(f"✅ Top-P 已调整为: {top_p}")
                continue
            
            response = generate_text_top_p(
                model, user_input, char2idx, idx2char, device, 
                max_new_tokens=60, temperature=temperature, top_p=top_p
            )
            print(f"🤖 AI: {response}")
            
        except KeyboardInterrupt:
            print("\n退出对话。")
            break

if __name__ == "__main__":
    main()