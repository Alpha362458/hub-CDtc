import math
import argparse
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def load_jsonl_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

class SFTDataset(Dataset):
    def __init__(self, data, char2idx, seq_len):
        self.seq_len = seq_len
        self.pad_id = char2idx["<pad>"]
        self.bos_id = char2idx["<bos>"]
        self.eos_id = char2idx["<eos>"]
        # 如果没有 <sep> 就用普通符号，如果有就加上
        self.sep_id = char2idx.get("。", char2idx.get("<pad>")) 
        
        self.samples = []
        
        for item in data:
            inst = item["instruction"]
            out = item["output"]
            
            # 输入格式: <bos> + instruction + [SEP] + output + <eos>
            inst_ids = [char2idx.get(c, self.pad_id) for c in inst]
            out_ids = [char2idx.get(c, self.pad_id) for c in out]
            
            input_ids = [self.bos_id] + inst_ids + [self.sep_id] + out_ids + [self.eos_id]
            
            # labels 掩码
            labels = list(input_ids)
            labels[0] = -100 
            # 掩蔽 instruction 和 分隔符
            for i in range(1, 1 + len(inst_ids) + 1):
                labels[i] = -100 
                
            if len(input_ids) > seq_len + 1:
                input_ids = input_ids[:seq_len + 1]
                labels = labels[:seq_len + 1]
                
            x = input_ids[:-1]
            y = labels[1:]
            
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def collate_fn(batch, pad_id=0):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    
    padded_xs = []
    padded_ys = []
    for x, y in zip(xs, ys):
        pad_len = max_len - len(x)
        padded_xs.append(torch.cat([x, torch.full((pad_len,), pad_id, dtype=torch.long)]))
        padded_ys.append(torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)]))
        
    return torch.stack(padded_xs), torch.stack(padded_ys)

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout, nhead=8):
        super().__init__()
        self.model_type = model_type
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        
        if self.model_type == "transformer":
            hidden_dim = hidden_dim * 4  
            self.pos_embedding = nn.Embedding(512, embed_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim, 
                dropout=dropout, batch_first=True, norm_first=True 
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(embed_dim, vocab_size)
        else:
            rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
            self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz: int, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        e = self.drop(self.embed(x))
        if self.model_type == "transformer":
            B, T, E = e.size()
            positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
            e = e + self.pos_embedding(positions)
            attn_mask = self.generate_square_subsequent_mask(T, x.device)
            out = self.transformer(e, mask=attn_mask)
        else:
            out, _ = self.rnn(e)
        logits = self.fc(self.drop(out))
        return logits

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        valid_tokens = (y != -100).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss, math.exp(avg_loss)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", default="best_model_transformer.pt")
    parser.add_argument("--data_path",  default="sft_train.jsonl")
    parser.add_argument("--epochs",     type=int,   default=15) # 增加到 15 轮！
    parser.add_argument("--seq_len",    type=int,   default=256) # 增加到 256，防止截断
    parser.add_argument("--batch_size", type=int,   default=16) # 调小 batch
    parser.add_argument("--lr",         type=float, default=1e-4) # 提高到 1e-4
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.pretrained_model, map_location="cpu")
    pt_args = ckpt["args"]
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    
    model = LM(
        vocab_size=len(char2idx), embed_dim=pt_args["embed_dim"],
        hidden_dim=pt_args["hidden_dim"], num_layers=pt_args["num_layers"],
        model_type=pt_args["model"], dropout=0.1
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    all_data = load_jsonl_data(args.data_path)
    random.shuffle(all_data)
    split = int(len(all_data) * (1 - args.val_ratio))
    train_ds = SFTDataset(all_data[:split], char2idx, args.seq_len)
    val_ds   = SFTDataset(all_data[split:], char2idx, args.seq_len)

    pad_id = char2idx["<pad>"]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id))

    criterion = nn.CrossEntropyLoss(ignore_index=-100) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}")
    print("-" * 36)
    for epoch in range(1, args.epochs + 1):
        tr_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {va_loss:>10.4f}")
        
        torch.save({"model_state": model.state_dict(), "char2idx": char2idx, "idx2char": idx2char, "args": vars(args)}, "best_model_sft.pt")

if __name__ == "__main__":
    main()