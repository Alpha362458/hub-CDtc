"""课程结构：Embedding -> RNN/LSTM -> masked max -> LN -> Dropout -> Linear。"""
import torch.nn as nn


class LangIDClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=64, hidden_dim=64,
                 dropout=0.3, encoder="rnn", layer_norm=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        cls = {"rnn": nn.RNN, "lstm": nn.LSTM}[encoder]
        self.encoder = cls(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h, _ = self.encoder(self.embedding(x))
        # 单向编码器的有效位置不受右侧 PAD 影响；池化时明确排除补齐位置。
        h = h.masked_fill(~x.ne(0).unsqueeze(-1), -1e9)
        return self.fc(self.dropout(self.norm(h.max(dim=1).values)))


def from_checkpoint(checkpoint):
    model = LangIDClassifier(len(checkpoint["vocab"]), len(checkpoint["labels"]), **checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
