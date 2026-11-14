import torch
import torch.nn as nn
import torch.nn.functional as F

class BertConfig:
    def __init__(self, **kwargs):
        self.vocab_size = 8
        self.hidden_size = 768
        self.layer_norm_eps = 1e-12
        self.hidden_dropout_prob = 0.1
        self.position_size = 512
        self.segment_size = 2
        self.pad_token = 0
        self.num_attention_heads = 12
        self.attention_probs_dropout_prob = 0.1
        self.intermediate_size = 3072
        self.num_hidden_layers = 12

        for key, value in kwargs.items():
            setattr(self, key, value)
# Embedding
class BertEmbedding(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,config.hidden_size,padding_idx = config.pad_token)
        self.position_embeddings = nn.Embedding(config.position_size,config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.segment_size,config.hidden_size)

        self.layerNorm = nn.LayerNorm(config.hidden_size,eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self,inputs,segments = None):
        """_summary_

        Args:
            inputs (_type_): 输入序列 (batch,seq_len)
            segments (_type_): 输入序列的类型索引
        Returns:
            : 输出序列的嵌入表示量
        """
        seq_len = inputs.size(1) # 取seq_len
        positions = torch.arange(seq_len,dtype=torch.long,device=inputs.device) # seq_len
        positions = positions.unsqueeze(0).expand_as(inputs) # batch,seq_len
        
        if segments is None:
            segments = torch.zeros_like(inputs)
        
        words_emb = self.word_embeddings(inputs)
        position_emb = self.position_embeddings(positions)
        segment_emb = self.token_type_embeddings(segments)

        embedding = words_emb+position_emb+segment_emb
        embedding = self.layerNorm(embedding)
        embedding = self.dropout(embedding)

        return embedding
        
class BertSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.head_dim*self.num_heads
        assert self.all_head_size == config.hidden_size
        
        # 线性投影
        self.query = nn.Linear(config.hidden_size,self.all_head_size)
        self.key = nn.Linear(config.hidden_size,self.all_head_size)
        self.value = nn.Linear(config.hidden_size,self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self,x):
        # batch,seq_len,head_dim -> batch,num_heads,seq_len,head_dim
        new_shape = x.size()[:-1]+(self.num_heads,self.head_dim) # 新张量形状
        x = x.view(*new_shape) # 重组形状
        return x.permute(0,2,1,3) # 维度换顺序



    def forward(self,hidden_states,attention_mask):
        """_summary_

        Args:
            hidden_states (_type_): _description_
            attention_mask (_type_): _description_
        """

        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # QK^T -> [batch,num_heads,seq_len,head_dim] @ [batch,num_heads,head_dim,seq_len]
        # -> batch,num_heads,seq_len,seq_len
        attn_scores = torch.matmul(q,k.transpose(-1,-2))/torch.sqrt(torch.tensor(self.head_dim,dtype=q.dtype,device=q.device))
        # ? 
        attn_scores = attn_scores + attention_mask

        # ?
        attn_prob = F.softmax(attn_scores,dim=-1)
        attn_prob = self.dropout(attn_prob)

        out = torch.matmul(attn_prob,v)
        out = out.permute(0,2,1,3).contiguous()
        new_shape = out.size()[:-2] + (self.all_head_size,)
        out = out.view(*new_shape)
        return out


# Add & LayerNorm
class BertSelfOutput(nn.Module):
    def __init__(self,config:BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size,config.hidden_size)
        self.layerNorm = nn.LayerNorm(config.hidden_size,eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self,hidden_states,input_tensor):
        out = self.dense(hidden_states)
        out = self.dropout(out)
        out = self.layerNorm(out+input_tensor)
        return out

# 自注意力
class BertAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attention = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
    
    def forward(self,inputs,attention_mask):
        attention_output = self.attention(inputs,attention_mask)
        out = self.output(attention_output,inputs)
        return out

# FFN
class BertIntermediate(nn.Module):
    def __init__(self,config:BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size,config.intermediate_size)
    
    def forward(self,hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states) # BERT用GELU
        return hidden_states  

# FFN后Add&LayerNorm
class BertOutput(nn.Module):
    def __init__(self,config:BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size,config.hidden_size)
        self.layerNorm = nn.LayerNorm(config.hidden_size,eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self,hidden_states,input_tensor):
        out = self.dense(hidden_states)
        out = self.dropout(out)
        out = self.layerNorm(out+input_tensor)
        return out

class BertLayer(nn.Module):
    def __init__(self,config:BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attn_out = self.attention(hidden_states, attention_mask)
        inter_out = self.intermediate(attn_out)
        layer_out = self.output(inter_out, attn_out)
        return layer_out

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbedding(config)
        self.encoder = BertEncoder(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)

        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_mask = (1.0 - extended_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_out = self.encoder(embedding_output, extended_mask)
        return encoder_out


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layerNorm(hidden_states)
        return hidden_states



class BertLMPredictionHead(nn.Module):
    def __init__(self, config, embedding_weight):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
        self.decoder.weight = embedding_weight

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.cls = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        hidden_states = self.bert(input_ids, attention_mask, token_type_ids)
        prediction_scores = self.cls(hidden_states)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            return loss, prediction_scores
        return prediction_scores


        



        
