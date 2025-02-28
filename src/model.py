import torch
import torch.nn as nn
import math

# ----------------- Input Embeddings -----------------
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Scale embeddings by sqrt(d_model)
        return self.embedding(x) * math.sqrt(x.size(-1))

# ----------------- Positional Encoding -----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)

# ----------------- FeedForward Block -----------------
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# ----------------- Multi-Head Attention Block -----------------
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        return torch.matmul(attn, value)

    def forward(self, x_q, x_k, x_v, mask):
        Q = self.w_q(x_q)
        K = self.w_k(x_k)
        V = self.w_v(x_v)
        batch_size = x_q.size(0)
        Q = Q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        out = self.attention(Q, K, V, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(out)

# ----------------- Residual Connection -----------------
class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, Y):
        return self.norm(x + self.dropout(Y))

# ----------------- Encoder Block & Encoder -----------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention_block
        self.residual1 = ResidualConnection(d_model, dropout)
        self.feed_forward = feed_forward_block
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x, src_mask):
        attn_output = self.self_attention(x, x, x, src_mask)
        x = self.residual1(x, attn_output)
        ff_output = self.feed_forward(x)
        x = self.residual2(x, ff_output)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# ----------------- Decoder Block & Decoder -----------------
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention_block
        self.cross_attention = cross_attention_block
        self.feed_forward = feed_forward_block
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.residual1(x, self_attn_output)
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.residual2(x, cross_attn_output)
        ff_output = self.feed_forward(x)
        x = self.residual3(x, ff_output)
        return x

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

# ----------------- Projection Layer -----------------
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> torch.Tensor:
        return self.proj(x)

# ----------------- Transformer Class -----------------
class Transformer(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src_embeddings = self.src_embed(src)
        src_embeddings = self.src_pos(src_embeddings)
        encoder_output = self.encoder(src_embeddings, src_mask)
        return encoder_output

    def decode(self,
               encoder_output: torch.Tensor,
               src_mask: torch.Tensor,
               tgt: torch.Tensor,
               tgt_mask: torch.Tensor) -> torch.Tensor:
        tgt_embeddings = self.tgt_embed(tgt)
        tgt_embeddings = self.tgt_pos(tgt_embeddings)
        decoder_output = self.decoder(tgt_embeddings, encoder_output, src_mask, tgt_mask)
        return decoder_output

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(x)

# ----------------- Build Transformer Function -----------------
def build_transformer(src_vocab_size: int,
                      tgt_vocab_size: int,
                      src_seq_len: int,
                      tgt_seq_len: int,
                      d_model: int = 512,
                      N: int = 6,
                      h: int = 8,
                      dropout: float = 0.1,
                      d_ff: int = 2048,
                      device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> Transformer:
    # Create embedding layers
    src_embed = InputEmbeddings(d_model=d_model, vocab_size=src_vocab_size)
    tgt_embed = InputEmbeddings(d_model=d_model, vocab_size=tgt_vocab_size)
    
    # Create positional encodings
    src_pos = PositionalEncoding(d_model=d_model, seq_len=src_seq_len, dropout=dropout)
    tgt_pos = PositionalEncoding(d_model=d_model, seq_len=tgt_seq_len, dropout=dropout)
    
    # Create encoder and decoder layers
    encoder_layers = nn.ModuleList([
        EncoderBlock(d_model=d_model,
                     self_attention_block=MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout),
                     feed_forward_block=FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout),
                     dropout=dropout)
        for _ in range(N)
    ])
    
    decoder_layers = nn.ModuleList([
        DecoderBlock(d_model=d_model,
                     self_attention_block=MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout),
                     cross_attention_block=MultiHeadAttentionBlock(d_model=d_model, h=h, dropout=dropout),
                     feed_forward_block=FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout),
                     dropout=dropout)
        for _ in range(N)
    ])
    
    encoder = Encoder(d_model=d_model, layers=encoder_layers).to(device)
    decoder = Decoder(features=d_model, layers=decoder_layers).to(device)
    
    projection_layer = ProjectionLayer(d_model=d_model, vocab_size=tgt_vocab_size).to(device)
    
    transformer = Transformer(encoder=encoder,
                              decoder=decoder,
                              src_embed=src_embed,
                              tgt_embed=tgt_embed,
                              src_pos=src_pos,
                              tgt_pos=tgt_pos,
                              projection_layer=projection_layer).to(device)
    
    # Initialize parameters with Xavier uniform initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer