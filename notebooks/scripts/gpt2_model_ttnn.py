import ttnn
import torch
from torch import nn
from .multihead_attention_ttnn import MultiHeadAttention_ttnn

class LayerNorm_ttnn(nn.Module):
  """
  Applies normalization of the weights
  """
  def __init__(self, emb_dim, device):
    super().__init__()

    self.device = device

    self.eps = 1e-5
    
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))

    self.scale_ttnn = ttnn.from_torch(
      self.scale,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device
    )
    self.shift_ttnn = ttnn.from_torch(
      self.shift,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device
    )

  def update_weights(self):
    self.scale_ttnn = ttnn.from_torch(
      self.scale,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device
    )
    self.shift_ttnn = ttnn.from_torch(
      self.shift,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device
    )

  def forward(self, x_ttnn):
    mean_ttnn = ttnn.mean(
      x_ttnn,
      dim=-1
    )
    var_ttnn = ttnn.var(
      x_ttnn,
      dim=-1
    )

    norm_x_ttnn = ttnn.div(
      ttnn.subtract(
        x_ttnn,
        mean_ttnn
      ),
      ttnn.sqrt(
        ttnn.add(
          var_ttnn,
          self.eps
        )
      )
    )

    return norm_x_ttnn

class GELU_ttnn(nn.Module):
  """
  GELU Module
  """
  def __init__(self, device):
    super().__init__()

    self.device = device

  def forward(self, x_ttnn):
    x_ttnn_cubed = ttnn.pow(
      x_ttnn,
      3
    )

    pi_tensor = ttnn.from_torch(
      torch.tensor(2.0 / torch.pi),
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device
    )

    sqrt_pi_tensor = ttnn.sqrt(
      pi_tensor
    )

    sqrt_factor_ttnn = ttnn.multiply(
      sqrt_pi_tensor,
      ttnn.add(
        x_ttnn,
        ttnn.multiply(
          x_ttnn_cubed,
          0.044715
        )
      )      
    )

    tanh_factor_ttnn = ttnn.tanh(
      sqrt_factor_ttnn
    )

    result_ttnn = ttnn.multiply(
      x_ttnn,
      ttnn.multiply(
        ttnn.add(tanh_factor_ttnn, 1),
        0.5
      )
    )

    return result_ttnn 

class FeedForward_ttnn(nn.Module):
  def __init__(self, cfg, device):
    super().__init__()

    self.device = device

    self.lin_1 = nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"])
    self.gelu = GELU_ttnn(self.device)
    self.lin_2 = nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])

    self.lin_1_ttnn = ttnn.from_torch(
      self.lin_1.weight,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )
    self.lin_2_ttnn = ttnn.from_torch(
      self.lin_2.weight,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )
    self.lin_1_bias_ttnn = ttnn.from_torch(
      self.lin_1.bias,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )
    self.lin_2_bias_ttnn = ttnn.from_torch(
      self.lin_2.bias,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )

    self.gelu_ttnn = GELU_ttnn(self.device)

  def update_weights(self):
    """
    Recreate the the linear layer tensors after
    they have been updated.
    """
    self.lin_1_ttnn = ttnn.from_torch(
      self.lin_1.weight,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )
    self.lin_2_ttnn = ttnn.from_torch(
      self.lin_2.weight,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )
    self.lin_1_bias_ttnn = ttnn.from_torch(
      self.lin_1.bias,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )
    self.lin_2_bias_ttnn = ttnn.from_torch(
      self.lin_2.bias,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )

  def forward(self, x_ttnn):
    x_ttnn = ttnn.linear(
      x_ttnn,
      self.lin_1_ttnn,
      transpose_b=True,
      bias=self.lin_1_bias_ttnn,
    )

    x_ttnn = self.gelu_ttnn(x_ttnn)

    x_ttnn = ttnn.linear(
      x_ttnn,
      self.lin_2_ttnn,
      transpose_b=True,
      bias=self.lin_2_bias_ttnn,
    )

    return x_ttnn 

class TransformerBlock_ttnn(nn.Module):
  def __init__(self, cfg, device):
    super().__init__()

    self.cfg = cfg
    self.device = device

    self.att = MultiHeadAttention_ttnn(
      d_in=cfg["emb_dim"],
      d_out=cfg["emb_dim"],
      context_length=cfg["context_length"],
      num_heads=cfg["n_heads"],
      dropout=cfg["drop_rate"],
      qkv_bias=cfg["qkv_bias"],
      device=self.device
    )

    self.ff = FeedForward_ttnn(cfg, self.device)

    self.norm1 = LayerNorm_ttnn(cfg["emb_dim"], self.device)
    self.norm2 = LayerNorm_ttnn(cfg["emb_dim"], self.device)

  def update_weights(self):
    self.att.update_weights()
    self.ff.update_weights()
    self.norm1.update_weights()
    self.norm2.update_weights()

  def do_dropout(self, x_ttnn):
    x_ttnn = ttnn.experimental.dropout(
      x_ttnn,
      seed=123,
      probability=self.cfg["drop_rate"],
      scale=1.0 / (1.0 - self.cfg["drop_rate"])
    )
    return x_ttnn

  def forward(self, x):
    
    shortcut = x
    x = self.norm1(x)
    x = self.att(x)

    if self.cfg["drop_rate"] > 0.0:
      x = self.do_dropout(x) 
    
    x = ttnn.add(x, shortcut)

    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)

    if self.cfg["drop_rate"] > 0.0:
      x = self.do_dropout(x)
    
    x = ttnn.add(x, shortcut)

    return x

class GPTModel_ttnn(nn.Module):
  def __init__(self, cfg, device):
    super().__init__()

    self.cfg = cfg
    self.device = device
    self.num_layers = self.cfg["n_layers"]

    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])    

    self.tok_emb_ttnn = ttnn.from_torch(
      self.tok_emb.weight,
      dtype=ttnn.bfloat16,
      device=self.device,
    )
    self.pos_emb_ttnn = ttnn.from_torch(
      self.pos_emb.weight,
      dtype=ttnn.bfloat16,
      device=self.device,
    )

    self.trf_blocks_ttnn = [
      TransformerBlock_ttnn(self.cfg, self.device) for _ in range(self.num_layers)
    ]

    self.final_norm = LayerNorm_ttnn(cfg["emb_dim"], self.device)

    self.out_head = nn.Linear(
      cfg["emb_dim"], cfg["vocab_size"], bias=False
    )

    self.out_head_ttnn = ttnn.from_torch(
      self.out_head.weight,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )

  def update_weights(self):
    for tr_block in self.trf_blocks_ttnn:
      tr_block.update_weights()

    self.tok_emb_ttnn = ttnn.from_torch(
      self.tok_emb.weight,
      dtype=ttnn.bfloat16,
      device=self.device,
    )
    self.pos_emb_ttnn = ttnn.from_torch(
      self.pos_emb.weight,
      dtype=ttnn.bfloat16,
      device=self.device,
    )

    self.final_norm.update_weights()

    self.out_head_ttnn = ttnn.from_torch(
      self.out_head.weight,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )



  def do_dropout(self, x_ttnn):
    x_ttnn = ttnn.experimental.dropout(
      x_ttnn,
      seed=123,
      probability=self.cfg["drop_rate"],
      scale=1.0 / (1.0 - self.cfg["drop_rate"])
    )

    return x_ttnn

  def forward(self, in_idx_ttnn):
    batch_size, seq_len = in_idx_ttnn.shape

    tok_embed_vals_ttnn = ttnn.embedding(
      in_idx_ttnn,
      self.tok_emb_ttnn
    )
    pos_embed_vals_ttnn = ttnn.embedding(
      ttnn.arange(
        start=0,
        end=seq_len,
        step=1,
        dtype=ttnn.uint32,
        device=self.device
      ),
      self.pos_emb_ttnn
    )

    x_ttnn = ttnn.add(
      ttnn.to_layout(tok_embed_vals_ttnn, layout=ttnn.TILE_LAYOUT),
      ttnn.to_layout(pos_embed_vals_ttnn, layout=ttnn.TILE_LAYOUT)
    )

    if self.cfg["drop_rate"] > 0.0:
      x_ttnn = self.do_dropout(x_ttnn)

    for trf_block_ttnn in self.trf_blocks_ttnn:
      x_ttnn = trf_block_ttnn(x_ttnn)

    x_ttnn = self.final_norm(x_ttnn)

    logits_ttnn = ttnn.linear(
      x_ttnn,
      self.out_head_ttnn,
      transpose_b=True,
    )

    return logits_ttnn