import ttnn
import torch
from torch import nn

core_grid_x = 8
core_grid_y = 8
MINUS_INFINITY = -1e9

class MultiHeadAttention_ttnn(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, device, qkv_bias=False):
    super().__init__()

    self.device = device
    self.dropout_prob = dropout

    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads # Reduce the projection dimension to match the output dim

    # Trainable weights
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj = nn.Linear(d_out, d_out)

    # STatic Tensors
    self.dropout = nn.Dropout(dropout)
    self.mask = torch.triu(
      torch.ones(context_length, context_length),
      diagonal=1
    )

    self.W_query_ttnn = ttnn.from_torch(
      self.W_query.weight,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )
    self.W_key_ttnn = ttnn.from_torch(
      self.W_key.weight,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )
    self.W_value_ttnn = ttnn.from_torch(
      self.W_value.weight,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )
    self.out_proj_ttnn = ttnn.from_torch(
      self.out_proj.weight,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )
    self.mask_ttnn = ttnn.from_torch(
      self.mask,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )

  def update_weights(self):
    self.W_query_ttnn = ttnn.from_torch(
      self.W_query.weight,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )
    self.W_key_ttnn = ttnn.from_torch(
      self.W_key.weight,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )
    self.W_value_ttnn = ttnn.from_torch(
      self.W_value.weight,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )
    self.out_proj_ttnn = ttnn.from_torch(
      self.out_proj.weight,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device,
    )

  def forward(self, x_ttnn):
    b, num_tokens, d_in = x_ttnn.shape

    keys_ttnn = ttnn.linear(
      x_ttnn,
      self.W_key_ttnn,
      transpose_b=True,
      core_grid=ttnn.CoreGrid(y=core_grid_y, x=core_grid_x)
    )
    queries_ttnn = ttnn.linear(
      x_ttnn,
      self.W_query_ttnn,
      transpose_b=True,
      core_grid=ttnn.CoreGrid(y=core_grid_y, x=core_grid_x)
    )
    values_ttnn = ttnn.linear(
      x_ttnn,
      self.W_value_ttnn,
      transpose_b=True,
      core_grid=ttnn.CoreGrid(y=core_grid_y, x=core_grid_x)
    )
    
    keys_ttnn = ttnn.reshape(keys_ttnn, (b, num_tokens, self.num_heads, self.head_dim))
    values_ttnn = ttnn.reshape(values_ttnn, (b, num_tokens, self.num_heads, self.head_dim))
    queries_ttnn = ttnn.reshape(queries_ttnn, (b, num_tokens, self.num_heads, self.head_dim))

    # NOTE! This is intentional. We want the transposed version of keys_ttnn. That's why the
    # shape has a different permutation than the values_ttnn and queries_ttnn!
    keys_transposed_ttnn = ttnn.permute(keys_ttnn, (0, 2, 3, 1))
    values_ttnn = ttnn.permute(values_ttnn, (0, 2, 1, 3))
    queries_ttnn = ttnn.permute(queries_ttnn, (0, 2, 1, 3))

    attn_scores_ttnn = ttnn.matmul(
      queries_ttnn, 
      keys_transposed_ttnn,
      core_grid=ttnn.CoreGrid(y=core_grid_y, x=core_grid_x)
    )

    attn_scores_ttnn = attn_scores_ttnn * (1 / (self.head_dim ** 0.5))

    attn_mask_ttnn = self.mask_ttnn[:num_tokens, :num_tokens] * MINUS_INFINITY
    attn_mask_ttnn = ttnn.reshape(attn_mask_ttnn, (1, 1, num_tokens, num_tokens))
    attn_scores_ttnn += attn_mask_ttnn
    
    attn_weights_ttnn = ttnn.softmax(attn_scores_ttnn, dim=-1)

    if self.dropout_prob > 0.0:
      attn_weights_ttnn = ttnn.experimental.dropout(
        attn_weights_ttnn,
        seed=123,
        probability=self.dropout_prob,
        scale=1.0 / (1.0 - self.dropout_prob)
      )

    
    context_vec_ttnn = ttnn.matmul(
      attn_weights_ttnn,
      values_ttnn,
      core_grid=ttnn.CoreGrid(y=core_grid_y, x=core_grid_x)
    ) 

    context_vec_ttnn = ttnn.permute(context_vec_ttnn, (0, 2, 1, 3))
    context_vec_ttnn = ttnn.reshape(context_vec_ttnn, (b, num_tokens, self.d_out))

    out_proj_bias_ttnn = ttnn.from_torch(
      self.out_proj.bias,
      dtype=ttnn.bfloat16,
      layout=ttnn.TILE_LAYOUT,
      device=self.device
    )

    context_vec_ttnn = ttnn.linear(
      context_vec_ttnn,
      self.out_proj_ttnn,
      transpose_b=True,
      core_grid=ttnn.CoreGrid(y=core_grid_y, x=core_grid_x)
    )
    context_vec_ttnn = ttnn.add(context_vec_ttnn, out_proj_bias_ttnn)

    return context_vec_ttnn