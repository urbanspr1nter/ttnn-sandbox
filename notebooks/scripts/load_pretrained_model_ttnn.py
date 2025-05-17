import torch
from scripts.gpt2_model_ttnn import GPTModel_ttnn 
from scripts.gpt2_model import GPTModel

def load_pretrained_gpt2_model_ttnn(path_to_model, cfg):
  model = GPTModel(cfg)
  model.load_state_dict(
    torch.load(path_to_model, weights_only=True)
  )

  model_ttnn = GPTModel_ttnn(cfg, device)

  model_ttnn.pos_emb.weight = torch.nn.Parameter(model.pos_emb.weight) 
  model_ttnn.tok_emb.weight = torch.nn.Parameter(model.tok_emb.weight)

  for i, block in enumerate(model.trf_blocks):
    t = model_ttnn.trf_blocks_ttnn[i]

    t.att.W_key.weight = torch.nn.Parameter(block.att.W_key.weight)
    t.att.W_key.bias = torch.nn.Parameter(block.att.W_key.bias)

    t.att.W_query.weight = torch.nn.Parameter(block.att.W_query.weight)
    t.att.W_query.bias = torch.nn.Parameter(block.att.W_query.bias)
    
    t.att.W_value.weight = torch.nn.Parameter(block.att.W_value.weight)
    t.att.W_value.bias = torch.nn.Parameter(block.att.W_value.bias)
    
    t.att.out_proj.weight = torch.nn.Parameter(block.att.out_proj.weight)
    t.att.out_proj.bias = torch.nn.Parameter(block.att.out_proj.bias)

    t.ff.lin_1.weight = torch.nn.Parameter(block.ff.layer[0].weight)
    t.ff.lin_1.bias = torch.nn.Parameter(block.ff.layer[0].bias)
    t.ff.lin_2.weight = torch.nn.Parameter(block.ff.layer[2].weight)
    t.ff.lin_2.bias = torch.nn.Parameter(block.ff.layer[2].bias)

    t.norm1.scale = torch.nn.Parameter(block.norm1.scale)
    t.norm1.shift = torch.nn.Parameter(block.norm1.shift)

    t.norm2.scale = torch.nn.Parameter(block.norm2.scale)
    t.norm2.shift = torch.nn.Parameter(block.norm2.shift)

  model_ttnn.final_norm.shift = torch.nn.Parameter(model.final_norm.shift)
  model_ttnn.final_norm.scale = torch.nn.Parameter(model.final_norm.scale)
  model_ttnn.out_head.weight = torch.nn.Parameter(model.out_head.weight)

  model_ttnn.update_weights()

  return model_ttnn