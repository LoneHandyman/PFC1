from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn

from labml_helpers.module import Module, TypedModuleList
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import PrepareForMultiHeadAttention
from labml_nn.transformers.xl.relative_mha import RelativeMultiHeadAttention
from labml_nn.utils import clone_module_list


class Conv1dCompression(Module):
  def __init__(self, compression_rate: int, d_model: int):
    super().__init__()
    self.conv = nn.Conv1d(d_model, d_model, kernel_size=compression_rate, stride=compression_rate)

  def forward(self, mem: torch.Tensor):
    mem = mem.permute(1, 2, 0)
    c_mem = self.conv(mem)
    return c_mem.permute(2, 0, 1)


class CompressiveTransformerLayer(Module):
  def __init__(self, *,
               d_model: int,
               self_attn: RelativeMultiHeadAttention,
               feed_forward: FeedForward,
               dropout_prob: float,
               compress: Conv1dCompression):
    super().__init__()
    self.compress = compress
    self.size = d_model
    self.self_attn = self_attn
    self.feed_forward = feed_forward
    self.dropout = nn.Dropout(dropout_prob)
    self.norm_self_attn = nn.LayerNorm([d_model])
    self.norm_ff = nn.LayerNorm([d_model])

  def concat_memory(self, z: torch.Tensor, mem: Optional[torch.Tensor], c_mem: Optional[torch.Tensor]):
      if mem is None:
        return z

      if c_mem is not None:
        mem = torch.cat((c_mem, mem), dim=0)

      mem = self.norm_self_attn(mem)
      return torch.cat((mem, z), dim=0)

  def forward(self, *,
            x: torch.Tensor,
            mem: Optional[torch.Tensor],
            c_mem: Optional[torch.Tensor],
            mask: torch.Tensor):

    z = self.norm_self_attn(x)
    m_z = self.concat_memory(z, mem, c_mem)
    self_attn = self.self_attn(query=z, key=m_z, value=m_z, mask=mask)
    x = x + self.dropout(self_attn)

    z = self.norm_ff(x)
    ff = self.feed_forward(z)
    x = x + self.dropout(ff)

    return x


class CompressiveTransformer(Module):
  def __init__(self, layer: CompressiveTransformerLayer, n_layers: int):
    super().__init__()
    self.layers = clone_module_list(layer, n_layers)
    self.norm = nn.LayerNorm([layer.size])

  def forward(self, x: torch.Tensor, mem: List[torch.Tensor], c_mem: List[torch.Tensor], mask: torch.Tensor):
    new_mem = []
    for i, layer in enumerate(self.layers):
      new_mem.append(x.detach())
      m = mem[i] if mem else None
      cm = c_mem[i] if c_mem else None
      x = layer(x=x, mem=m, c_mem=cm, mask=mask)
    return self.norm(x), new_mem


class AttentionReconstructionLoss:
  def __init__(self, layers: TypedModuleList[CompressiveTransformerLayer]):
    self.layers = layers
    self.loss_func = nn.MSELoss()

  def prepare_for_attn(self, pmha: PrepareForMultiHeadAttention, x: torch.Tensor):
    head_shape = x.shape[:-1]

    weight = pmha.linear.weight.detach()
    bias = pmha.linear.bias.detach() if pmha.linear.bias is not None else None
    x = F.linear(x, weight, bias)

    x = x.view(*head_shape, pmha.heads, pmha.d_k)

    return x

  def attn(self, layer: RelativeMultiHeadAttention, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    query = self.prepare_for_attn(layer.query, query)
    key = self.prepare_for_attn(layer.key, key)
    value = self.prepare_for_attn(layer.value, value)

    scores = torch.einsum('ibhd,jbhd->ijbh', query, key)

    scores *= layer.scale

    attn = layer.softmax(scores)

    return torch.einsum("ijbh,jbhd->ibhd", attn, value)

  def norm(self, ln: nn.LayerNorm, x: torch.Tensor):
    weight = ln.weight.detach() if ln.weight is not None else None
    bias = ln.bias.detach() if ln.bias is not None else None

    return F.layer_norm(x, ln.normalized_shape, weight, bias, ln.eps)

  def calc_loss(self, layer: CompressiveTransformerLayer, h: torch.Tensor, mem: torch.Tensor):
    h = h.detach()
    mem = mem.detach()

    c_mem = layer.compress(mem)

    h = self.norm(layer.norm_self_attn, h)
    mem = self.norm(layer.norm_self_attn, mem)
    c_mem = self.norm(layer.norm_self_attn, c_mem)

    attn_mem = self.attn(layer.self_attn, h, mem, mem)
    attn_cmem = self.attn(layer.self_attn, h, c_mem, c_mem)

    return self.loss_func(attn_cmem, attn_mem)

  def __call__(self, h: List[torch.Tensor], mem: List[torch.Tensor]):
    losses = [self.calc_loss(layer, h[n], mem[n]) for n, layer in enumerate(self.layers)]
    return sum(losses)
