from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn

from labml_helpers.module import Module, TypedModuleList
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.xl.relative_mha import RelativeMultiHeadAttention
from labml_nn.utils import clone_module_list

from basis_functions import GaussianBasisFunctions

class InfiniteTransformerLayer(Module):
  def __init__(self, *, 
               d_model: int,
               n_basis: int,
               ltm_n_heads: int,
               ltm_head_size: int,
               self_attn: RelativeMultiHeadAttention,
               feed_forward: FeedForward,
               dropout_prob: float):
    super().__init__()
    self.size = d_model
    self.self_attn = self_attn
    self.feed_forward = feed_forward
    self.dropout = nn.Dropout(dropout_prob)
    self.norm_self_attn = nn.LayerNorm([d_model])
    self.norm_ff = nn.LayerNorm([d_model])
    self.x_past = None
    self.B_past = None
    self.n_basis = n_basis

    d = ltm_n_heads * ltm_head_size
    self.proj_query = nn.Linear(d, d, bias=False)
    self.proj_key = nn.Linear(d, d, bias=False)
    self.proj_value = nn.Linear(d, d, bias=False)
    
    self.psi = [None]
    locations = torch.linspace(0, 1 , 1 / self.d_model)
    self.__set_basis_functions(self.psi[0])
    self.G_inf = self.__calculate_G(self.psi, locations)

  def __set_basis_functions(self, psi: GaussianBasisFunctions):
    sigmas = [.01, .05]
    mu, sigma = torch.meshgrid(torch.linspace(0, 1, self.n_basis // 2), torch.Tensor(sigmas))
    self.basis_mu = mu
    self.basis_sigma = sigma
    psi.append(GaussianBasisFunctions(mu, sigma))

  def __calculate_G(self, psi: GaussianBasisFunctions, locations: torch.Tensor):
    F = torch.zeros(self.n_basis, locations.size(0))

    F[:, :] = psi.evaluate(locations.unsqueeze(1)).t()

    I = torch.eye(self.n_basis)
    G = F.t().matmul((F.matmul(F.t()) + 1 * I).inverse())

    return G

  def sample_discrete(self, x: torch.Tensor):
    B = torch.matmul(x, self.G_inf)
    B = B.permute(0, 2, 1)
    return B

  def update_infinite_mem(self, x: torch.Tensor):
    if self.B_past is not None:
      xm_tau = self.B_past.transpose(-1, -2).matmul(self.samples.transpose(0, 1))
      x = torch.cat([xm_tau, x], dim=2)

    B = self.sample_discrete(x)

    self.B_past = B.detach()
    self.x_past = x
    return B

  def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor], mask: torch.Tensor):
    z = self.norm_self_attn(x)

    if mem is not None:
      mem = self.norm_self_attn(mem)
      m_z = torch.cat((mem, z), dim=0)
    else:
      m_z = z

    self_attn = self.self_attn(query=z, key=m_z, value=m_z, mask=mask)
    B = self.update_inf(z)
    K_h = self.proj_key(B)
    V_h = self.proj_value(B)
    
    x = x + self.dropout(self_attn)

    z = self.norm_ff(x)
    ff = self.feed_forward(z)
    x = x + self.dropout(ff)

    return x

class InfiniteTransformer(Module):
  def __init__(self, layer: InfiniteTransformerLayer, n_layers: int):
    super().__init__()
    self.layers = clone_module_list(layer, n_layers)
    self.norm = nn.LayerNorm([layer.size])

  def forward(self, x: torch.Tensor,  mem: List[torch.Tensor], mask: torch.Tensor):
    new_mem = []
    for i, layer in enumerate(self.layers):
      new_mem.append(x.detach())
      m = mem[i] if mem else None
      x = layer(x=x, mem=m, mask=mask)
    return self.norm(x), new_mem