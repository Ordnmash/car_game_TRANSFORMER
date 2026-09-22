import torch
import torch.nn as nn
import torch.optim as optim

from tools import LearnedPE, ResidualBlock, forward, fit, sample_game
from tools import clean_data, get_traind, get_deflt, capture

train_data = "/home/ordn/Documents/ordn_projects/car_game_ai/training_data.txt"
data       = open(train_data, 'r').read().splitlines()

# model hyper-parameters
batch      = 16
insize     = 43
block_size = 500
n_embed    = 500
num_heads  = 5


class CarGame(nn.Module):
  """Transformer Model to simulate Car Game playing"""
  forward    = forward
  fit        = fit
  get_traind = get_traind
  get_deflt  = get_deflt
  sample_game= sample_game

  def __init__(self):  
      self.batch      = batch
      self.block_size = block_size
      self.capture    = capture
      # dataset handling...
      self._games, self.vocabs = clean_data(data, self.block_size)
      self.vocab_size = len(self.vocabs)
  
      super().__init__()
      self.norm= nn.LayerNorm(insize)
      self.ff1 = nn.Linear(insize, n_embed, bias=True)
      self.lpe = LearnedPE(block_size, n_embed)
      self.rb1 = ResidualBlock(self.capture, True)
      self.mha = nn.MultiheadAttention(n_embed, num_heads, 0.2, batch_first=True)
      self.ff2 = nn.Sequential(
         nn.Linear(n_embed, n_embed,        bias=False),
         nn.LayerNorm(n_embed),               nn.Tanh(),
         ResidualBlock(self.capture,just_capture=False),
         nn.Linear(n_embed, self.vocab_size, bias=True),
      )
      self.load_state_dict(torch.load("carTransformer1-5m.pt", weights_only=True))
      parameter_count = sum(p.nelement() for p in self.parameters())
      print(
        f"Number of parameters: "
        f"{parameter_count}")

#model = CarGame()
#model.fit(10000)
