import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def clean_data(data, batch):
  # this function handles the data of multiple games separated by '/'
  done_all = False
  _games   = {'x':[],'y':[]}

  while not done_all:
    idata    = []
    gdata    = {'x':[],'y':[]}
    for i,d in enumerate(data):
      if d != '/':
        idata.append(int(d))
      else:
        data = data[i+1:] # strip the collected data and strip the separator
        break
    # after break idata is now the whole each game massive list of int
    c  = 0
    gd = []
    for d in idata:
      gd.append(d)
      c+=1
      if c == 34:
        gdata['x'].append(gd[:33])
        gdata['y'].append(gd[33])
        c  = 0
        gd = []
      
    # now gdata == {'x':[xframe_1,xframe_2,...xframe_n],'y':[y1, y2, y3,...yn]} for each game
    _games['x'].append(gdata['x'])
    _games['y'].append(gdata['y'])
    vocabs = list(set(y for game in _games['y'] for y in game))
    if len(data) == 0:
      done_all = True

  for i,d in enumerate(_games['x']):
    if len(d) < batch:
      _games['x'].pop(i)
      _games['y'].pop(i)
      i-=1
  return _games, vocabs

def get_traind(self, batch):
  # it has to return [B, T]
  x, y = [], []
  for _ in range(batch):
    grand = torch.randint(0, len(self._games['x']), (1,)).item()
    trand = torch.randint(0, (len(self._games['x'][grand])-self.block_size-1),(1,)).item()
    x.append(self._games['x'][grand][trand:trand+self.block_size])
    y.append(self._games['y'][grand][trand+1:trand+self.block_size+1])

  x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
  return x,y

# encoding and decoding tools:
def encode(s, stoi):
  d = []
  for si in s:
    d.append(stoi[si])
  return d

def decode(l, itos):
  d = []
  for li in l:
    d.append(itos[li])
  return ''.join(d)

def get_deflt(self):
  return [self._games['x'][0][0]]

# used to store residual running inputs!!
class capture:
  x = 0
  def __init__(self):
    self.x = 0

class ResidualBlock(nn.Module):
  def __init__(self, cap=capture, just_capture=False):
    super().__init__()
    self.cap = cap
    self.just_capture = just_capture

  def forward(self, x):
    if self.just_capture:
      self.cap.x = x
      return x

    out = x + self.cap.x
    self.cap.x = out
    
    return out

class LearnedPE(nn.Module):
  def __init__(self, max_seq_len:int, n_embed:int):
    super().__init__()
    self.emb = nn.Embedding(max_seq_len, n_embed)

  def forward(self,x):
    # x = [B, T, C]
    _,T,_   = x.shape
    seq_len = T
    inn     = torch.arange(0, seq_len)
    out     = self.emb(inn)
    return x + out

# this forward pass logic works when nn.MultiheadAttention - batch_first = False
def forward(self, x, targets=None):
  x      = self.norm(x)
  x      = self.ff1(x)
  x      = self.lpe(x)
  x      = self.rb1(x)                                
  _,T,_  = x.shape
  mask   = torch.triu(torch.ones(T, T), 1).bool()
  x,_    = self.mha(x,x,x, attn_mask=mask)           
  logits = self.ff2(x)                               
  # logits = [B, T, C]

  if targets is not None: # cross_entropy expects [B, C, T]
    logits = logits.transpose(1,2)
    loss   = F.cross_entropy(logits, targets)
  else:
    loss   = None

  return logits, loss

def sample_game(self, x):
  self.eval()
  # expects x [B, T, C]
  logits, _   = self(x)
  next_logits = logits[0, -1]
  probs       = F.softmax(next_logits, dim=-1)
  ix          = torch.multinomial(probs, num_samples=1).item()
  return ix

def fit(self, epochs=1000, batch_size=1, lr=1e-3):
  self.optimizer = optim.AdamW(self.parameters(), lr=lr)
  for i in range(epochs):
    self.train()
    self.optimizer.zero_grad(set_to_none=True)

    x, y   = self.get_traind(batch_size)

    _,loss= self(x, y)
    loss.backward()

    # update
    self.optimizer.step()

    if (i+1) % max(1, int(epochs/20)) == 0:
      print(f"epoch:{i+1}   | loss={loss.item():.4f}")
      torch.save(self.state_dict(), "carTransformer1-5m.pt") # save checkpoint during training...