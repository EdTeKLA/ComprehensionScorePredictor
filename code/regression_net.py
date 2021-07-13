import torch.nn as nn
import torch
torch.manual_seed(0)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = nn.Linear(9,27)  # 9-(10-10)-1
    # self.drop = nn.Dropout(p=0.1)
    self.hid2 = nn.Linear(27, 10)
    self.oupt = nn.Linear(10, 1)

    nn.init.xavier_uniform_(self.hid1.weight)
    nn.init.zeros_(self.hid1.bias)
    nn.init.xavier_uniform_(self.hid2.weight)
    nn.init.zeros_(self.hid2.bias)
    nn.init.xavier_uniform_(self.oupt.weight)
    nn.init.zeros_(self.oupt.bias)

    self.relu = nn.ReLU()

  def forward(self, x):
    z = self.relu(self.hid1(x))
    # z = self.drop(z)
    z = self.relu(self.hid2(z))
    z = self.oupt(z)  # no activation
    return z