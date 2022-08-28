import torch
import torch.nn as nn

class encoder(nn.Module):
  def __init__(self,in_channels =216):
    super(encoder,self).__init__()
    
    self.layer_1 = nn.Linear(in_channels,in_channels//2)
    self.layer_2 = nn.Linear(in_channels//2,in_channels//4)
    self.layer_3 = nn.Linear(in_channels//4,in_channels//8)
    self.layer_middle = nn.Linear(in_channels//8,in_channels//8)

    self.relu = nn.ReLU()
  def forward(self,x):
    x = self.layer_1(x)
    x  = self.relu(x)
    x = self.layer_2(x)
    x  = self.relu(x)
    x = self.layer_3(x)
    x  = self.relu(x)
    x = self.layer_middle(x)
    x = self.relu(x)
    return x

class decoder(nn.Module):
  def __init__(self,in_channels=216):
    super(decoder,self).__init__()
    self.layer_middle = nn.Linear(in_channels//8,in_channels//8)
    self.layer_4 = nn.Linear(in_channels//8,in_channels//4)
    self.layer_5 = nn.Linear(in_channels//4,in_channels//2)
    self.layer_6 = nn.Linear(in_channels//2,in_channels)
    self.relu = nn.ReLU()

  def forward(self,x):
    x = self.layer_middle(x)
    x  = self.relu(x)
    x = self.layer_4(x)
    x  = self.relu(x)
    x = self.layer_5(x)
    x  = self.relu(x)
    x = self.layer_6(x)
    x  = self.relu(x)
    return x