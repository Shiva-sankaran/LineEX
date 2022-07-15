
import torch
from torch import nn


class legend_network(nn.Module):
    def __init__(self,embed_dim):
        super(legend_network, self).__init__()
        self.fc1 = nn.Linear(embed_dim,embed_dim//2)
        self.fc3 = nn.Linear(embed_dim//2,1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(0.2)
        

        
    def forward(self, x1,x2):
        out = torch.cat((x1,x2),dim = 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out