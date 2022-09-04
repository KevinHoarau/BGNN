from torch_geometric.nn.models import GCN
from torch import nn, flatten
from torch_geometric.data import Batch

class BGP_GNN(nn.Module):
    def __init__(self, layer, device="cpu", w=30):
        super(BGP_GNN, self).__init__()
        
        self.device = device
        
        self.emb = GCN(in_channels=1, hidden_channels=8, num_layers=layer)
        
        self.linear1 = nn.Linear(8*w, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
        
        self.w = w

    def forward(self, graphs, node):
        
        graphs = graphs[:self.w]
        
        for g in graphs:
            g.x = g.nbIp.reshape(-1,1).float()
        
        batch = Batch().from_data_list(graphs)
        
        x = self.emb(batch.x.to(self.device), batch.edge_index.to(self.device))
        
        start=0
        indices = []
        for g in graphs:
            indices.append(start + g.ASN.index(node))
            start += g.num_nodes
            
        x = x[indices]
        x = flatten(x)
        
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x).sigmoid()
        
        return x