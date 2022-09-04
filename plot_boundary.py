from torch_geometric.nn.models import GCN
from torch import nn, flatten
from torch_geometric.data import Batch
import pickle
import torch
import io
from tqdm.auto import tqdm

class BGP_GNN(nn.Module):
    def __init__(self, layer):
        super(BGP_GNN, self).__init__()
        self.emb = GCN(in_channels=1, hidden_channels=8, num_layers=layer)

    def forward(self, graphs, node):
        
        for g in graphs:
            g.x = g.nbIp.reshape(-1,1).float()
        
        batch = Batch().from_data_list(graphs)
        
        x = self.emb(batch.x.to(device), batch.edge_index.to(device))
        
        start=0
        indices = []
        for g in graphs:
            indices.append(start + g.ASN.index(node))
            start += g.num_nodes
            
        x = x[indices]
        
        return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.linear1 = nn.Linear(8*30, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, x):
        
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x).sigmoid()
        
        return x

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

results = CPU_Unpickler(open("results.pickle","rb")).load()

data = pickle.load(open("data_pyg.pickle","rb"))

e = "India"
device = "cpu"
layer = 4
seed = 0 

model = BGP_GNN(4)
classifier = MLP()

sd = { k:results[layer][seed]["model_state_dict"][e][k] 
          for k in results[layer][seed]["model_state_dict"][e] if("emb" in k)
     }

sd2 = { k:results[layer][seed]["model_state_dict"][e][k] 
          for k in results[layer][seed]["model_state_dict"][e] if(not "emb" in k)
     }

model.load_state_dict(sd)
classifier.load_state_dict(sd2)

events = list(data.keys())
embsA = []
embsNA = []

for k in tqdm(events):

    emb = flatten(model(data[k]["A"], data[k]["node"])).tolist()
    embsA.append(emb)

    emb = flatten(model(data[k]["NA"], data[k]["node"])).tolist()
    embsNA.append(emb)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

pca= PCA(n_components=2)
x = pca.fit_transform(embsNA+embsA)

h = 1
pad = 100

x_min, x_max = x[:, 0].min()-pad - 1, x[:, 0].max()+pad + 1
y_min, y_max = x[:, 1].min()-pad - 1, x[:, 1].max()+pad + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

d = np.array([xx.ravel(), yy.ravel()]).transpose()

x_new = pca.inverse_transform(d)

cmap=sns.diverging_palette(250, 30, l=65, as_cmap=True)

fts = 14

plt.figure(figsize=(5,4))

with torch.no_grad():
    Z = classifier(torch.Tensor(x_new))

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
    sns.scatterplot(data=pd.DataFrame(x), x=0, y=1,hue=[0]*7+[1]*7, s=100)

    ldg = plt.legend(loc='upper right', fontsize=fts)
    ldg.set_title("Label")
    ldg.get_title().set_fontsize(fts)

    plt.xlabel('1st PC',fontsize=fts)
    plt.ylabel('2nd PC',fontsize=fts)

    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.title(e, fontsize= fts*1.2)

plt.savefig('boundary.png', format='png', bbox_inches='tight')

print("Figure saved as: boundary.png")