from torch_geometric.nn.models import GCN
from torch import nn
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

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

results = CPU_Unpickler(open("results.pickle","rb")).load()

data = pickle.load(open("data_pyg.pickle","rb"))

embA = {}
embNA = {}

device = "cpu"
layer = 4
seed = 0 

for e in tqdm(data.keys()):

    model = BGP_GNN(layer)

    sd = { k:results[layer][seed]["model_state_dict"][e][k] 
              for k in results[layer][seed]["model_state_dict"][e] if("emb" in k)
         }

    model.load_state_dict(sd)

    embA[e] = model(data[e]["A"], data[e]["node"]).tolist()
    embNA[e] = model(data[e]["NA"], data[e]["node"]).tolist()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
fts = 12

plt.figure(figsize=(5,11))

i = 0
for e in data.keys():
    i+=1
    ax = plt.subplot(4,2,i)

    cmap=sns.diverging_palette(250, 30, l=65, as_cmap=True)

    x = PCA(n_components=2).fit_transform(embNA[e]+embA[e])

    plt.plot(x[:30,0], label="0")
    plt.plot(x[30:,0], label="1")
    

    ax.set_xticks(range(0,31,5))
    ax.set_xticklabels(range(0,61,10))
    ax.set_yticklabels([])

    if(i in [6,7]):
        ax.set_xlabel("Time [Minutes]", fontsize=fts)
    
    if(i%2==1):
        ax.set_ylabel("1st PC", fontsize=fts)
    
    ax.set_title(e, fontsize=fts*1.2)
    plt.tight_layout(h_pad=2) 
    
    if(i==7):
        plt.rcParams['legend.title_fontsize'] = fts
        ax.legend(loc="lower left", fontsize=fts, title="Label", bbox_to_anchor=[1.2,0.2])

plt.savefig('embeddings.png', format='png', bbox_inches='tight')

print("Figure saved as: embeddings.png")
