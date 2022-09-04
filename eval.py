from model import BGP_GNN
from os import mkdir, path
import pickle

import torch, torch_geometric
from torch import nn, Tensor, optim
from torch_geometric.data import Batch
import random
from tqdm.auto import tqdm

def train(trainEv, seed, epoch, lr, layer):
    
    torch.manual_seed(seed)
    random.seed(seed)

    model = BGP_GNN(layer, device=device).to(device)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    def init_weights(m):
        if isinstance(m, torch_geometric.nn.conv.GCNConv):
            torch.nn.init.kaiming_normal_(m.lin.weight)

    model.apply(init_weights)
    
    for e in range(epoch):
        
        model.train()

        loss_sum = 0

        random.shuffle(trainEv)

        for name in trainEv:
            
            node = data[name]["node"]

            graphs = data[name]["A"]
            pred = model(graphs, node)
            loss = loss_fn(pred.cpu(), Tensor([1]))
            
            loss_sum += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            graphs = data[name]["NA"]
            pred = model(graphs, node)
            loss = loss_fn(pred.cpu(), Tensor([0]))
            
            loss_sum += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    return(model)


def test(testEv, model):

    model.eval()
    with torch.no_grad():

        name = testEv
        node = data[name]["node"]
        y_true = [1,0]
        y_score = []
        y_pred = []
        

        graphs = data[name]["A"]
        score = model(graphs, node).item()
        pred = int(score>0.5)
        y_score.append(score)
        y_pred.append(pred)

        graphs = data[name]["NA"]
        score = model(graphs, node).item()
        pred = int(score>0.5)
        y_score.append(score)
        y_pred.append(pred)

    return(y_true, y_score, y_pred)

print("BGNN model evaluation")

device = "cuda"
nb_seeds = 30
nb_epoch = 50

data = pickle.load(open("data_pyg.pickle","rb"))

events = list(data.keys())
results = {}

layers = [1,2,4,8,16]

progress = tqdm(total=len(layers)*nb_seeds*len(events))

for layer in layers:

    results[layer] = {}
    
    for seed in range(nb_seeds):

        results[layer][seed] = {
            "y_true": [],
            "y_score": [],
            "y_pred": [],
            "model_state_dict": {}
        }
        
        for testEv in events:

            trainEv = events.copy()
            trainEv.remove(testEv)

            model = train(trainEv, seed, nb_epoch, 0.001, layer)

            y_true, y_score, y_pred = test(testEv, model)
            
            results[layer][seed]["y_true"] += y_true
            results[layer][seed]["y_score"] += y_score
            results[layer][seed]["y_pred"] += y_pred
            results[layer][seed]["model_state_dict"][testEv] = model.state_dict()
            
            progress.update(1)
    
    pickle.dump(results, open("results.pickle","wb"))

print("##########")
print("Results saved to: results.pickle")