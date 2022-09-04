import pickle
from torch_geometric.utils import from_networkx
from tqdm.auto import tqdm

def loadGraphs(filepath):
    
    graphs = []
    for g in pickle.load(open(filepath, "rb")):
        graphs.append(from_networkx(g))

    return(graphs)
    

def loadData(folder):
    
    d = {}
    
    names = ["TTNet", "IndoSat", "TM", "AWS", "Google","ChinaTelecom","India"]
    nodes = ["9121", "4761", "4788", "200759", "15169","21217","55410"]
    
    for n, node in tqdm(zip(names,nodes), total=len(names)):
        d[n] = {
            "A" : loadGraphs(folder+"/anomaly/"+n+"/transform/Graph/WeightedGraph_2.pickle"),
            "NA" : loadGraphs(folder+"/no_anomaly/"+n+"/transform/Graph/WeightedGraph_2.pickle"),
            "node" : node
        }
        
    return(d)

print("Transforming data to PyTorch Geometric")
print("It may take several minutes, please wait ...")
    
data = loadData("data/")
filename = "data_pyg.pickle"

pickle.dump(data, open(filename,"wb"))

print("Data saved to: "+ filename)