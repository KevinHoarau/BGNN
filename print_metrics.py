import pickle
import numpy as np
import pandas as pd
import torch
import io
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

results = CPU_Unpickler(open("results.pickle","rb")).load()

metrics = []

for layer in results.keys():
    
    accs = []
    f1s = []
    aucs = []

    for seed, data in results[layer].items():

        accs.append(accuracy_score(data["y_true"], data["y_pred"]))
        f1s.append(f1_score(data["y_true"], data["y_pred"]))
        aucs.append(roc_auc_score(data["y_true"], data["y_score"]))
        

    metrics.append({
        "layer": layer,
        "Accuracy (Mean)": np.array(accs).mean(),
        "Accuracy (Std dev)": np.array(accs).std(),
        "F1 score (Mean)": np.array(f1s).mean(),
        "F1 score (Std dev)": np.array(f1s).std(),
        "Auc (Mean)": np.array(aucs).mean(),
        "Auc (Std dev)": np.array(aucs).std(),
    })

metrics = pd.DataFrame(metrics)

print(metrics.to_string())

