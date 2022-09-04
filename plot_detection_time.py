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

results = CPU_Unpickler(open("results_detection_time.pickle","rb")).load()

metrics = []

for w in results.keys():
    
    accs = []
    f1s = []
    aucs = []

    for seed, data in results[w].items():

        accs.append(accuracy_score(data["y_true"], data["y_pred"]))
        f1s.append(f1_score(data["y_true"], data["y_pred"]))
        aucs.append(roc_auc_score(data["y_true"], data["y_score"]))
        

    metrics.append({
        "Time": (w-15)*2,
        "Accuracy (Mean)": np.array(accs).mean(),
        "Accuracy (Std dev)": np.array(accs).std(),
        "F1 score (Mean)": np.array(f1s).mean(),
        "F1 score (Std dev)": np.array(f1s).std(),
        "Auc (Mean)": np.array(aucs).mean(),
        "Auc (Std dev)": np.array(aucs).std(),
    })

metrics = pd.DataFrame(metrics)

print(metrics.to_string())

data_plot = {"x":[],"y":[], "Metrics":[]}

for w in results.keys():
    
    for seed, data in results[w].items():
        data_plot["Metrics"].append("Accuracy")
        data_plot["x"].append(w)
        data_plot["y"].append(accuracy_score(data["y_true"], data["y_pred"]))
        
        data_plot["Metrics"].append("F1 score")
        data_plot["x"].append(w)
        data_plot["y"].append(f1_score(data["y_true"], data["y_pred"]))
        
        data_plot["Metrics"].append("AUC")
        data_plot["x"].append(w)
        data_plot["y"].append(roc_auc_score(data["y_true"], data["y_score"]))

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5,4))

fts = 12

sns.set_style("whitegrid")

palette = palette = sns.color_palette("mako", 3)

ax = sns.lineplot(x='x',y='y',data=data_plot, hue="Metrics", ci="sd", palette=palette)

ax.set_yticks(np.arange(0,1.1,0.1))
ax.set_ylim(0.2,1.1)
ax.set_xticks(range(16,31))
ax.set_xticklabels(range(2,31,2))

ax.set_xlabel("Time [minutes]", fontsize=fts)
ax.set_ylabel("Value", fontsize=fts)

ax.margins(x=0.01)

plt.savefig('detection_time.png', format='png', bbox_inches='tight')

print("Figure saved as: detection_time.png")