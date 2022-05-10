import matplotlib.pyplot as plt
import numpy as np
import pickle
from torch import nn

def normalize(v):
    # v : numpy vector
    v = v - v.min()
    v = v / v.max()
    return v

human_sim = [3.92, 3.38, 3.46, 3.46, 3.50, 1.50, 1.50, 4.00, 2.04, 1.96, 2.31, 3.00, 2.46, 4.15, 
 2.73, 4.15, 4.50, 4.42, 4.46, 5.88, 4.35, 3.00, 4.31, 2.12, 1.92, 3.73, 5.00, 4.73, 
 4.85, 3.31, 2.77, 5.04, 3.77, 3.46, 3.04, 2.35, 3.92, 3.92, 3.73, 1.12, 2.46, 3.23, 
 1.46, 1.12, 1.88, 3.42, 4.58, 1.85]

with open('similarities.pkl', 'rb') as f:
    similarities = pickle.load(f)

with open(f'model_dict_style.pkl', 'rb') as f:
    model_dict_style = pickle.load(f)




###### FIG 1: CORRELATION PLOT ######
fig, ax = plt.subplots(1, 3, figsize=(19,5))
for i, model in enumerate(model_dict_style.keys()):
    v_sim_net_norm = normalize(similarities[model])
    v_sim_human_norm = normalize(np.array(human_sim))

    r = np.corrcoef(v_sim_net_norm,v_sim_human_norm)[0][1]

    # Scatter plot
    ax[i].set_title(f'{model} - r={round(r,3)}', fontsize=16)
    ax[i].scatter(v_sim_net_norm,v_sim_human_norm)

fig.text(0.5, 0.02, 'network similarity (normalized)', ha='center', fontsize=12)
fig.text(0.09, 0.5, 'human similarity (normalized)', va='center', rotation='vertical', fontsize=12)
plt.savefig('corr_plot.png')


###### FIG 1: TRAIN/VAL LOSS PLOT ######
fig, ax = plt.subplots(1, 2, figsize=(12,5))

for i, (k, v) in enumerate(model_dict_style.items()):
    train_losses, val_losses, test_accuracy = v
    
    ax[0].plot(range(100), train_losses, label=k)
    ax[1].plot(range(100), val_losses, label=k)
    
    ax[0].set_title(f'Train Losses', fontsize=16)
    ax[1].set_title(f'Validation Losses', fontsize=16)
    
    ax[0].legend()
    ax[1].legend()
    
fig.text(0.5, 0.02, 'Num Epochs', ha='center', fontsize=12)
fig.text(0.06, 0.5, 'Loss', va='center', rotation='vertical', fontsize=12)
plt.savefig('loss_plot.png')

for k, v in model_dict_style.items():
    print(f'Test Accuracy for {k}: {v[2]}')