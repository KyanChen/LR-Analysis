import json

import numpy as np
import torch
from matplotlib import pyplot as plt

weights = torch.load('../../checkpoints/last.pth')
data = json.load(open('../../data/data.json'))
one_item = data[list(data.keys())[0]]

item_keys = []
for key, value in one_item.items():
    if isinstance(value, list):
        item_keys += [f'{key}_{i}' for i in range(len(value))]
    else:
        item_keys += [key]
item_keys = item_keys[:-1]
importances = weights['lr_layer.weight'].numpy()
# importances = (importances - np.min(importances))/ (np.max(importances) - np.min(importances))
importances = importances.reshape(-1)
# bar_width = 1
# pos = np.arange(len(importances))+bar_width/2
plt.figure(figsize=(20, 15))
for i in range(len(importances)):
    plt.barh(item_keys[i], importances[i])

# plt.figure(figsize=(16, 4))
# plt.bar(np.arange(len(importances)), importances, align='center')
plt.show()