import json
import random

data = json.load(open('../../data/data.json'))
keys = list(set(data.keys()))
random.shuffle(keys)
val_keys = keys[:300]
train_keys = keys[300:]
split_keys = {'train': train_keys, 'val': val_keys}
json.dump(split_keys, open('../../data/train_val_split.json', 'w'), indent=4)
