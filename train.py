import json
import random
from torchmetrics.functional.classification import precision_recall
import torch
import torch.nn as nn


class LR(nn.Module):
    def __init__(self, in_channel=6):
        super(LR, self).__init__()
        self.lr_layer = nn.Linear(in_channel, 1)

    def forward(self, x):
        x = self.lr_layer(x)
        x = torch.sigmoid(x)
        return x


def train():
    model = LR(15)
    criteria = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.load_state_dict(torch.load('checkpoints/last.pth'))

    keys = json.load(open('data/train_val_split.json'))
    data = json.load(open('data/data.json'))
    train_keys = keys['train']

    # {2: 243, 3: 1419}
    neg_keys = []
    for key in train_keys:
        if data[key]['quality'] == 2:
            neg_keys.append(key)
    train_keys += 5*neg_keys

    val_keys = keys['val']

    class_mapping = {2: 0, 3: 1}
    batch_size = 32
    for i_epoch in range(3000):
        random.shuffle(train_keys)
        model.train()
        for i_iter in range(len(train_keys)//batch_size):
            iter_keys = train_keys[i_iter*batch_size:(i_iter+1)*batch_size]
            inputs = []
            outputs = []
            for key in iter_keys:
                values = list(data[key].values())
                gt = values[-1]
                values = values[:-1]
                re_values = []
                for v in values:
                    if isinstance(v, list):
                        re_values += v
                    else:
                        re_values += [v]
                inputs.append(re_values)
                outputs.append([class_mapping[gt]])
            inputs = torch.tensor(inputs)
            outputs = torch.tensor(outputs)
            pred = model(inputs)
            loss = criteria(pred, outputs.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss.item(), sum(torch.eq(pred > 0.5, outputs)) / len(outputs))

        # eval
        model.eval()
        pred_gt = []
        output_gt = []
        for i_iter in range(len(val_keys)//batch_size):
            iter_keys = val_keys[i_iter*batch_size:(i_iter+1)*batch_size]
            inputs = []
            outputs = []
            for key in iter_keys:
                values = list(data[key].values())
                gt = values[-1]
                values = values[:-1]
                re_values = []
                for v in values:
                    if isinstance(v, list):
                        re_values += v
                    else:
                        re_values += [v]
                inputs.append(re_values)
                outputs.append([class_mapping[gt]])
            inputs = torch.tensor(inputs)
            outputs = torch.tensor(outputs)
            output_gt.append(outputs)
            with torch.no_grad():
                pred = model(inputs)
            pred_gt.append(pred > 0.5)
        pred_gt = torch.cat(pred_gt)
        output_gt = torch.cat(output_gt)
        # print(pred_gt)
        # pred_gt = torch.ones_like(output_gt)
        pr = precision_recall(pred_gt, output_gt)
        print(pr)
        # print(sum(torch.eq(pred_gt, output_gt)) / len(pred_gt))

        torch.save(model.state_dict(), 'checkpoints/last.pth')


if __name__ == '__main__':
    train()

