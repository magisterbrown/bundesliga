import torch.nn as nn
import torch
def train_one_epoch(model, optimizer, data_loader):
    model.train()
    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        res = model(data)
        loss = resp_loss(res, target)
        print(loss)
        loss.backward()
        optimizer.step()
        
