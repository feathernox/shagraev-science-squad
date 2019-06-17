import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def get_mnist_models(embedding_size=256):
    sizes = [4704, 2352, 1024]
    models = []
    for i in range(3):
        model = nn.Sequential()
        model.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=24, padding=2, kernel_size=5))
        model.add_module('relu_conv1', nn.ReLU())
        model.add_module('bn_conv1', nn.BatchNorm2d(24))
        model.add_module('pool1', nn.MaxPool2d(2))
        
        if i > 0:
            model.add_module('conv2', nn.Conv2d(in_channels=24, out_channels=48, padding=2, kernel_size=5))
            model.add_module('relu_conv2', nn.ReLU())
            model.add_module('bn_conv2', nn.BatchNorm2d(48))
            model.add_module('pool2', nn.MaxPool2d(2))
            
        if i > 1:
            model.add_module('conv3', nn.Conv2d(in_channels=48, out_channels=64, padding=2, kernel_size=5))
            model.add_module('relu_conv3', nn.ReLU())
            model.add_module('bn_conv3', nn.BatchNorm2d(64))
            model.add_module('pool3', nn.MaxPool2d(2, stride=2, padding=1))
            
        model.add_module('flatten', Flatten())
        model.add_module('linear1', nn.Linear(sizes[i], embedding_size))
        model.add_module('relu_linear1', nn.ReLU())
        model.add_module('bn_linear1', nn.BatchNorm1d(embedding_size))
        model.add_module('logits', nn.Linear(embedding_size, 10))
        models.append(model)
    return models
    
def compute_loss(model, X_batch, y_batch):
    X_batch = Variable(torch.FloatTensor(X_batch)).cuda()
    y_batch = Variable(torch.LongTensor(y_batch)).cuda()
    logits = model(X_batch)
    return F.cross_entropy(logits, y_batch).sum()

def compute_loss_and_acc(model, X_batch, y_batch):
    X_batch = Variable(torch.FloatTensor(X_batch)).cuda()
    y_batch = Variable(torch.LongTensor(y_batch)).cuda()
    logits = model(X_batch)
    loss = F.cross_entropy(logits, y_batch).sum().item()
    matches = np.sum((y_batch == logits.max(1)[1]).cpu().numpy())
    return loss, matches

def evaluate_accuracy(model, test_batch_gen):
    model.train(False) # disable dropout / use averages for batch_norm
    test_accuracy = 0
    test_loss = 0
    for X_batch, y_batch in test_batch_gen:
        loss, acc = compute_loss_and_acc(model, X_batch, y_batch)
        test_accuracy += acc
        test_loss += loss
    test_accuracy = test_accuracy / len(test_batch_gen.dataset)
    test_loss = test_loss / len(test_batch_gen.dataset)
    return test_loss, test_accuracy

def train_neural_network(train_subset, val_subset, model, opt, callback, batch_size=32,
                         n_epochs=100):
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    for _ in range(n_epochs):
        model.train(True)
        for X_batch, y_batch in train_loader:
            loss = compute_loss(model, X_batch, y_batch)
            loss.backward()
            opt.step()
            opt.zero_grad()
            callback.on_batch_end(loss.cpu().item())
            
        val_loss, val_acc = evaluate_accuracy(model, val_loader)
        flag_break = callback.on_epoch_end(model, val_loss, val_acc)
        if flag_break:
            break
    
    model.load_state_dict(torch.load(callback.path_model))
    return model
