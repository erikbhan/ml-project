#!/usr/bin/env python
# coding: utf-8

from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.CNN import CNN
from src.MCTS import Monte_Carlo_Tree_Search

BOARD_SIZE = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_game(x, y, model):
    mcts = Monte_Carlo_Tree_Search(BOARD_SIZE, model)
    # mcts.run_game()
    mcts.run(1000)
    a, b = mcts.get_tree_data()
    x.append(a)
    y.append(b)

def generate_games(x, y, n_games, model):
    print(f"Generating {n_games} games")
    Parallel(n_jobs=10)(delayed(generate_game)(x, y, model) for _ in range(1, n_games))

def train_model(model, criterion, optimizer, x, y):
    print("Training model")
    for i in range(len(x)):
        inputs, labels = torch.tensor(x[i], device=device), torch.tensor(y[i], device=device)
        labels = F.softmax(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

model = CNN()   
model.to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for i in range(5):
    x, y = [], []
    if i == 0: generate_games(x, y, 10, None) # first iteration we don't have a trained model
    else: generate_games(x, y, 10, model)
    train_model(model, criterion, optimizer, x, y)
    torch.save(model.state_dict(), f"models/{i}-times.pth")