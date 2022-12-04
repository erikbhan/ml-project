#!/usr/bin/env python
# coding: utf-8

from joblib import Parallel, delayed

import time, os, torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from CNN import CNN
from MCTS import Monte_Carlo_Tree_Search

BOARD_SIZE = 5
N_JOBS = 8
TRAINING_DEPTH = 5
N_GAMES = 100
N_ITERATIONS = 1000

def generate_game(x, y, model):
    mcts = Monte_Carlo_Tree_Search(BOARD_SIZE, model)
    mcts.run(N_ITERATIONS)
    a, b = mcts.get_tree_data()
    x.append(a)
    y.append(b)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN()   
model.to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

path = f"models/black_bias/{N_GAMES}-games"
if not os.path.exists(path): os.makedirs(path)

from warnings import filterwarnings
filterwarnings("ignore")

x, y = [], []
print("Generating games...")
Parallel(n_jobs=N_JOBS)(delayed(generate_game)(x, y, None) for _ in range(N_GAMES))
for i in range(TRAINING_DEPTH):
    x, y = [], []
    print("Generating games...")
    start_gen = time.time()
    Parallel(n_jobs=N_JOBS)(delayed(generate_game)(x, y, model) for _ in range(N_GAMES))
    stop_gen = time.time()
    for i in range(len(x)):
        inputs, labels = torch.tensor(x[i], device=device), torch.tensor(y[i], device=device)
        labels = F.softmax(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Generating {N_GAMES} games on {N_JOBS} processes took {stop_gen - start_gen} seconds")
    print(f"Saving model as '{path}/model-{i}.pth'")
    torch.save(model.state_dict(), path + f"/model-{i}.pth")