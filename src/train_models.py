import os, torch

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from joblib import Parallel, delayed
from CNN import CNN
from MCTS import Monte_Carlo_Tree_Search

BOARD_SIZE = 5
N_JOBS = 6
TRAINING_DEPTH = 10
N_GAMES = 100
N_ITERATIONS = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CNN()   
model.to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

path = f"models/{N_GAMES}-games"
if not os.path.exists(path): os.makedirs(path)

train_accu = []
train_losses = []

def generate_game(model=None):
    mcts = Monte_Carlo_Tree_Search(BOARD_SIZE, model)
    mcts.run(N_ITERATIONS)
    x, y = mcts.get_tree_data()
    return x, y

for i in range(TRAINING_DEPTH):
    if i == 0: games = Parallel(n_jobs=N_JOBS)(delayed(generate_game)() for _ in range(N_GAMES))
    else: games = Parallel(n_jobs=N_JOBS)(delayed(generate_game)(model) for _ in range(N_GAMES))

    x, y = [], []
    running_loss, correct, total = 0, 0, 0

    for game in games:
        x.extend(game[0])
        y.extend(game[1])

    for j in range(len(x)):
        inputs, labels = torch.tensor(x[i]).to(device), torch.tensor(y[i]).to(device)
        labels = F.softmax(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted = outputs.max()
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(x)
    accu = 100. * correct / total
    train_accu.append(accu)
    train_losses.append(train_loss)
    torch.save(model.state_dict(), path + f"/model-{i}.pth")

plt.plot(train_losses)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Training accuracy')
plt.savefig("accu.png")
plt.plot(train_losses)
plt.xlabel('epoch')
plt.ylabel('losses')
plt.title('Training loss')
plt.savefig("loss.png")