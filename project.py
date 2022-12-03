#!/usr/bin/env python
# coding: utf-8

from warnings import filterwarnings
filterwarnings("ignore")

from joblib import Parallel, delayed
import gym, torch, time

from src.CNN import CNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BOARD_SIZE = 5

players = []
for i in range(5):
    player = CNN()
    player.to(device)
    player.load_state_dict(torch.load(f"models/{i}-times.pth"))
    players.append(player)
print(players)

def play_model_vs_model_no_render(model1 : CNN, model2 : CNN, go_env: gym.Env):
    go_env.reset()
    done = go_env.done
    turn_nr = 0
    while not done:
        weights = model1.forward(go_env.state()).cpu().detach().numpy()
        for i in range(len(weights) - 1):
            action2d = i // BOARD_SIZE, i % BOARD_SIZE, i
            if go_env.state()[3, action2d[0], action2d[1]] == 1:
                weights[i] = 0.0
        _, _, done, _ = go_env.step(weights.argmax())

        if done: continue

        weights = model2.forward(go_env.state()).cpu().detach().numpy()
        for i in range(len(weights) - 1):
            action2d = i // BOARD_SIZE, i % BOARD_SIZE, i
            if go_env.state()[3, action2d[0], action2d[1]] == 1:
                weights[i] = 0.0
        _, _, done, _ = go_env.step(weights.argmax())
        turn_nr += 1
        if turn_nr > 300: break
    return go_env

def play_tournament(wins, draws, games_played, b_w):
    print("Starting a tournament")
    start1 = time.time()
    for player1_index in range(len(players)):
        for player2_index in range(len(players)):
            if player1_index == player2_index: continue
            games_played[player1_index] += 1
            games_played[player2_index] += 1
            p1, p2 = players[player1_index], players[player2_index]
            print(f"  Starting a game between {player1_index} and {player2_index}", end="")
            start = time.time()
            go_env = play_model_vs_model_no_render(p1, p2, gym.make('gym_go:go-v0', size=BOARD_SIZE, komi=0, reward_method='heuristic'))
            stop = time.time()
            if go_env.reward() > 0:
                wins[player1_index] += 1
                b_w[0] += 1
            elif go_env.reward() < 0:
                wins[player2_index] += 1
                b_w[1] += 1
            else:
                draws[player1_index] += 1
                draws[player2_index] += 1
            print(f", reward: {go_env.reward()}, time: {stop-start} sec")
    stop1 = time.time()
    print(f"Tournament took {stop1-start1}")
b_w = [0]*2
wins, draws, games_played = [0]*len(players), [0]*len(players), [0]*len(players)
# Parallel(n_jobs=10)(delayed(play_tournament)(wins, draws, games_played, b_w) for _ in range(1000))
for game_number in range(100):
    play_tournament(wins, draws, games_played, b_w )

print(wins)
print(games_played)
print(draws)
