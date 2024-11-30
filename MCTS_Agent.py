import gym
import gym_chess
import numpy as np
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim


class MCTSAgent:
    def __init__(self, num_simulations=50, exploration_weight=1.0):
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight

    def select_move(self, env, move_stack):
        legal_moves = list(env.legal_moves)

        if not legal_moves:
            return None

        scores = defaultdict(float)
        visits = defaultdict(int)

        for _ in range(self.num_simulations):
            move = random.choice(legal_moves)
            reward = self.simulate(move, move_stack)
            scores[move] += reward
            visits[move] += 1

        # UCT formula: Exploit vs. Explore
        def uct_score(uct_move):
            try:
                return (scores[uct_move] / visits[uct_move]) + self.exploration_weight * np.sqrt(np.log(sum(visits.values())) / visits[uct_move])
            except:
                return float('inf')

        uct_values = {move: uct_score(move) for move in legal_moves}

        return max(uct_values, key=uct_values.get)

    def simulate(self, move, move_stack):
        env = gym.make('Chess-v0')  # Reinitialize environment
        env.reset()  # Start fresh

        # Replicate the state from the original environment
        for past_move in move_stack:
            env.step(past_move)

        # Apply the move to simulate
        env.step(move)

        # Perform random playout to completion
        done = False
        total_reward = 0
        while not done:
            legal_moves = list(env.legal_moves)
            if not legal_moves:  # No legal moves means game over
                break
            random_move = random.choice(legal_moves)
            _, reward, done, _ = env.step(random_move)
            total_reward += reward

        return total_reward
