import gym
import gym_chess
import numpy as np
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim

from MCTS_Agent import MCTSAgent
from SAC_Agent import SACAgent
from ReplayBuffer import ReplayBuffer
from train import train_agents, evaluate_results


def main():
    # Initialize environment and agents
    env = gym.make("Chess-v0")
    mcts_agent = MCTSAgent(num_simulations=50, exploration_weight=1.0)
    sac_agent = SACAgent(state_dim=64, action_dim=4672)  # Adjust dimensions for chess state-action representation
    replay_buffer = ReplayBuffer()

    # Train agents
    results = train_agents(env, mcts_agent, sac_agent, replay_buffer, episodes=500, batch_size=64, train_interval=5)

    # Evaluate results
    evaluate_results(results)


if __name__ == '__main__':
    main()

