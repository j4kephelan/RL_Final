
from A2C_Agent import A2CAgent
from ReplayBuffer import ReplayBuffer
from SAC_Agent import SACAgent
from Random_Agent import RandomAgent
from Greedy_Agent import GreedyAgent
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from encoding_states import encode_state
import random
import math
from training_loop import train_agents, evaluate_results
from plot import plot_results
import time
from concurrent.futures import ProcessPoolExecutor


def visualize_results(results, agent1, agent2):
  evaluate_results(results, agent1.name, agent2.name)
  plot_results(results, agent1.name, agent2.name)


def train_and_visualize(agent, opponent, agent_name, opponent_name, episodes):
    env = gym.make("Chess-v0")
    replay_buffer = ReplayBuffer()
    results = train_agents(env, agent, opponent, replay_buffer, episodes=episodes, batch_size=64)
    visualize_results(results, agent, opponent)
    return results

def train_agents_parallel(agent_opponent_pairs, episodes):
    results = []
    with ProcessPoolExecutor() as executor:
        tasks = [
            executor.submit(train_and_visualize, agent, opponent, agent_name, opponent_name, episodes)
            for (agent, opponent, agent_name, opponent_name) in agent_opponent_pairs
        ]
        for future in tasks:
            results.append(future.result())
    return results

def main_training_pipeline():
    start_time = time.time()

    # Create independent agents for isolation
    a2c_agent = A2CAgent(state_dim=64, action_dim=4672)
    sac_agent = SACAgent(state_dim=64, action_dim=4672)
    greedy_agent = GreedyAgent()
    random_agent = RandomAgent()

    # Phase 1: Train against the GreedyAgent
    print("Phase 1: Training against GreedyAgent...")
    greedy_phase_pairs = [
        (A2CAgent(state_dim=64, action_dim=4672), greedy_agent, "A2CAgent", "GreedyAgent"),
        (SACAgent(state_dim=64, action_dim=4672), greedy_agent, "SACAgent", "GreedyAgent"),
    ]
    train_agents_parallel(greedy_phase_pairs, episodes=10)

    # Phase 2: Train against the RandomAgent
    print("Phase 2: Training against RandomAgent...")
    random_phase_pairs = [
        (A2CAgent(state_dim=64, action_dim=4672), random_agent, "A2CAgent", "RandomAgent"),
        (SACAgent(state_dim=64, action_dim=4672), random_agent, "SACAgent", "RandomAgent"),
    ]
    train_agents_parallel(random_phase_pairs, episodes=10)

    # Phase 3: Have A2CAgent and SACAgent face each other
    print("Phase 3: A2CAgent vs SACAgent...")
    env = gym.make("Chess-v0")
    replay_buffer = ReplayBuffer()
    results = train_agents(env, a2c_agent, sac_agent, replay_buffer, episodes=10, batch_size=64)
    visualize_results(results, a2c_agent, sac_agent)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time}")

main_training_pipeline()
