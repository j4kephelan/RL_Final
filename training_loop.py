from A2C_Agent import A2CAgent
from ReplayBuffer import ReplayBuffer
from SAC_Agent import SACAgent
import gym
import gym_chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from encoding_states import encode_state
import random
import math

def train_agents(env, a2c_agent, sac_agent, replay_buffer, episodes=1000, batch_size=64, train_interval=5, first_agent="A2C"):
    results = {"a2c_wins": 0, "sac_wins": 0, "draws": 0, "game_lengths": [], "total_rewards": {"a2c": [], "sac": []}}

    for episode in range(episodes):
        state = env.reset()
        done = False
        # Set the initial turn based on who should go first
        turn = 0 if first_agent == "A2C" else 1  # If A2C goes first, start with turn = 0, otherwise SAC goes first
        episode_length = 0
        total_rewards = {"a2c": 0, "sac": 0}

        while not done and episode_length < 500:
            prev_env = encode_state(env)
            # Determine which agent's turn
            if turn == 0:  # A2C Agent
                action, action_index = a2c_agent.select_action(env)
            else:  # SAC Agent
                action, action_index = sac_agent.select_action(env)

            # Step the environment
            next_state, reward, done, info = env.step(action)

            # Custom reward logic
            if done or episode_length >= 499:
                if reward == 0:  # If the game ended in a draw
                    reward = -1  # Assign -1 for a draw
                else:  # Win or loss
                    # Use the given reward formula for the winner
                    reward = math.exp((-episode_length / 50) + 2)

            # Update total rewards for each agent
            if turn == 0:  # A2C's turn
                total_rewards["a2c"] += reward
                if reward > 0:  # A2C wins
                    total_rewards["sac"] -= 2  # SAC loses
                elif reward == -1:  # Draw
                    total_rewards["a2c"] -= 1  # A2C loses in draw
                    total_rewards["sac"] -= 1  # SAC loses in draw
            else:  # SAC's turn
                total_rewards["sac"] += reward
                if reward > 0:  # SAC wins
                    total_rewards["a2c"] -= 2  # A2C loses
                elif reward == -1:  # Draw
                    total_rewards["a2c"] -= 1  # A2C loses in draw
                    total_rewards["sac"] -= 1  # SAC loses in draw

            replay_buffer_next_state = encode_state(env)
            # A2C-specific updates
            if turn == 0:  # A2C's turn
                a2c_agent.store_transition(prev_env, action_index, reward,
                                           replay_buffer_next_state, done)
                a2c_agent.train()

            # SAC-specific updates
            if turn == 1:  # SAC's turn
                replay_buffer.store(prev_env, action_index, reward, replay_buffer_next_state, done)
                if replay_buffer.size() >= batch_size:
                    sac_agent.train(replay_buffer, batch_size)
            # Update state and turn
            turn = 1 - turn  # Alternate turns
            episode_length += 1

        # Handle the end of the episode
            # Handle the end of the episode
            if done:
                if reward == -1:  # Draw
                    results["draws"] += 1
                elif reward > 0:  # A win for the agent who took the last move
                    if turn == 1:  # A2C just moved
                        results["a2c_wins"] += 1
                    else:
                        results["sac_wins"] += 1
                else:  # If reward < 0, it means the opponent won
                    if turn == 1:  # SAC just moved
                        results["sac_wins"] += 1
                    else:
                        results["a2c_wins"] += 1

        results["game_lengths"].append(episode_length)
        results["total_rewards"]["a2c"].append(total_rewards["a2c"])
        results["total_rewards"]["sac"].append(total_rewards["sac"])

        first_agent = "A2C" if first_agent == "SAC" else "SAC"

        # Print progress
        winner = 'A2C' if reward > 0 and turn == 1 else 'SAC' if reward > 0 else 'Draw'
        win_stmt = f'Winner: {winner}' if winner != "Draw" else 'Draw'
        print(f"Episode {episode + 1}/{episodes} - {win_stmt} in {episode_length} moves, for {reward}")

    return results


def evaluate_results(results):
    total_games = results["a2c_wins"] + results["sac_wins"] + results["draws"]
    print("Final Results:")
    print(f"A2C Wins: {results['a2c_wins']} ({(results['a2c_wins'] / total_games) * 100:.2f}%)")
    print(f"SAC Wins: {results['sac_wins']} ({(results['sac_wins'] / total_games) * 100:.2f}%)")
    print(f"Draws: {results['draws']} ({(results['draws'] / total_games) * 100:.2f}%)")
    print(f"Average Game Length: {np.mean(results['game_lengths']):.2f}")
    print(f"Average A2C Reward: {np.mean(results['total_rewards']['a2c']):.2f}")
    print(f"Average SAC Reward: {np.mean(results['total_rewards']['sac']):.2f}")


if __name__ == "__main__":
    random.seed(1234)

    # Initialize environment and agents
    env = gym.make("Chess-v0")

    # A2C Agent
    a2c_agent = A2CAgent(state_dim=64, action_dim=4672)  # Adjust dimensions for chess state-action representation

    # SAC Agent
    sac_agent = SACAgent(state_dim=64, action_dim=4672)  # Adjust dimensions for chess state-action representation

    # Replay buffer for SAC
    replay_buffer = ReplayBuffer()

    # Train agents
    results = train_agents(env, a2c_agent, sac_agent, replay_buffer, episodes=500, batch_size=64, train_interval=5)

    # Evaluate results
    evaluate_results(results)
