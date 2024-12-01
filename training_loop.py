from A2C_Agent import A2CAgent
from ReplayBuffer import ReplayBuffer
from SAC_Agent import SACAgent
import gym
import gym_chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 


def train_agents(env, a2c_agent, sac_agent, replay_buffer, episodes=1000, batch_size=64, train_interval=5):
    results = {"a2c_wins": 0, "sac_wins": 0, "draws": 0, "game_lengths": [], "total_rewards": {"a2c": [], "sac": []}}

    for episode in range(episodes):
        state = env.reset()
        done = False
        turn = 0  # 0 for A2C, 1 for SAC
        episode_length = 0
        total_rewards = {"a2c": 0, "sac": 0}

        while not done:
            # Determine which agent's turn
            if turn == 0:  # A2C Agent
                action = a2c_agent.select_action(state)
            else:  # SAC Agent
                action = sac_agent.select_action(state)

            # Step the environment
            next_state, reward, done, info = env.step(action)

            # Update rewards
            if turn == 0:
                total_rewards["a2c"] += reward
            else:
                total_rewards["sac"] += reward

            # A2C-specific updates
            if turn == 0:  # A2C's turn
                a2c_agent.store_transition(state, action, reward, next_state, done)
                a2c_agent.train()

            # SAC-specific updates
            if turn == 1:  # SAC's turn
                replay_buffer.store(state, action, reward, next_state, done)
                if replay_buffer.size() >= batch_size and episode % train_interval == 0:
                    sac_agent.train(replay_buffer, batch_size)

            # Update state and turn
            state = next_state
            turn = 1 - turn  # Alternate turns
            episode_length += 1

        # Record game result
        if reward > 0:  # Last move resulted in a win
            if turn == 1:  # A2C just moved
                results["a2c_wins"] += 1
            else:
                results["sac_wins"] += 1
        else:  # Draw
            results["draws"] += 1

        results["game_lengths"].append(episode_length)
        results["total_rewards"]["a2c"].append(total_rewards["a2c"])
        results["total_rewards"]["sac"].append(total_rewards["sac"])

        # Print progress
        print(
            f"Episode {episode + 1}/{episodes} - Winner: {'A2C' if reward > 0 and turn == 1 else 'SAC' if reward > 0 else 'Draw'}")

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
