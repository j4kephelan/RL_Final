from A2C_Agent import A2CAgent
from ReplayBuffer import ReplayBuffer
from SAC_Agent import SACAgent
from A2C_Agent import A2CAgent
from ReplayBuffer import ReplayBuffer
from SAC_Agent import SACAgent
import gym
import gym_chess
import numpy as np
import torch
import random
import math
from encoding_states import encode_state

def train_agents(env, agent1, agent2, replay_buffer, episodes=1000, batch_size=64):
    first_agent = agent1
    second_agent = agent2

    results = {
        f"{agent1.name} wins": 0,
        f"{agent2.name} wins": 0,
        "draws": 0,
        "game_lengths": [],
        "total_rewards": {agent1.name: [], agent2.name: []}
    }

    for episode in range(episodes):
        turn = 0
        state = env.reset()
        done = False

        episode_length = 0
        total_rewards = {agent1.name: 0, agent2.name: 0}

        while not done and episode_length < 500:
            if (turn % 2 == 0):
              current_agent = agent1
            else:
              current_agent = agent2
              
            prev_env = encode_state(env)

            # Agent selects action
            action, action_index = current_agent.select_action(env)
            # Step environment
            next_state, reward, done, info = env.step(action)

            # Adjust reward logic
            if done or episode_length >= 499:
                if reward == 0:  # Draw
                    reward = -1
                else:
                    reward = math.exp((-episode_length / 50) + 2)

                # Update total rewards
                total_rewards[current_agent.name] += reward

                opponent_name = agent2.name if current_agent == agent1 else agent1.name

                if reward > 0:  # Current agent wins
                    total_rewards[opponent_name] += -reward  # Opponent loses
                elif reward == -1:  # Draw
                    total_rewards[agent1.name] -= 1
                    total_rewards[agent2.name] -= 1
                else:  # Current agent loses (reward < 0)
                    total_rewards[opponent_name] += -reward  # Opponent wins
                    total_rewards[current_agent.name] += reward  # Current agent loses

            replay_buffer_next_state = encode_state(env)

            # Training logic for agents
            if current_agent.name == 'a2c':  # A2C agent
                current_agent.store_transition(prev_env, action_index, reward, replay_buffer_next_state, done)
                current_agent.train()
            if current_agent.name == 'sac':  # SAC agent
                replay_buffer.store(prev_env, action_index, reward, replay_buffer_next_state, done)
                if replay_buffer.size() >= batch_size:
                    current_agent.train(replay_buffer, batch_size)

            turn += 1  # Increment turn counter
            episode_length += 1

        # End of episode results
        if done:
            if reward == -1:
                results["draws"] += 1
            else:
                winning_agent = current_agent.name
                results[f"{winning_agent} wins"] += 1

        results["game_lengths"].append(episode_length)
        results["total_rewards"][agent1.name].append(total_rewards[agent1.name])
        results["total_rewards"][agent2.name].append(total_rewards[agent2.name])

        # Alternate first agent
        first_agent, second_agent = second_agent, first_agent

        # Print progress
        winner = current_agent.name if reward > 0 else "Draw"
        print(f"Episode {episode + 1}/{episodes} - Winner: {winner} in {episode_length} moves with reward {reward}")

    return results


def evaluate_results(results, agent1_name, agent2_name):

    total_games = results[f"{agent1_name} wins"] + results[f"{agent2_name} wins"] + results["draws"]
    print("Final Results:")
    for key in results.keys():
        if key.endswith("wins"):
            print(f"{key}: {results[key]} ({(results[key] / total_games) * 100:.2f}%)")
    print(f"Draws: {results['draws']} ({(results['draws'] / total_games) * 100:.2f}%)")
    print(f"Average Game Length: {np.mean(results['game_lengths']):.2f}")
    for agent, rewards in results["total_rewards"].items():
        print(f"Average {agent} Reward: {np.mean(rewards):.2f}")


