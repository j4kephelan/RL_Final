import matplotlib.pyplot as plt
import numpy as np

def plot_episode_lengths(game_lengths):
    """
    Plots the lengths of episodes over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(game_lengths) + 1), game_lengths, label="Episode Length", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.title("Episode Lengths Over Episodes")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_total_rewards(total_rewards, agent1_name, agent2_name):
    """
    Plots the total rewards of two agents over episodes.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(total_rewards[agent1_name]) + 1),
        total_rewards[agent1_name],
        label=f"{agent1_name} Total Rewards",
        color="green"
    )
    plt.plot(
        range(1, len(total_rewards[agent2_name]) + 1),
        total_rewards[agent2_name],
        label=f"{agent2_name} Total Rewards",
        color="red"
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Rewards Over Episodes")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_avg_wins_draws(agent1_wins, agent2_wins, draws, agent1_name, agent2_name):
    """
    Plots the average wins and draws as percentages.
    """
    total_games = agent1_wins + agent2_wins + draws
    averages = {
        f"{agent1_name} wins": agent1_wins / total_games * 100,
        f"{agent2_name} wins": agent2_wins / total_games * 100,
        "Draws": draws / total_games * 100,
    }

    plt.figure(figsize=(8, 6))
    plt.bar(averages.keys(), averages.values(), color=["green", "red", "blue"])
    plt.xlabel("Category")
    plt.ylabel("Percentage (%)")
    plt.title("Average Wins and Draws")
    plt.grid(axis="y")
    plt.show()

def plot_results(results, agent1_name, agent2_name):
    """
    Combines the individual plots for episode lengths, total rewards, and average wins/draws.
    """
    plot_episode_lengths(results["game_lengths"])
    plot_total_rewards(results["total_rewards"], agent1_name, agent2_name)
    plot_avg_wins_draws(
        results[f"{agent1_name} wins"],
        results[f"{agent2_name} wins"],
        results["draws"],
        agent1_name,
        agent2_name
    )
