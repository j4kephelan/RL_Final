import numpy as np


def train_agents(env, mcts_agent, sac_agent, replay_buffer, episodes=1000, batch_size=64, train_interval=5):
    results = {"mcts_wins": 0, "sac_wins": 0, "draws": 0, "game_lengths": [], "total_rewards": {"mcts": [], "sac": []}}

    for episode in range(episodes):
        move_stack = []
        state = env.reset()
        done = False
        turn = 0  # 0 for MCTS, 1 for SAC
        episode_length = 0
        total_rewards = {"mcts": 0, "sac": 0}

        while not done:
            # Determine which agent's turn
            if turn == 0:
                print('MCTS choosing move')
                action = mcts_agent.select_move(env, move_stack)
                move_stack.append(action)
            else:
                print('SAC choosing move')
                action = sac_agent.select_action(env)
                move_stack.append(action)

            # Step the environment
            print('Taking action')
            print(action, type(action))

            next_state, reward, done, info = env.step(action)

            # Track rewards
            if turn == 0:
                total_rewards["mcts"] += reward
            else:
                total_rewards["sac"] += reward

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
            if turn == 1:  # MCTS just moved
                results["mcts_wins"] += 1
            else:
                results["sac_wins"] += 1
        else:  # Draw
            results["draws"] += 1

        results["game_lengths"].append(episode_length)
        results["total_rewards"]["mcts"].append(total_rewards["mcts"])
        results["total_rewards"]["sac"].append(total_rewards["sac"])

        # Print progress
        print(
            f"Episode {episode + 1}/{episodes} - Winner: {'MCTS' if reward > 0 and turn == 1 else 'SAC' if reward > 0 else 'Draw'}")

    return results


def evaluate_results(results):
    total_games = results["mcts_wins"] + results["sac_wins"] + results["draws"]
    print("Final Results:")
    print(f"MCTS Wins: {results['mcts_wins']} ({(results['mcts_wins'] / total_games) * 100:.2f}%)")
    print(f"SAC Wins: {results['sac_wins']} ({(results['sac_wins'] / total_games) * 100:.2f}%)")
    print(f"Draws: {results['draws']} ({(results['draws'] / total_games) * 100:.2f}%)")
    print(f"Average Game Length: {np.mean(results['game_lengths']):.2f}")
    print(f"Average MCTS Reward: {np.mean(results['total_rewards']['mcts']):.2f}")
    print(f"Average SAC Reward: {np.mean(results['total_rewards']['sac']):.2f}")