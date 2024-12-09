import random

class RandomAgent:
    def __init__(self):
        self.name = "random"

    def select_action(self, env):
        """
        Selects a random legal move from the environment.
        
        Args:
            env: The chess environment with a method to get legal moves.
        
        Returns:
            A random legal move.
        """
        legal_moves = list(env.legal_moves)  # Get all legal moves from the environment
        if not legal_moves:
            return None  # No moves available (e.g., checkmate or stalemate)
        
        selected_action = random.choice(legal_moves)
        action_index = legal_moves.index(selected_action)

        return selected_action, action_index

    def train(self, replay_buffer, batch_size=64):
        pass
