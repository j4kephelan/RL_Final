import copy
import numpy as np
from encoding_states import encode_state


class GreedyAgent:
    def __init__(self):
        self.name = "greedy"

    def get_piece_value(self, piece):
        """
        Returns the point value of a given piece.
        """
        piece_values = {
            1: 1,   # White pawn
            2: 5,   # White rook
            3: 3,   # White knight
            4: 3,   # White bishop
            5: 9,   # White queen
            6: float('inf'),  # White king (invaluable)
            -1: 1,  # Black pawn
            -2: 5,  # Black rook
            -3: 3,  # Black knight
            -4: 3,  # Black bishop
            -5: 9,  # Black queen
            -6: float('inf'),  # Black king (invaluable)
        }
        return piece_values.get(piece, 0)  # Default to 0 for empty squares or unknown values

    def get_reward(self, board_before, board_after):
        """
        Calculate the reward for a move based on the piece captured.
        """
        # Identify the captured piece by comparing the before and after states
        for before, after in zip(board_before, board_after):
            if before != 0 and after == 0:  # A piece was removed
                return self.get_piece_value(before)

        # No piece was captured
        return 0

    def select_action(self, env):
        """
        Selects the best action based on potential reward from capturing a piece.
        """
        legal_moves = list(env.legal_moves)  # Get all legal moves from the environment
        if not legal_moves:
            return None  #

        best_move = None
        best_reward = float('-inf')

        board_before = encode_state(env)

        for action in legal_moves:

            test_env = copy.deepcopy(env)  # Create a deep copy of the environment to simulate the action
            test_env.step(action)  # Apply the move

            board_after = encode_state(test_env)

            reward = self.get_reward(board_before, board_after)
            if reward > best_reward:
                best_reward = reward
                best_move = action

        # Return the best move and its index
        return best_move, legal_moves.index(best_move) if best_move else None

    def train(self, replay_buffer, batch_size=64):
        pass
