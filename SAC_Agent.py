import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict


def parse_unicode_state(unicode_str):
    # Mapping from Unicode pieces to integers
    piece_map = {
        '♙': 1,  # White pawn
        '♖': 2,  # White rook
        '♘': 3,  # White knight
        '♗': 4,  # White bishop
        '♕': 5,  # White queen
        '♔': 6,  # White king
        '♟': -1,  # Black pawn
        '♜': -2,  # Black rook
        '♞': -3,  # Black knight
        '♝': -4,  # Black bishop
        '♛': -5,  # Black queen
        '♚': -6,  # Black king
        '⭘': 0,  # Empty square (if using ⭘ for empty)
        ' ': 0,  # Empty square (if using space for empty)
    }

    state = []

    # Split the unicode_str into rows and process each row
    rows = unicode_str.splitlines()
    for row in rows:
        for char in row:
            # Append the numerical value corresponding to the piece in the square
            state.append(piece_map.get(char, 0))  # Default to 0 if unknown character

    # Ensure the state is a 1D array with 64 elements
    return np.array(state[:64])  # Cut off any excess data (if any) to keep the board size 64


class SACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Actor-Critic Networks
        self.actor = self.build_actor()
        self.critic1 = self.build_critic()
        self.critic2 = self.build_critic()
        self.target_critic1 = self.build_critic()
        self.target_critic2 = self.build_critic()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.update_target_networks(tau=1.0)  # Hard copy initially

    def build_actor(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh()  # Ensure output between -1 and 1 for continuous action space
        )

    def build_critic(self):
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def update_target_networks(self, tau=None):
        tau = self.tau if tau is None else tau
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def select_action(self, env):
        # Render the board to unicode string

        unicode_str = env.render(mode='unicode')

        # Convert the unicode string to numerical representation
        encoded_env = parse_unicode_state(unicode_str)

        # Ensure the state is a 1D array of size 64
        assert encoded_env.shape == (64,), f"Expected state shape (64,), got {encoded_env.shape}"

        # Convert the encoded state to a PyTorch tensor and add batch dimension
        encoded_env = torch.FloatTensor(encoded_env).unsqueeze(0)

        # Select action using the actor (continuous vector output)
        action = self.actor(encoded_env).detach().numpy()[0]

        # Get the list of legal moves from the environment
        legal_moves = list(env.legal_moves)

        if not legal_moves:
            return None  # If no legal moves, return None (checkmate or stalemate)

        # Scale the continuous action to a valid legal move index
        move_index = int(np.clip(action[0] * len(legal_moves), 0, len(legal_moves) - 1))  # Map to a valid index

        # Select the corresponding legal move from the list
        return legal_moves[move_index]

        # # Convert the move to algebraic notation
        # move_str = self.convert_move_to_notation(selected_move)
        #
        # return move_str

    def convert_move_to_notation(self, move):
        # Convert the move (e.g., chess.Move) to algebraic notation
        return move.uci()  # UCI (Universal Chess Interface) format is like 'e2e4', 'b1c3'


    def train(self, replay_buffer, batch_size=64):
        # Training logic here (similar to the previous implementation)
        pass

