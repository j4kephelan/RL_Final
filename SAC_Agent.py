import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from encoding_states import encode_state


class SACAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, tau=0.005, alpha=0.2, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Detect device (GPU or CPU)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor-Critic Networks
        self.actor = self.build_actor().to(self.device)
        self.critic1 = self.build_critic().to(self.device)
        self.critic2 = self.build_critic().to(self.device)
        self.target_critic1 = self.build_critic().to(self.device)
        self.target_critic2 = self.build_critic().to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.update_target_networks(tau=1.0)  # Hard copy initially

        self.name = "sac"
        
    def build_actor(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Softmax(dim=-1)  # Softmax for discrete probabilities
        )

    def build_critic(self):
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 256),  # Correct input size
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
        # Encode the current state
        encoded_env = encode_state(env)

        # Ensure encoded state is a numpy array with the correct dtype
        encoded_env = np.array(encoded_env, dtype=np.float32)  # Ensure it's float32
        assert encoded_env.shape == (64,), f"Expected state shape (64,), got {encoded_env.shape}"

        # Convert the encoded state to a PyTorch tensor and add batch dimension
        encoded_env = torch.FloatTensor(encoded_env).unsqueeze(0).to(self.device)

        # Get action probabilities from the actor
        action_probs = self.actor(encoded_env).detach().cpu().numpy()[0]

        # Get the list of legal moves from the environment
        legal_moves = list(env.legal_moves)

        if not legal_moves:
            return None  # If no legal moves, return None (checkmate or stalemate)

        # Convert the legal moves to a mapping of indices
        legal_moves_map = {i: move for i, move in enumerate(legal_moves)}

        # Get the probabilities of legal moves
        legal_probs = np.array([action_probs[i] for i in legal_moves_map.keys()])

        # Normalize the probabilities to ensure they sum to 1
        if legal_probs.sum() == 0:
            legal_probs = np.ones_like(legal_probs)
        legal_probs /= legal_probs.sum()

        # Select the action based on the probabilities of legal moves
        selected_index = np.random.choice(list(legal_moves_map.keys()), p=legal_probs)
        return legal_moves_map[selected_index], selected_index

    def convert_move_to_notation(self, move):
        # Convert the move (e.g., chess.Move) to algebraic notation
        return move.uci()  # UCI (Universal Chess Interface) format is like 'e2e4', 'b1c3'

    def train(self, replay_buffer, batch_size=64):
        # Sample a batch of transitions from the replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Convert to PyTorch tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.nn.functional.one_hot(torch.LongTensor(actions), num_classes=self.action_dim).float().to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute target Q values
        with torch.no_grad():
            next_action_probs = self.actor(next_states)
            next_action_log_probs = torch.log(next_action_probs + 1e-8)
            next_q1 = self.target_critic1(torch.cat([next_states, next_action_probs], dim=1))
            next_q2 = self.target_critic2(torch.cat([next_states, next_action_probs], dim=1))
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_action_log_probs
            target_q = rewards + self.gamma * (1 - dones) * next_q

        # Update critic networks
        current_q1 = self.critic1(torch.cat([states, actions], dim=1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=1))
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor network
        action_probs = self.actor(states)
        log_probs = torch.log(action_probs + 1e-8)
        q1 = self.critic1(torch.cat([states, action_probs], dim=1))
        q2 = self.critic2(torch.cat([states, action_probs], dim=1))
        actor_loss = (self.alpha * log_probs - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_target_networks()
