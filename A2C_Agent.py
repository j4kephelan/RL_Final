import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from encoding_states import encode_state

class A2CAgent:
<<<<<<< HEAD
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, device=None):
=======
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
>>>>>>> 40271a03c060f09b4879cfd10ada693f7ca34eb4
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

<<<<<<< HEAD
        # Set device to GPU if available, otherwise fallback to CPU
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Actor-Critic networks
        self.policy_net = self.build_policy_network(state_dim, action_dim).to(self.device)
        self.value_net = self.build_value_network(state_dim).to(self.device)
=======
        # Actor-Critic networks
        self.policy_net = self.build_policy_network(state_dim, action_dim)
        self.value_net = self.build_value_network(state_dim)
>>>>>>> 40271a03c060f09b4879cfd10ada693f7ca34eb4

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        # Transition storage
        self.transitions = []

<<<<<<< HEAD
        self.name = "a2c"

=======
>>>>>>> 40271a03c060f09b4879cfd10ada693f7ca34eb4
    def build_policy_network(self, state_dim, action_dim):
        """Build the policy network."""
        return nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def build_value_network(self, state_dim):
        """Build the value network."""
        return nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def select_action(self, env):
        """Select an action based on the policy network, restricted to legal moves."""
        # Encode the current state
        encoded_env = encode_state(env)

        # Ensure the state is a 1D array of size 64
        assert len(encoded_env) == 64, f"Expected state size 64, got {len(encoded_env)}"

        # Convert the encoded state to a PyTorch tensor and add batch dimension
<<<<<<< HEAD
        encoded_env = torch.FloatTensor(encoded_env).unsqueeze(0).to(self.device)

        # Get action probabilities from the policy network
        action_probs = self.policy_net(encoded_env).detach().cpu().numpy().squeeze()
=======
        encoded_env = torch.FloatTensor(encoded_env).unsqueeze(0)

        # Get action probabilities from the policy network
        action_probs = self.policy_net(encoded_env).detach().numpy().squeeze()
>>>>>>> 40271a03c060f09b4879cfd10ada693f7ca34eb4

        # Get the list of legal moves from the environment
        legal_moves = list(env.legal_moves)

        if not legal_moves:
            return None  # No legal moves, game over

        # Map legal moves to indices
        legal_moves_map = {i: move for i, move in enumerate(legal_moves)}

        # Filter probabilities to include only legal moves
        legal_probs = np.array([action_probs[i] for i in legal_moves_map.keys()])

        # Replace NaNs with 0s
        legal_probs = np.nan_to_num(legal_probs, nan=0.0)

        # Handle case where all probabilities are zero (could happen due to a bug or incorrect output)
        if legal_probs.sum() == 0:
<<<<<<< HEAD
=======
            # print("Warning: All legal move probabilities are zero. Assigning uniform probabilities.")
>>>>>>> 40271a03c060f09b4879cfd10ada693f7ca34eb4
            legal_probs = np.ones_like(legal_probs)  # Assign uniform probability to all legal moves
        else:
            # Normalize probabilities to ensure they sum to 1
            legal_probs /= legal_probs.sum()  # Normalize to sum to 1

        # Ensure the probabilities sum to exactly 1, with a tolerance for floating-point errors
        if not np.isclose(legal_probs.sum(), 1.0):
            legal_probs /= legal_probs.sum()  # Re-normalize if necessary

        # Select an action index from the filtered probabilities
        selected_index = np.random.choice(list(legal_moves_map.keys()), p=legal_probs)

        # Return the selected legal move and its index
        return legal_moves_map[selected_index], selected_index

    def store_transition(self, prev_env, action_idx, reward, next_env, done):
        """Store transition for training."""
        self.transitions.append((prev_env, action_idx, reward, next_env, done))

    def compute_returns(self, rewards, dones):
        """Compute discounted returns."""
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

    def train(self):
        """Train the A2C agent using stored transitions."""
        if len(self.transitions) == 0:
            return

        # Extract transitions
        states, actions, rewards, next_states, dones = zip(*self.transitions)

        # Convert the list of states to a single numpy array first, then to a tensor
<<<<<<< HEAD
        states = torch.FloatTensor(np.array(states)).to(self.device)  # Move to GPU
        actions = torch.LongTensor(actions).to(self.device)  # Move to GPU
        returns = torch.FloatTensor(self.compute_returns(rewards, dones)).to(self.device)  # Move to GPU
=======
        states = torch.FloatTensor(np.array(states))  # Convert list of numpy arrays to numpy array before tensor
        actions = torch.LongTensor(actions)
        returns = torch.FloatTensor(self.compute_returns(rewards, dones))
>>>>>>> 40271a03c060f09b4879cfd10ada693f7ca34eb4

        # Reset transitions
        self.transitions = []

        # Compute advantage
        values = self.value_net(states).squeeze(1)  # Ensure values have the shape [batch_size]
        advantages = returns - values.detach()

        # Update policy network
        log_probs = torch.log(self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze())
        policy_loss = -(log_probs * advantages).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update value network
        value_loss = nn.MSELoss()(values, returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
