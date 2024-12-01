import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Actor-Critic networks
        self.policy_net = self.build_policy_network(state_dim, action_dim)
        self.value_net = self.build_value_network(state_dim)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        # Transition storage
        self.transitions = []

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

    def select_action(self, state):
        """Select an action based on the policy network."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state_tensor).detach().numpy().squeeze()
        action = np.random.choice(len(probs), p=probs)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition for training."""
        self.transitions.append((state, action, reward, next_state, done))

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
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        returns = torch.FloatTensor(self.compute_returns(rewards, dones))

        # Reset transitions
        self.transitions = []

        # Compute advantage
        values = self.value_net(states).squeeze()
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
