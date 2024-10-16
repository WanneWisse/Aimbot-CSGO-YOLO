import torch
import torch.nn as nn
import torch.optim as optim
from policy_gradient_network import PolicyNetwork



def env(action):
    return -action**2

def reinforce_one_step(policy_net, lr=1e-3):
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    state = 10
    for episode in range(10000):  # Training loop
        # state = torch.FloatTensor(state)
        # print(state)
        state = torch.tensor([state], dtype=torch.float32)
        # Sample actions from policy
        print(state)
        action_mean, action_std = policy_net.forward(state)
        action_distribution = torch.distributions.Normal(action_mean, action_std)
        action = action_distribution.sample()
        
        # Evaluate rewards for sampled actions
        reward = env(action.item())
        reward = torch.tensor([reward], dtype=torch.float32)
        
        # Compute loss: negative log likelihood weighted by rewards
        log_prob = action_distribution.log_prob(action)
        loss = -(log_prob * reward)
        print(loss)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

policy_net = PolicyNetwork(1,1)
reinforce_one_step(policy_net)




