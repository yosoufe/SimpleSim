# -*- coding: utf-8 -*-

# Import and create an environment, such as Pendulum
import gymnasium as gym
from mujoco_env import ClawbotCan
#env = gym.make('Pendulum-v1', render_mode='human')
env = ClawbotCan()

# Import some libraries that we will use in this example to do RL
import torch
from torch import nn
import random
from collections import deque

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and the correct PyTorch installation.")


class DQN(nn.Module):
  """
  The DQN Module takes in the size of the state space as well as the number of actions.
  For example - if the action space was [-1 -1] [-1 0] [-1 1] [0 -1] [0 0] [0 1] [1 -1] [1 0] [1 1]
  then act_dim = 9.
  """
  def __init__(self, obs_dim, act_dim, all_actions):
    super().__init__()
    self.obs_dim = obs_dim
    self.act_dim = act_dim

    # Create one-hot vectors for each action
    self.all_actions = torch.eye(act_dim)

    # Create a basic neural network and optimizer which takes in an observation and action one-hot
    # then outputs a value representing the Q-value of doing that action on this state
    self.net = nn.Sequential(
      nn.Linear(obs_dim + act_dim, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 1)
    )
    self.optim = torch.optim.Adam(self.parameters(), lr=0.001)
    self.all_actions = all_actions

  def forward(self, obs: torch.Tensor):
    """
    Return the index of the action to take.
    Can either be the max-qvalue action on this state or a random action.
    """
    if random.random() < 0.2: # 20% random action
      return torch.tensor(random.randint(0, self.all_actions.shape[0]-1), dtype=torch.int64)
    else: # 80% "optimal" action
      # Prevent gradients from being generated for backprop
      with torch.no_grad():
        # Repeat the state |A| times in order to evaluate all of them at the same time
        obs = torch.tile(obs, [self.all_actions.shape[0], 1])
        # Combine observations and actions to send into the neural network
        obs_act = torch.cat([obs, self.all_actions], dim=1)
        # Grab Q-values for all (s, a) pairs
        q_values = self.net(obs_act)
        # The action which gives the highest Q-value is our best action, return the idx of that action
        max_action = torch.argmax(q_values)
      return max_action
    
  def train(self, dataset, num_samples=10):
    if len(dataset) < num_samples: return 0.0
    # Sample a bunch of datapoints (s, a, s', r, term)
    # Note that the neural network is typed float32 by default, so we have to convert the dtype
    with torch.no_grad():
      minibatch = random.sample(dataset, k=num_samples)
      state = torch.tensor([item[0] for item in minibatch], dtype=torch.float32)
      action = torch.tensor([item[1].numpy() for item in minibatch], dtype=torch.float32)
      next_state = torch.tensor([item[2] for item in minibatch], dtype=torch.float32)
      reward = torch.tensor([item[3] for item in minibatch], dtype=torch.float32)
      term = torch.tensor([item[4] for item in minibatch], dtype=torch.float32)

      # Calculate what the Q-value "should" be using the Bellman Equation and use that as our label
      # Bellman Equation: Q(s, a) = ∑{ {R(s,a,s') + γ max(Q(s', a') for a' in A)*(1-T)} * P(s'|s,a) for s' in S }
      # Note that if we look at the dataset's probability distribution, it will look like P(s'|s,a)
      # So we really only need to calculate R(s,a,s') + γ max(Q(s', a') for a' in A)*(1-T)
      next_state = torch.tile(next_state, [1, self.all_actions.shape[0]]).view(-1, self.obs_dim)  # [s1' s1' s1' ... s2' s2' s2' ... sN' sN' sN' ...]
      next_action = torch.tile(self.all_actions, [num_samples, 1]) # [a1' a2' a3' ... a1' a2' a3' ... a1' a2' a3' ...] for a' in A
      next_q_state = torch.cat([next_state, next_action], dim=1) # [s1'a1' s1'a2' s1'a3' ... sN'a1' sN'a2' sN'a3' ...]
      next_q_value = self.net(next_q_state).view(num_samples, self.all_actions.shape[0]) # calculate the q-values for all s'a' (B, |S|*|A|)
      target_q_value = reward + 0.9 * torch.max(next_q_value, dim=1)[0].flatten() * (1 - term) # get the max q-value and compute Bellman

    # Use the output from the Bellman equation to learn the actual q-value
    self.optim.zero_grad()
    q_state = torch.cat([state, action], dim=1).to(torch.float32)
    pred_q_value = self.net(q_state).flatten()
    error = nn.MSELoss()(pred_q_value, target_q_value)
    error.backward()
    self.optim.step()

    return error

# Define torques that the pendulum can swing
#actions = [[-2], [-1], [0], [1], [2]]

# Create model and dataset storage
allactions = [[0, 0,0, 0], [-1, 1, 0, 0], [1, -1, 0, 0]]
allactions = torch.tensor(allactions, dtype=torch.float32)
model = DQN(3, 4, allactions)
dataset = deque(maxlen=10000) # (obs, act, new_obs, rew, terminal)

# Train for 100 episodes
for episode in range(100):
  obs = env.reset()
  total_reward = 0

  # For each epoch, try 200 steps before ending the episode and resetting pendulum position
  for _ in range(200):
    # Sample an action
    action = model.forward(torch.tensor(obs, dtype=torch.float32).view(1, -1))
    action_idx = action.cpu().numpy()
    action = allactions[action_idx]

    # Try the action out
    new_obs, rew, term, info = env.step(action)
    env.render()

    # Store the result in the dataset and redefine the current observation
    dataset.append([obs, model.all_actions[action_idx], new_obs, rew, int(term)])
    obs = new_obs
    total_reward += rew

    # Train our DQN
    model.train(dataset)
    if term:
      print('Total Reward:', total_reward)
      break
#env.close()