from mujoco_env import ClawbotCan
import numpy as np
import random

import torch
from torch import nn
import torch.optim as optim

# Check CUDA availability and set device
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and the correct PyTorch installation.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class QNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128, all_actions=None): 
        # obs = Observation, action = Action
        super(QNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim+action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3) # lr=0.001
        self.all_actions = all_actions.to(device) if all_actions is not None else None

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Move the entire model to GPU
        self.to(device)
    
    def forward(self, obs, action):
        # Tiling the observation
        if random.random() < 0.2: # 20% random action
            return torch.tensor(random.randint(0, self.action_dim-1), dtype=torch.int64, device=device)
        else:
            with torch.no_grad():
                obs = torch.tile(obs, [self.action_dim, 1])
                obs_action = torch.cat([obs, self.all_actions], dim=1)
                q_values = self.net(obs_action)

                max_action = torch.argmax(q_values)
            return max_action
    
    def forward(self, obs):
        # Tiling the observation
        if random.random() < 0.2: # 20% random action
            return torch.tensor(random.randint(0, self.action_dim-1), dtype=torch.int64, device=device)
        else:
            with torch.no_grad():
                obs = torch.tile(obs, [self.all_actions.shape[0], 1])
                obs_action = torch.cat([obs, self.all_actions], dim=1)
                q_values = self.net(obs_action)

                max_action = torch.argmax(q_values)
            return max_action
    
    def train(self, dataset, sample_size=50):
        if len(dataset) < sample_size: return 0.0
        
        # sample a bunch of data points (s, a, s', r, terminal_state)
        with torch.no_grad():
            minibatch = random.sample(dataset, k=sample_size)

            state = torch.tensor([item[0] for item in minibatch], dtype=torch.float32, device=device)
            action = torch.tensor([item[1] for item in minibatch], dtype=torch.float32, device=device)
            next_state = torch.tensor([item[2] for item in minibatch], dtype=torch.float32, device=device)
            reward = torch.tensor([item[3] for item in minibatch], dtype=torch.float32, device=device)
            terminal_state = torch.tensor([item[4] for item in minibatch], dtype=torch.float32, device=device)

            next_state = torch.tile(next_state, [1, self.all_actions.shape[0]]).view(-1, self.obs_dim)
            next_action = torch.tile(self.all_actions, [sample_size, 1])
            next_q_state = torch.cat([next_state, next_action], dim=1)
            next_q_values = self.net(next_q_state).view(sample_size, self.all_actions.shape[0])

            target_q_values = reward + 0.9 * torch.max(next_q_values, dim=1)[0].flatten() * (1 - terminal_state)

        self.optimizer.zero_grad()
        pred_q_values = self.net(torch.cat([state, action], dim=1)).flatten()
        # loss = F.mse_loss(pred_q_values, target_q_values) 
        error = nn.MSELoss()(pred_q_values, target_q_values)
        error.backward()
        self.optimizer.step()
        return error



all_actions = torch.tensor([
    [0, 0, 0, 0], # <---- Stay still
    [-1, 1, 0, 0], # <---- Spin left
    [1, -1, 0, 0], # <---- Spin right
    [1, 1, 0, 0], # <---- Move forward
    [-1, -1, 0, 0] # <---- Move backward
], dtype=torch.float32, device=device)

model = QNet(
    obs_dim=3,  # <---- distance, angle, grab done (termination state)
    action_dim=4,
    all_actions=all_actions
)

from collections import deque
dataset = deque(maxlen=10000) # [obs, action, new_obs, reward, terminal


if __name__ == "__main__":
    env = ClawbotCan()
    # obs = env.reset()
    # while True:
    #     # left, right, arm, claw
    #     # action = [0, 0, 0, 0] # <---- this is the default action
    #     # action = [-1, 1, 0, 0] # <---- this is the action that spins the robot
    #     # action = [1, -1, 0, 0] # <---- this is the action that spins the robot
    #     # action = [1, 1, 0, 0] # <---- this is the action that moves the robot forward
    #     action = [1, 1, 0, 0] # <---- this is the action that moves the robot backward
    #     # env.step(action)
    #     # obs, reward, terminated, truncated, info = env.step(np.random.uniform(-1, 1, [4]))
    #     obs, reward, terminated, info = env.step(np.random.uniform(-1, 1, [4]))
    #     env.render()

    for episode in range(10000):
        obs = env.reset()

        for _ in range(200):
            #  action = env.action_space.sample()
            action_idx = model.forward(torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)) # model(obs)
            action_idx = action_idx.cpu().numpy().item() # .squeeze()
            # action = model.forward(obs, env.action_space.sample())
            action = all_actions[action_idx].cpu().numpy()
            new_obs, reward, terminated, info = env.step(action)

            dataset.append((obs, model.all_actions[action_idx].cpu().numpy(), new_obs, reward, int(terminated)))
            obs = new_obs

            model.train(dataset)
            if terminated:
                break
            env.render()

        print(f"Episode {episode} finished with reward {reward}")

    # env.close()