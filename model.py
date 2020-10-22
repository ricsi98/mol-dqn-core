import numpy as np
import random

from collections import OrderedDict

import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import torch

class VariableActionReplayBuffer():

    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []

    def remember(self, state, action, possible_actions, next_possible_actions, state_, reward, done):
        self.memory.append((state, action, possible_actions, next_possible_actions, state_, reward, done))
        if len(self.memory) > self.max_size:
            self.memory = self.memory[-self.max_size:]

    def sample(self, N):
        return random.sample(self.memory, min(len(self.memory), N))


class ReplayBuffer():

    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []

    def remember(self, state, action, state_, reward, done):
        self.memory.append((state, action, state_, reward, done))
        if len(self.memory) > self.max_size:
            self.memory = self.memory[-self.max_size:]

    def sample(self, N):
        return random.sample(self.memory, min(len(self.memory), N))

class QNetwork(nn.Module):

    def __init__(self, input_dims, output_dims, fc1=128, fc2=128):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, output_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class QNetwork2(nn.Module):

    def __init__(self, state_input_dims, action_input_dims, fc1=128, fc2=128):
        super().__init__()
        self.fc1_state = nn.Linear(state_input_dims, fc1)
        self.fc1_action = nn.Linear(action_input_dims, fc1)
        self.fc2 = nn.Linear(2*fc1, fc2)
        self.fc3 = nn.Linear(fc2, 1)

    def forward(self, state, action):
        state_x = F.relu(self.fc1_state(state))
        action_x = F.relu(self.fc1_action(action))
        conc = torch.cat((state_x, action_x), 0)
        x = F.relu(self.fc2(conc))
        x = F.relu(self.fc3(x))
        return x

class BootstrappedQNetwork(nn.Module):

    def __init__(self, state_input_dims, action_input_dims, layers, head_cnt):
        assert layers[0] % 2 == 0, "Dimension of first layer must be even!"
        super().__init__()
        self.fc_state = nn.Linear(state_input_dims, layers[0] // 2)
        self.fc_action = nn.Linear(action_input_dims, layers[0] // 2)

        self.head_cnt = head_cnt
        self.heads = []

        for h in range(head_cnt):
            self.heads.append(self._init_head(layers, h))

        self.selected_head = random.choice(self.heads)

    def _init_head(self, layers_, head):
        layers = []
        names = []

        for i in range(1, len(layers_)):
            l_prev = layers_[i-1]
            l_curr = layers_[i]
            layers.append(nn.Linear(l_prev, l_curr))
            layers.append(nn.ReLU())
            names.append('head-%d-layer-%d' % (head, i))
            names.append('head-%d-activation-%d' % (head, i))

        h = nn.Sequential(OrderedDict(list(zip(names, layers))))
        self.add_module('head-%d' % head, h)
        return h

    def use_random_head(self):
        self.selected_head = random.choice(self.heads)
        return self.heads.index(self.selected_head)

    def use_head(self, idx):
        self.selected_head = self.heads[idx]

    def forward(self, state, action):
        state_x = F.relu(self.fc_state(state))
        action_x = F.relu(self.fc_action(action))
        conc = torch.cat((state_x, action_x), 0)
        return self.selected_head.forward(conc)

class BootstrappedDQN():

    def __init__(self, input_size, action_size, n_heads=10, lr=0.0001, 
                 memory_size=2000, gamma=0.99, tau=1e-5, eps_start=1.0, eps_decay=0.99):
        self.memory = VariableActionReplayBuffer(memory_size)
        self.n_heads = n_heads
        layers = [128, 128, 1]
        self.q_online = BootstrappedQNetwork(input_size, action_size, layers, n_heads)
        self.q_target = BootstrappedQNetwork(input_size, action_size, layers, n_heads)

        self.action_space = [i for i in range(action_size)]

        self.gamma = gamma
        self.tau = tau
        self.eps  = eps_start
        self.eps_decay = eps_decay

        self.optimizer = Adam(self.q_online.parameters(), lr=lr)
        self.mse = nn.MSELoss()

    def remember(self, transition):
        self.memory.remember(*transition)

    def update_target(self):
        self.q_target.load_state_dict(self.q_online.state_dict())

    def select_action(self, state, actions):
        self.q_online.use_random_head()
        if np.random.uniform() > self.eps:
            state = torch.tensor(state, dtype=torch.float)
            actions = [torch.tensor(action, dtype=torch.float) for action in actions]
            q_values = torch.tensor([self.q_online.forward(state, action) for action in actions])
            action = torch.argmax(q_values).item()
        else:
            action = random.choice(self.action_space)

        return action

    def learn(self, batch_size):

        for state, action, possible_actions, next_possible_actions, state_, reward, done in self.memory.sample(batch_size):
            online_head = self.q_online.use_random_head()
            self.q_target.use_head(online_head)

            state = torch.tensor(state, dtype=torch.float)
            possible_actions = [torch.tensor(a, dtype=torch.float) for a in possible_actions]
            next_possible_actions = [torch.tensor(a, dtype=torch.float) for a in next_possible_actions]
            state_ = torch.tensor(state_, dtype=torch.float)
            done = 1 if done else 0

            current_q = self.q_online.forward(state, possible_actions[action])
            
            with torch.no_grad():
                next_q = torch.max(torch.tensor([self.q_target.forward(state_, a_).item() for a_ in next_possible_actions]))
                td = reward + self.gamma * next_q * (1-done)

            loss = self.mse(current_q[0], td)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.update_target()
        self.eps *= self.eps_decay

class VariableActionDQN():

    def __init__(self, input_size, action_size, lr=0.0001, memory_size=2000,
                 gamma=0.99, tau=1e-5, eps_start=1.0, eps_decay=0.99):
        self.memory = VariableActionReplayBuffer(memory_size)
        self.q_online = QNetwork2(input_size, action_size, 1)
        self.q_target = QNetwork2(input_size, action_size, 1)

        self.action_space = [i for i in range(action_size)]

        self.gamma = gamma
        self.tau = tau
        self.eps  = eps_start
        self.eps_decay = eps_decay

        self.optimizer = Adam(self.q_online.parameters(), lr=lr)
        self.mse = nn.MSELoss()

    def remember(self, transition):
        self.memory.remember(*transition)

    def update_target(self):
        self.q_target.load_state_dict(self.q_online.state_dict())

    def select_action(self, state, actions):
        if np.random.uniform() > self.eps:
            state = torch.tensor(state, dtype=torch.float)
            actions = [torch.tensor(action, dtype=torch.float) for action in actions]
            q_values = torch.tensor([self.q_online.forward(state, action) for action in actions])
            action = torch.argmax(q_values).item()
        else:
            action = random.choice(self.action_space)

        return action

    def learn(self, batch_size):

        for state, action, possible_actions, next_possible_actions, state_, reward, done in self.memory.sample(batch_size):
            state = torch.tensor(state, dtype=torch.float)
            possible_actions = [torch.tensor(a, dtype=torch.float) for a in possible_actions]
            next_possible_actions = [torch.tensor(a, dtype=torch.float) for a in next_possible_actions]
            state_ = torch.tensor(state_, dtype=torch.float)
            done = 1 if done else 0

            current_q = self.q_online.forward(state, possible_actions[action])
            
            with torch.no_grad():
                next_q = torch.max(torch.tensor([self.q_target.forward(state_, a_).item() for a_ in next_possible_actions]))
                td = reward + self.gamma * next_q * (1-done)

            loss = self.mse(current_q[0], td)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.update_target()
        self.eps *= self.eps_decay

class DQN():

    def __init__(self, input_size, action_size, lr=0.0001, memory_size=2000,
                 gamma=0.99, tau=1e-5, eps_start=1.0, eps_decay=0.99):
        self.memory = ReplayBuffer(memory_size)
        self.q_online = QNetwork(input_size, action_size)
        self.q_target = QNetwork(input_size, action_size)

        self.action_space = [i for i in range(action_size)]

        self.gamma = gamma
        self.tau = tau
        self.eps  = eps_start
        self.eps_decay = eps_decay

        self.optimizer = Adam(self.q_online.parameters(), lr=lr)
        self.mse = nn.MSELoss()

    def remember(self, transition):
        self.memory.remember(*transition)

    def update_target(self):
        self.q_target.load_state_dict(self.q_online.state_dict())

    def select_action(self, state):
        if np.random.uniform() > self.eps:
            state = torch.tensor(state, dtype=torch.float)
            q_values = self.q_online.forward(state)
            action = torch.argmax(q_values).item()
        else:
            action = random.choice(self.action_space)

        return action

    def learn(self, batch_size):
        dones = 0
        for state, action, state_, reward, done in self.memory.sample(batch_size):
            state = torch.tensor(state, dtype=torch.float)
            state_ = torch.tensor(state_, dtype=torch.float)
            done = 1 if done else 0

            current_q = self.q_online.forward(state)[action]
            with torch.no_grad():
                next_q = self.q_target.forward(state_).detach().max()
                td = reward + self.gamma * next_q * (1-done)

            loss = self.mse(current_q, td)

            #print(current_q.item(), next_q.item(), reward, done, loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        #print('DONE RATIO', float(dones) / float(batch_size))
        self.update_target()
        self.eps *= self.eps_decay
