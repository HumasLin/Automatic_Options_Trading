import numpy as np
import random
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# configure a model for deep Q-learning networks
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(8, action_size),
        )
    def forward(self, input):
        return self.main(input)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
# class that operates the history of the simulated trading actions
class History_operations(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.history = []
        self.position = 0
    # Update history with latest results
    def push(self, *args):
        if len(self.history) < self.capacity:
            self.history.append(None)
        self.history[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    # Take out samples from action history
    def sampling(self, batch_size):
        return random.sample(self.history, batch_size)
    def __len__(self):
        return len(self.history)

# class that trains a model for trading
class Trader:
    def __init__(self, state_size, is_eval=False):
        self.state_size = state_size  
        self.action_size = 3 
        self.history = History_operations(10000)
        self.portfolio = []
        self.is_eval = is_eval
        # parameters that determine the training preferences
        self.gamma = 0.95
        self.epsilon = 0.8
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        # read existing models if there exist two models
        if os.path.exists('target_model'):
            self.predict_net = torch.load('predict_model', map_location=device)
            self.target_net = torch.load('target_model', map_location=device)
        else:
            self.predict_net = DQN(state_size, self.action_size)
            self.target_net = DQN(state_size, self.action_size)
        # optimize predict network
        self.optimizer = optim.RMSprop(self.predict_net.parameters(), lr=0.005, momentum=0.9)

    # use predicted results to decide which action to take
    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        tensor = torch.FloatTensor(state).to(device)
        options = self.target_net(tensor)
        return np.argmax(options[0].detach().numpy())

    # optimize the predict modeli
    def optimize(self):
        if len(self.history) < self.batch_size:
            return
        transitions = self.history.sampling(self.batch_size)
        # transpose the batch model and then create new states to update predict model
        batch = Transition(*zip(*transitions))
        next_state = torch.FloatTensor(batch.next_state).to(device)
        temp_states = torch.tensor(tuple(map(lambda s: s is not None, next_state)))
        temp_states_next = torch.cat([s for s in next_state if s is not None])
        states_batch = torch.FloatTensor(batch.state).to(device)
        actions_batch = torch.LongTensor(batch.action).to(device)
        rewards_batch = torch.FloatTensor(batch.reward).to(device)

        # the model computes Q value at first, then we best actions with best predicted outcome
        state_action = self.predict_net(states_batch).reshape((self.batch_size, 3)).gather(1, actions_batch.reshape(
            (self.batch_size, 1)))
        # compute the expected outcome pf action from target model and select the best reward
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[temp_states] = self.target_net(temp_states_next).max(1)[0].detach()
        # compute expected Q values
        state_action_expected = (next_state_values * self.gamma) + rewards_batch

        # get Huber loss
        loss = F.smooth_l1_loss(state_action, state_action_expected.unsqueeze(1))
        # optimize our model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.predict_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
