"""
MDP for Game
- agent: player
- actions:
    * move up/down
    * move left/right
Continuous state space:
    - player position: (x,y)
    - player motion vector: (x,y)
    - turret position: (x,y)
    - projectile position: (x,y)
    - projectile motion vector: (x,y)
    - obstacle position: (x,y)
    - gate position: (x,y)
- reward: 
    - 1 for each step that takes the player closer to the gate
    - 100 for reaching the gate 
    - -100 for colliding with a projectile
- discount: 0.99
- return: sum of discounted rewards

Bellamn equation:
Q(s,a) = R(s) + gamma * max(Q(s',a'))




"""
import math
import random
import numpy as np
from itertools import count

from game import Game, GameOptions
from gym.wrappers import FlattenObservation
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from collections import namedtuple, deque
import matplotlib.pyplot as plt

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QModel(nn.Module):
    def __init__(self, obs_dim, act_dim, output_shape):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, act_dim)
        self.output_shape = output_shape
        

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(self.output_shape) 


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state' ))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = [deque([],maxlen=capacity)]

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __iter__(self):
        return iter(self.memory)

    def __getitem__(self, idx):
        return self.memory[idx]

    def get_last(self, size):
        ret = []
        for _ in range(size):
            ret.append(self.memory.pop())
        return ret



# Training

MEM_SIZE = 1000
BATCH_SIZE = 100
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 4000 # The higher the longer it takes to decay
PARAM_UPDATE_RATE = 0.01

def epsilon_greedy_action(model_output, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    # Epsilon greedy
    if sample > eps_threshold:
        with torch.no_grad():
            return torch.max(model_output, dim=-1).indices
    # Exploration
    else:
        return torch.tensor(np.random.randint(0, 2, size=2), device=device, dtype=torch.int64).unsqueeze(dim=0)

options = GameOptions(  height=400, 
                        width=400, 
                        n_obstacles=0, 
                        n_turrets=1, 
                        max_projectiles_per_turret=2, 
                        fire_turret_step_delay=100
                        )
env = FlattenObservation(Game(render_mode="human", options=options, render_fps=120)) 

obs, _ = env.reset()
action_shape = (-1, len(env.action_space.nvec), np.max(env.action_space.nvec))
# q_model is the Q network
qmodel = QModel(obs.shape[0], np.sum(env.action_space.nvec), action_shape).to(device)
# q_new model helps us do soft updates
q_new = QModel(obs.shape[0], np.sum(env.action_space.nvec), action_shape).to(device)
print("action_shape", action_shape)
print(qmodel)

optimizer = optim.RMSprop(qmodel.parameters())
memory = ReplayMemory(500)

steps_done = 0
episode_durations = []
num_episodes = 100

print("Training for {} episodes".format(num_episodes), "with batch size {}".format(BATCH_SIZE))

for i_episode in range(num_episodes):
    state, info = env.reset()
    for t in count():
        input = torch.from_numpy(state).to(device, dtype=torch.float32).unsqueeze(dim=0)
        output = qmodel(input)
        action = epsilon_greedy_action(output, steps_done)
        obs, reward, done, _, info = env.step(action.squeeze(dim=0).numpy())
        reward = torch.tensor([reward], device=device, dtype=torch.float32)
        env.render()
        
        if not done:
            next_state = obs
        else:
            next_state = None
        memory.push(state, action, reward, next_state)
        
        state = next_state
        steps_done += 1

        if steps_done % MEM_SIZE == 0:
            # read last MINI_BATCH elements
            for _ in range(int(MEM_SIZE/BATCH_SIZE)):
                transitions = memory.get_last(BATCH_SIZE)
                #store current q model parameters
                q_new.load_state_dict(qmodel.state_dict())

                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Transition(*zip(*transitions))
                
                # Compute a mask of non-final states and concatenate the batch elements
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.stack([torch.from_numpy(s).to(device, dtype=torch.float32) for s in batch.next_state if s is not None])
                state_batch = torch.stack([torch.from_numpy(s).to(device, dtype=torch.float32) for s in batch.state])
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                # We are training a model with inputs x and outputs y
                # x is the state, s and the action a
                # y is the reward + gamma * max(Q(s',a'))
                # the prediction is Q(s,a)

                # Q(s,a)
                outputs = qmodel(state_batch)
                state_action_values = outputs.gather(2, action_batch.unsqueeze(dim=1)).squeeze(dim=1)
                # Diff sum between state_action_values and outputs.max(-1).values
                # print(torch.sum(state_action_values - outputs.max(-1).values))

                # max(Q(s',a')) only for non final states (for final states is 0)
                next_state_values = torch.zeros((BATCH_SIZE,action_shape[1]), device=device)
                next_state_values[non_final_mask, :] = qmodel(non_final_next_states).max(-1).values.detach()
                # reward + gamma * max(Q(s',a'))
                expected_state_action_values = (next_state_values * GAMMA) + reward_batch.unsqueeze(dim=-1)
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

                optimizer.zero_grad()
                loss.backward()
                for param in qmodel.parameters():
                    # Clip gradients to avoid exploding gradients
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

                #soft update of q_new
                for target_param, param in zip(q_new.parameters(), qmodel.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - PARAM_UPDATE_RATE) + param.data * PARAM_UPDATE_RATE)
                
                #load back to qmodel    
                qmodel.load_state_dict(q_new.state_dict())

            print("Step: {:7} Loss: {:10.4f}".format(steps_done, loss.item()), "episode: {:3}/{:3}".format(i_episode, num_episodes), end='\r', flush=True)

        if done:
            episode_durations.append(t + 1)
            print("Episode: {}/{} Duration: {}".format(i_episode, num_episodes, t + 1))
            break

#plot episode durations
plt.figure(2)
plt.clf()
plt.title('Training...')
plt.xlabel('Episode')
plt.ylabel('Duration')
plt.plot(episode_durations)
plt.show()
