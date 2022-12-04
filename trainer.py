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
    - 100 for colliding with a projectile
- discount: 0.99
- return: sum of discounted rewards

Bellamn equation:
Q(s,a) = R(s) + gamma * max(Q(s',a'))
"""
import math
import random
import numpy as np
from itertools import count

import torch
import torch.nn.functional as F

from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state' ))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([],maxlen=capacity)

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

    def pop(self, size):
        ret = []
        for _ in range(size):
            ret.append(self.memory.popleft())
        return ret

# Training

MEM_SIZE = 1000
BATCH_SIZE = 100
GAMMA = 0.95
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 4000 # The higher the longer it takes to decay
PARAM_UPDATE_RATE = 0.01


class Trainer():

    def __init__(self, target_model, aux_model, env, action_shape, optimizer, device, memory_size=MEM_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, param_update_rate=PARAM_UPDATE_RATE):
        self.target_model = target_model
        self.aux_model = aux_model
        self.optimizer = optimizer
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.env = env
        self.param_update_rate = param_update_rate
        self.action_shape = action_shape

    def epsilon_greedy_action(self, model_output, steps_done):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        # Epsilon greedy
        if sample > eps_threshold:
            with torch.no_grad():
                return torch.max(model_output, dim=-1).indices
        # Exploration
        else:
            return torch.tensor(np.random.randint(0, 2, size=2), device=self.device, dtype=torch.int64).unsqueeze(dim=0)

    def _optimize(self):
        # read last MINI_BATCH elements
        loss_array = []
        for _ in range(int(len(self.memory)/self.batch_size)):
            transitions = self.memory.pop(self.batch_size)
            #store current q model parameters
            self.aux_model.load_state_dict(self.target_model.state_dict())

            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))
            
            # Compute a mask of non-final states and concatenate the batch elements
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.stack([torch.from_numpy(s).to(self.device, dtype=torch.float32) for s in batch.next_state if s is not None])
            state_batch = torch.stack([torch.from_numpy(s).to(self.device, dtype=torch.float32) for s in batch.state])
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # We are training a model with inputs x and outputs y
            # x is the state, s and the action a
            # y is the reward + gamma * max(Q(s',a'))
            # the prediction is Q(s,a)

            # Q(s,a)
            outputs = self.target_model(state_batch)
            state_action_values = outputs.gather(2, action_batch.unsqueeze(dim=1)).squeeze(dim=1)
            # Diff sum between state_action_values and outputs.max(-1).values
            # print(torch.sum(state_action_values - outputs.max(-1).values))

            # max(Q(s',a')) only for non final states (for final states is 0)
            next_state_values = torch.zeros((self.batch_size,self.action_shape[1]), device=self.device)
            next_state_values[non_final_mask, :] = self.target_model(non_final_next_states).max(-1).values.detach()
            # reward + gamma * max(Q(s',a'))
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch.unsqueeze(dim=-1)
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.target_model.parameters():
                # Clip gradients to avoid exploding gradients
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            #soft update of aux model
            for target_param, param in zip(self.aux_model.parameters(), self.target_model.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.param_update_rate) + param.data * self.param_update_rate)
            
            #load back to target model    
            self.target_model.load_state_dict(self.aux_model.state_dict())

            loss_array.append(loss.item())

        return loss_array

    def do_train(self, num_episodes, max_step_per_episode=10000, target_max_steps=100, target_success_window=5):

        print("Training for {} episodes".format(num_episodes), "with batch size {}".format(self.batch_size))
        
        episode_durations = []
        last_rewards = []
        steps_done = 0
        success = False

        for i_episode in range(num_episodes):
            state, info = self.env.reset()
            for t in count():
                input = torch.from_numpy(state).to(self.device, dtype=torch.float32).unsqueeze(dim=0)
                output = self.target_model(input)
                action = self.epsilon_greedy_action(output, steps_done)
                obs, reward, done, _, info = self.env.step(action.squeeze(dim=0).numpy())
                reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
                self.env.render()
                
                if not done:
                    next_state = obs
                else:
                    next_state = None
                self.memory.push(state, action, reward, next_state)
                
                state = next_state
                steps_done += 1

                if steps_done % self.memory.capacity == 0:
                    loss = self._optimize()
                    print("Step: {:7} Loss: {:4.4}".format(steps_done, np.asarray(loss).mean()) , " - episode: {:3}/{:3}".format(i_episode, num_episodes), end='\r', flush=True)

                if done:
                    loss = self._optimize()
                    episode_durations.append(t + 1)
                    last_rewards.append(reward.item())
                    print("Episode: {}/{} Duration: {}. Final reward: {}".format(i_episode, num_episodes, t + 1, reward.item()))
                    break
                
                # Early stopping if not converging in this episode
                if t >= max_step_per_episode:
                    print("Max steps reached, stopping at episode {}".format(i_episode))
                    return episode_durations, last_rewards, success
            
            # If the last 5 episode durations are less than target_max_steps and the last 5 rewards are greater than 900, we consider the task solved
            if len(episode_durations) > target_success_window and len(last_rewards) > target_success_window:
                if np.asarray(episode_durations[-target_success_window:]).mean() < target_max_steps and np.asarray(last_rewards[-target_success_window:]).mean() > 900:
                    print("Task solved in {} episodes".format(i_episode))
                    success = True
                    return episode_durations, last_rewards, success
        
        return episode_durations, last_rewards, success

