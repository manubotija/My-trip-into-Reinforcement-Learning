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
import torch.nn as nn
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

class HyperParams():
    def __init__(self,
                memory_size=MEM_SIZE, 
                batch_size=BATCH_SIZE, 
                gamma=GAMMA, 
                param_update_rate=PARAM_UPDATE_RATE,
                eps_start=EPS_START,
                eps_end=EPS_END,
                eps_decay=EPS_DECAY) -> None:
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.param_update_rate = param_update_rate
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def get_params(self):
        return {
            "memory_size": self.memory_size,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "param_update_rate": self.param_update_rate,
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
            "eps_decay": self.eps_decay
        }
    def __str__(self):
        return str(self.get_params())

class Trainer():

    def __init__(self, 
                target_model, 
                aux_model, 
                env, 
                action_shape, 
                optimizer, 
                device, 
                hyperparams: HyperParams = None):
        self.target_model = target_model
        self.aux_model = aux_model
        self.optimizer = optimizer
        self.device = device
        self.env = env
        self.action_shape = action_shape
        self.memory = ReplayMemory(hyperparams.memory_size)
        if hyperparams is None:
            self.hyperparams = HyperParams()
        else:
            self.hyperparams = hyperparams
        
    # returns a dictionary of hyperparameters
    def get_params(self):
        return self.hyperparams.get_params()

    def _epsilon_greedy_action(self, model_output, steps_done):
        sample = random.random()
        eps_threshold = self.hyperparams.eps_end + (self.hyperparams.eps_start - self.hyperparams.eps_end) * \
            math.exp(-1. * steps_done / self.hyperparams.eps_decay)
        # Epsilon greedy
        if sample > eps_threshold:
            with torch.no_grad():
                return torch.max(model_output, dim=-1).indices
        # Exploration
        else:
            return torch.tensor(np.random.randint(0, 2, size=2), device=self.device, dtype=torch.int64).unsqueeze(dim=0)

    def _optimize(self):

        loss_array = []
        for _ in range(int(len(self.memory)/self.hyperparams.batch_size)):
            transitions = self.memory.pop(self.hyperparams.batch_size)
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

            """
             We are training a model with inputs x and outputs y
             x is the state, s and the action a
             y is the reward + gamma * max(Q(s',a'))
             the prediction is Q(s,a)
            """
            # Q(s,a) using actions taken with epsilon greedy
            outputs = self.target_model(state_batch)
            state_action_values = outputs.gather(2, action_batch.unsqueeze(dim=1)).squeeze(dim=1)

            # max(Q(s',a')) only for non final states (for final states is 0)
            next_state_values = torch.zeros((self.hyperparams.batch_size, self.action_shape[1]), device=self.device)
            with torch.no_grad(): 
                next_state_values[non_final_mask, :] = self.target_model(non_final_next_states).max(-1).values
            # reward + gamma * max(Q(s',a'))
            expected_state_action_values = (next_state_values * self.hyperparams.gamma) + reward_batch.unsqueeze(dim=-1)
            
            # Compute the loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values)

            self.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.target_model.parameters(), 100)
            self.optimizer.step()

            #soft update of aux model
            aux_model_state_dict = self.aux_model.state_dict()
            target_model_state_dict = self.target_model.state_dict()
            for key in target_model_state_dict:
                target_model_state_dict[key] = target_model_state_dict[key]*self.hyperparams.param_update_rate + aux_model_state_dict[key]*(1-self.hyperparams.param_update_rate)
            self.target_model.load_state_dict(target_model_state_dict)
            
            # for target_param, param in zip(self.aux_model.parameters(), self.target_model.parameters()):
            #     target_param.data.copy_(target_param.data * (1.0 - self.hyperparams.param_update_rate) + param.data * self.hyperparams.param_update_rate)  
            # self.target_model.load_state_dict(self.aux_model.state_dict())

            loss_array.append(loss.item())

        return loss_array

    def do_train(self, num_episodes, target_max_steps=100, target_success_window=5):

        print("## Training for {} episodes".format(num_episodes), "with batch size {} ##".format(self.hyperparams.batch_size))
        
        episode_durations = []
        last_rewards = []
        loss_per_batch = []
        score_per_episode = []
        steps_done = 0
        success = False
        positive_final_rewards = 0
        negative_final_rewards = 0

        for i_episode in range(num_episodes):
            state, info = self.env.reset()
            for t in count():
                input = torch.from_numpy(state).to(self.device, dtype=torch.float32).unsqueeze(dim=0)
                output = self.target_model(input)
                action = self._epsilon_greedy_action(output, steps_done)
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
                    loss_per_batch += loss

                # loss = self._optimize()
                # loss_per_batch += loss

                if done:
                    # loss = self._optimize()
                    # loss_per_batch += loss
                    episode_durations.append(t + 1)
                    last_rewards.append(reward.item())
                    
                    if reward.item() > 0:
                        positive_final_rewards += 1
                    else:
                        negative_final_rewards += 1

                    score = positive_final_rewards/(positive_final_rewards+negative_final_rewards)

                    self._print_status(i_episode, 
                                        num_episodes, 
                                        steps_done, 
                                        np.asanyarray(loss_per_batch).mean(), 
                                        score)
                    break
                
                # Early stopping if not converging in this episode
                # if t >= max_step_per_episode:
                #     print("## Max steps reached, stopping at episode {} ##".format(i_episode))
                #     return episode_durations, last_rewards, success, loss_per_batch
            
            # success condition
            if len(episode_durations) > target_success_window and len(last_rewards) > target_success_window:
                if np.asarray(episode_durations[-target_success_window:]).mean() < target_max_steps and np.asarray(last_rewards[-target_success_window:]).mean() > 900:
                    print("## Task solved in {} episodes ##".format(i_episode))
                    success = True
                    return episode_durations, last_rewards, success, loss_per_batch, score
        
        self._print_status(i_episode, 
                            num_episodes, 
                            steps_done, 
                            np.asanyarray(loss_per_batch).mean(), 
                            score,
                            end="\n")
        return episode_durations, last_rewards, success, loss_per_batch, score

    def _print_status(self, episode, num_episodes, steps_done, loss, score, end='\r'):
        print("Step: {:7} Loss: {:4.4}".format(steps_done, loss) , 
            " - episode: {:3}/{:3}".format(episode, num_episodes),
            " - score: {:4.4}".format(score),
             end=end, flush=True)