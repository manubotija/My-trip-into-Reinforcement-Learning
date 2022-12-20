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
import sys
import math
import random
import numpy as np
from itertools import count
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state' ))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    # def pop(self, size):
    #     ret = []
    #     for _ in range(size):
    #         ret.append(self.memory.popleft())
    #     return ret

# Training

MEM_SIZE = 1000
BATCH_SIZE = 100
GAMMA = 0.95
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 4000 # The higher the longer it takes to decay
PARAM_UPDATE_RATE = 0.01
LR = 0.001

class HyperParams():
    def __init__(self,
                memory_size=MEM_SIZE, 
                batch_size=BATCH_SIZE, 
                gamma=GAMMA, 
                tau=PARAM_UPDATE_RATE,
                eps_start=EPS_START,
                eps_end=EPS_END,
                eps_decay=EPS_DECAY,
                lr=LR) -> None:
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.lr = lr

    def get_params(self):
        return {
            "memory_size": self.memory_size,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
            "eps_decay": self.eps_decay,
            "lr": self.lr
        }
    def __str__(self):
        return str(self.get_params())

class Agent():

    def __init__(self, 
                policy_model, 
                target_model, 
                env,
                action_shape, 
                optimizer, 
                device, 
                hyperparams: HyperParams = None,
                eval_env=None,
                final_plot=True,
                live_plot=False):
        self.policy_model = policy_model
        self.target_model = target_model
        self.optimizer = optimizer
        self.device = device
        self.env = env
        self.eval_env = eval_env
        self.action_shape = action_shape
        self.memory = ReplayMemory(hyperparams.memory_size)
        self.final_plot = final_plot
        self.live_plot = live_plot
        if hyperparams is None:
            self.hyperparams = HyperParams()
        else:
            self.hyperparams = hyperparams
        if self.live_plot:
            self._init_plot()
    
    def _init_plot(self):
        self.fig, (self.loss_axis, self.duration_axis, self.score_axis) = plt.subplots(1, 3)

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
                return torch.max(model_output.view(self.action_shape), dim=-1).indices
        # Exploration
        else:
            return torch.tensor(np.asarray([self.env.action_space.sample()]), device=self.device, dtype=torch.int64)

    def _replay(self, off_policy=True):

        if len(self.memory) < self.hyperparams.batch_size:
            return
        
        if off_policy:
            # Sample random transitions from memory
            transitions = self.memory.sample(self.hyperparams.batch_size)
        else:
            # Get last transitions from memory
            transitions = self.memory.pop(self.hyperparams.batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s.squeeze() for s in batch.next_state if s is not None])
        state_batch = torch.stack([s.squeeze() for s in batch.state])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        """
            We are training a model that given input x, produces prediction y' 
            x is the state, s and the action a
            y' (the prediction) is Qp(s,a)
            the ground truth, y, is the reward + gamma * max(Qt(s',a'))
        """
        # Q(s,a) using actions taken with epsilon greedy
        outputs = self.policy_model(state_batch)
        state_action_values = outputs.gather(1, action_batch.unsqueeze(dim=1)).squeeze(dim=1)

        # max(Q(s',a')) only for non final states (for final states is 0)
        # reward + gamma * max(Q(s',a'))
        next_state_values = torch.zeros((self.hyperparams.batch_size), device=self.device)
        with torch.no_grad(): 
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.hyperparams.gamma) + reward_batch
        
        # Compute the loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)
        self.optimizer.step()

        return loss

    def _soft_model_update(self):
        #soft update of target model
        target_model_state_dict = self.target_model.state_dict()
        policy_model_state_dict = self.policy_model.state_dict()
        for key in policy_model_state_dict:
            target_model_state_dict[key] = policy_model_state_dict[key]*self.hyperparams.tau + target_model_state_dict[key]*(1-self.hyperparams.tau)
        self.target_model.load_state_dict(target_model_state_dict)
    
    def _evaluate_target_model(self, num_episodes=10):
        if self.eval_env is None:
            return 0.0
        # Evaluate the target model
        average_score = 0
        for i_episode in range(num_episodes):
            state, info = self.eval_env.reset()
            episode_score = 0
            for t in count():
                with torch.no_grad(): 
                    output = self.target_model(state)
                    action = torch.max(output.view(self.action_shape), dim=-1).indices
                obs, reward, done, _, info = self.eval_env.step(action)
                episode_score += reward.item()
                #self.eval_env.render()
                if done:
                    break
                state = obs
            average_score += episode_score
        return average_score/num_episodes

    # method that performs live updates two plots, one duration and one for loss during training
    def _plot_durations(self, episode_durations, loss, score, show_result=False):
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        loss_t = torch.tensor(loss, dtype=torch.float)
        score_t = torch.tensor(score, dtype=torch.float)

        # Clear the previous plots
        self.loss_axis.clear()
        self.duration_axis.clear()
        self.score_axis.clear()

        self.loss_axis.set_title("Loss per batch")
        self.duration_axis.set_title("Episode Durations")
        self.score_axis.set_title("Episode Scores")

        self.loss_axis.plot(loss_t.numpy())
        self.duration_axis.plot(durations_t.numpy())
        self.score_axis.plot(score_t.numpy())

        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            self.duration_axis.plot(means.numpy())
        if len(loss_t) >= 100:
            means = loss_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            self.loss_axis.plot(means.numpy())
        if len(score_t) >= 100:
            means = score_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            self.score_axis.plot(means.numpy())

        # Redraw the figure
        self.fig.canvas.draw()
        plt.pause(0.01)

    def do_train(self, num_episodes, average_factor=100, target_avg_score=None):

        print("## Training for {} episodes".format(num_episodes), "with batch size {} ##".format(self.hyperparams.batch_size))
        
        episode_durations = []
        score_hist = []
        loss_per_batch = []
        steps_done = 0
        success = False
        avg_score = 0
        avg_speed = 0
        avg_loss = 0

        for i_episode in range(num_episodes):
            state, info = self.env.reset()
            episode_score = 0
            for t in count():
                output = self.target_model(state)
                action = self._epsilon_greedy_action(output, steps_done)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_score += reward.item()
                self.env.render()
                
                if not done:
                    next_state = obs
                else:
                    next_state = None
                self.memory.push(state, action, reward, next_state)
                
                state = next_state
                steps_done += 1

                loss = self._replay(off_policy=True)
                if loss is not None: loss_per_batch.append(loss.item())
                self._soft_model_update()

                if done:
                    episode_durations.append(t + 1)
                    score_hist.append(episode_score)
                    if i_episode < average_factor:
                        avg_score = sum(score_hist)/(i_episode+1)
                        avg_speed = steps_done/(i_episode+1)
                        avg_loss = sum(loss_per_batch)/(i_episode+1)
                    else:
                        avg_score = sum(score_hist[-average_factor:])/(average_factor)
                        avg_speed = sum(episode_durations[-average_factor:])/(average_factor)
                        avg_loss = sum(loss_per_batch[-average_factor:])/(average_factor)

                    self._print_status(i_episode, 
                                        num_episodes, 
                                        steps_done, 
                                        avg_loss, 
                                        avg_score,
                                        avg_speed)
                    break
                
                # Early stopping if not converging in this episode
                if t >= 1e5:
                    print("## Max steps reached, stopping at episode {} ##".format(i_episode))
                    score = avg_score = 0/(i_episode+1)
                    avg_speed = steps_done/(i_episode+1)
                    return episode_durations, score_hist, success, loss_per_batch, score, avg_speed
            
            if self.live_plot:
                self._plot_durations(episode_durations, loss_per_batch, score_hist)
            
            # success condition
            if target_avg_score is not None:
                if avg_score >= target_avg_score:
                    print("## Task solved in {} episodes ##".format(i_episode))
                    success = True
                    break
        
        self._print_status(i_episode, 
                            num_episodes, 
                            steps_done, 
                            avg_loss, 
                            avg_score,
                            avg_speed,
                            end="\n")
        
        if self.final_plot:
            if not self.live_plot:
                self._init_plot()
            self._plot_durations(episode_durations,loss_per_batch, score_hist, show_result=True)
            plt.ioff()
            plt.show()
            
        return episode_durations, score_hist, success, loss_per_batch, avg_score, avg_speed

    def _print_status(self, episode, num_episodes, steps_done, loss, score, speed, end='\r'):
        
        eval_score = self._evaluate_target_model()

        print("Step: {:7} Loss: {:4.4}".format(steps_done, loss) , 
            " - episode: {:3}/{:3}".format(episode+1, num_episodes),
            " - avg_score: {:4.4}".format(score),
            " - avg_speed: {:4.4} steps/episode".format(speed),
            " - eval score: {:4.4}".format(eval_score),
            "   ---",
             end=end, flush=True)