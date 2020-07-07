#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################################################################################
# About: Script containing utility functions, methods, and attributes to train DQN agents
# Authors: Sahil Dhingra, Miguel Bravo

# Source reference: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
# Source reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Source reference: INM 707 Deep Learning Optimisation, Lab 05 and Lab 06
###################################################################################################


###################################################################################################
# Import libraries
###################################################################################################

import gym
import numpy as np
import random
import torch
from collections import deque, namedtuple
import math
import cv2
cv2.ocl.setUseOpenCL(False)

###################################################################################################
# Env wrapper - to preprocess and store frames efficiently
# Source reference: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
###################################################################################################

def make_SI_env(env, stack_frames=True, episodic_life=True, clip_rewards=False, scale=False):
    if episodic_life:
        env = EpisodicLifeEnv(env)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=3) #changed to 3 for Space Invaders
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = WarpFrame(env)
    if stack_frames:
        env = FrameStack(env, 4)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]
    
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))
    
    
class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs
    
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs
    
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs
    
###################################################################################################
# Get state - last 4 stacked frames with 3 frames skipped in between
###################################################################################################
    
def get_state(observation):
    state = np.array(observation)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)
    
###################################################################################################
# Replay memory - to store transitions to train DQN agent
# Source reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Source reference: INM 707 Deep Learning Optimisation, Lab 05 and Lab 06
# adapted: Miguel & Sahil
###################################################################################################
        
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print('running on device: {}'.format(device))
        
# create transition object
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# basic replay memory
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
# prioritised replay memory
class PrioritisedReplayMemory(object):
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=10000):
        self.prob_alpha = alpha
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame = 1
        self.beta_start = beta_start
        self.beta_frames = beta_frames

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, *args):
        max_prio = self.priorities.max() if self.buffer else 1.0**self.prob_alpha

        total = len(self.buffer)
        if total < self.capacity:
            pos = total
            self.buffer.append(Transition(*args))
        else:
            prios = self.priorities[:total]
            probs = (1 - prios / prios.sum()) / (total - 1)
            pos = np.random.choice(total, 1, p=probs)

        self.priorities[pos] = max_prio

    def sample(self, batch_size):
        total = len(self.buffer)
        prios = self.priorities[:total]
        probs = prios / prios.sum()

        indices = np.random.choice(total, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Min of ALL probs, not just sampled probs
        prob_min = probs.min()
        max_weight = (prob_min*total)**(-beta)

        weights = (total * probs[indices]) ** (-beta)
        weights /= max_weight
        weights = torch.tensor(weights, device=device, dtype=torch.float)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = (prio + 1e-5)**self.prob_alpha

    def __len__(self):
        return len(self.buffer)
    
###################################################################################################
# Select action - exploration vs exploitation with epsilon decay
# Source reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Source reference: INM 707 Deep Learning Optimisation, Lab 05 and Lab 06
# adapted: Miguel & Sahil
###################################################################################################    

# training functions
def select_action(policy_net, n_actions, state, eps_start, eps_end, eps_decay, steps_done):
    #global steps_done #check steps done
    random_number = random.random()
    epsilon_val = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay) #Epsilon decay schedule
    # epsilon greedy policy
    if random_number > epsilon_val:
        # exploitation
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1, 1), epsilon_val
    else:
        # exploration
        if n_actions == 5:
            return torch.tensor([[random.randrange(0,5)]], device=device, dtype=torch.long), epsilon_val
        elif n_actions == 6:
            return torch.tensor([[random.randrange(0,6)]], device=device, dtype=torch.long), epsilon_val
        
###################################################################################################
# Optimise model
# Source reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Source reference: INM 707 Deep Learning Optimisation, Lab 05 and Lab 06
# adapted: Miguel & Sahil
################################################################################################### 
        
def optimize_model(policy_net, target_net, optimizer, gamma, memory, batch_size, 
                   double_dqn, prioritised_replay):
    
    # don't optimise model until batch size is reached 
    if len(memory) < batch_size:
        return
    
    # randomly sample batch from memory
    if prioritised_replay:
        transitions, ids, weights = memory.sample(batch_size)
    else:
        transitions = memory.sample(batch_size)
    
    # convert batch-arrays of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    
    #create tuples to store action and rewards
    actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward))) 
    
    # tensor cannot be None, so strip out terminal states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), 
                                              device=device, 
                                              dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to(device) 
    
    #create state, action reward batch
    state_batch = torch.cat(batch.state).to(device) 
    action_batch = torch.cat(actions) 
    reward_batch = torch.cat(rewards) 

    # get all Q values at current and next state
    online_Qs = policy_net(state_batch)
    next_online_Qs = policy_net(non_final_next_states)
    target_Qs = target_net(non_final_next_states)
    
    # get Q value for current state and action pair
    current_Q = online_Qs.gather(1, action_batch).squeeze()
    
    # get Q value at next state 
    next_Q = torch.zeros(batch_size, device=device)
    # double dqn - use online policy to select action, and target policy to evaluate it (i.e. calculate Q value)
    if double_dqn:
        next_Q[non_final_mask] = target_net(non_final_next_states).\
                                 gather(1, next_online_Qs.max(1)[1].unsqueeze(1)).squeeze().detach()
    else:
        # regular dqn - use online policy to select and evaluate action
        next_Q[non_final_mask] = target_Qs.max(1)[0].detach()
        #next_Q[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    #expected Q using Bellman Equation
    expected_Q = (next_Q * gamma) + reward_batch
    # compute loss
    #loss = F.smooth_l1_loss(current_Q, expected_Q.unsqueeze(1))
    diff = current_Q - expected_Q
    if prioritised_replay:
        loss = (0.5 * (diff * diff) * weights).mean() 
    else:
        loss = (0.5 * (diff * diff)).mean()
    
    # update memory (if prioritised replay used)
    if prioritised_replay:
        delta = diff.abs().detach().cpu().numpy().tolist() 
        memory.update_priorities(ids, delta)

    # optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return current_Q
