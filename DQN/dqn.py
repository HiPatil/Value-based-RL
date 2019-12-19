#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import gym
import numpy as np
import torch.optim as optim
import random
from torch.autograd import Variable


# In[3]:


class Model(nn.Module):
    def __init__(self, observation, action):
        super(Model, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(observation, 100, bias=False),
            nn.Linear(100, action, bias=False)
        )
        
    def forward(self, input):
        return self.network(input)


# In[4]:


done = False
learning_rate = 0.0001
discount = 0.95
batch_size=4
n_episodes = 25000
epsilon = 0.9
memory_d = 0
final_d = 100
episode_rewards = np.array([])
avg_loss = 0
rewards_path = 'rewards.npy'

env = gym.make("CartPole-v1")
env.reset()

# Make 2 models
q_hat = Model(env.observation_space.shape[0], env.action_space.n)
q_hat_target = Model(env.observation_space.shape[0], env.action_space.n)


# In[5]:


#Optimizer

criterion = nn.MSELoss()
optimizer = optim.SGD(q_hat.parameters(), lr = learning_rate)
        


# In[6]:


# env.close()
print(env.reset()) #[cart_position, cart_velocity, angle, angular velocity]


# In[7]:


for episode in range(n_episodes):
    state = torch.from_numpy(env.reset()).float()
    done = False
    episode_reward = 0
    
    
    if episode%300 == 0:
        print('-'*70)
        RENDER = True
    else:
        RENDER = False
        
    total_loss = 0
    count = 0
    while not done:
        if RENDER:
            env.render(1000)
        memory_d += 1
        
        if memory_d%final_d == 0:
            q_hat_target.load_state_dict(q_hat.state_dict()) # storing q_hat into q_hat_target for memory
            memory_d = 0
        
        q = q_hat(state)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(q).item()
            
        next_state, reward, done, _ = env.step(action)
        next_state = torch.from_numpy(next_state).float()
        
        q_target = q.clone()
        q_target[action] = torch.tensor(reward) + discount*torch.max(q_hat_target(next_state))
        
        if done:
            q_target[action]=torch.tensor(reward)
            
            
        #calculate loss
        loss = criterion(q,q_target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        
        episode_reward += reward
        total_loss += loss.item()
        
    epsilon *= 0.999
    episode_rewards = np.append(episode_rewards, episode_reward)
    avg_loss += total_loss
    
    env.close()
    if episode%100 == 0:
        print('Episode: %5d | Average Loss: %5.2f | Epsilon: %2.4f | Avg. Reward: %5.4f'%(episode, avg_loss/100, epsilon, np.mean(episode_rewards[episode-100:episode])))
        avg_loss = 0
        ep_reward = 0
        np.save(rewards_path, episode_rewards)


# In[ ]:





# In[ ]:




