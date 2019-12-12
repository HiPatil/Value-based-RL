#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import numpy as np
from custom_environment import ENVIRONMENT
import cv2
import pickle
import random


# In[2]:


size=10
episodes = 25000
epsilon=0.9
discount = 0.95
learning_rate=0.1
total_reward=0
display_every = 500
render_every = 500
EPS_DECAY=0.998


# In[3]:


env = ENVIRONMENT(diagonal=False, size=10, num_enemy=3, num_food=1)


# In[4]:


class parameter():
    def __init__(self, size, episode, discount, epsilon, learning_rate, render_every=500, verbose_every=500, EPS_DECAY=0.998, random_start=True):
        self.episode = episodes
        self.discount= discount
        self.size = size
        self. learning_rate = learning_rate
        self.epsilon = epsilon
        self.render_every=render_every
        self.random_start = random_start
        self.verbose_every = display_every
        self.EPS_DECAY = EPS_DECAY

    def decay_epsilon(self):
        self.epsilon *= self.EPS_DECAY

# In[5]:


def q_table(size, action):
    q_table = np.random.randn(size,size,action)
    return q_table


# In[8]:


def q_improve(env, q, parameter, verbose=True):
    total_reward = 0
    for episode in range(parameter.episode):
        state, reward, terminal = env.startover(newpos=parameter.random_start)
        while not terminal:
            current_q = q[state[0],state[1],:]
            # to make policy e-greedy
            if random.random() > parameter.epsilon:
                action = np.argmax(current_q)
            else:
                action=env.sample_action()
            #Now take a step and see what happens
            next_state, (next_reward, terminal) = env.step(action)
            total_reward += next_reward
            future_q = q[next_state[0],next_state[1],:]
            
            q[state[0],state[1],action] =current_q[action] + parameter.learning_rate*(next_reward + parameter.discount*np.max(future_q) - current_q[action])
            
            if terminal and next_reward == 100:
                q[state[0],state[1],:]=0
            if episode%parameter.render_every == 0:
                env.render(100)
            state = next_state
        parameter.decay_epsilon()
        cv2.destroyAllWindows()
        if episode%parameter.verbose_every == 0 and verbose:
            print('Episode: ',episode,'state:',state,'| Total Average Reward:', total_reward/500,'| Epsilon:', parameter.epsilon)
            total_reward= 0
    return q
            


# In[9]:


env = ENVIRONMENT(diagonal=True, size=10, num_enemy = 3, num_food = 1)
q = q_table(size=10, action=4)
parameters = parameter(size, episodes, discount, epsilon, learning_rate)

# Test Environment
# print(env.startover())

# for i in range(10):
#     print(env.step(np.random.randint(0,4)))
#     env.render(100)

# cv2.destroyAllWindows()

# print(env.startover())


# Improve the Q-value table
q = q_improve(env, q, parameter=parameters)


# In[ ]:





# In[ ]:




