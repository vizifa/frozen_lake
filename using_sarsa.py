import numpy as np
import gym
import time

#making the env
env = gym.make('FrozenLake-v0')

#Defining the different parameters
epsilon = 0.9
total_episodes = 10000
max_steps = 100
alpha = 0.85
gamma = 0.95

#Initializing the Q-matrix
Q = np.zeros((env.observation_space.n, env.action_space.n))


    
def epsilon_greedy(state):
    action = 0
    
    #explore:
    if np.random.uniform(0,1) < epsilon:
        action = env.action_space.sample()
    #greedy policy:
    else:
        action = np.argmax(Q[state, :])
    
    return action

#Function to learn the Q-value
def update(state, state2, reward, action, action2):
    #bootstrapped value:
    predict = Q[state, action]
    target = reward + gamma * Q[state2, action2]
    #update:
    Q[state, action] = Q[state, action] + alpha * (target - predict)
    
reward = 0

for episode in range(total_episodes):
    print("******episode ",episode, " ********")
    steps = 0
    state1 = env.reset()
    action1 = epsilon_greedy(state1)
    epsilon /= 1.001
    
    while(steps < max_steps):
        env.render()
        #time.sleep(0.25)
        state2, rew, done, info = env.step(action1)
        
        #Choosing the next action
        action2 = epsilon_greedy(state2)
        
        #Learning the Q-value
        update(state1, state2, rew, action1, action2)
        
        state1 = state2
        action1 = action2
        
        steps += 1
        reward += rew
        
        if done:
            break
        
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
      print("Episode reward:", rew)
      
print ("Performance : ", reward/total_episodes)