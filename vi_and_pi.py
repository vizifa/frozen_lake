### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
        tuple of the form (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, diff_policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    
    #filling value_function with zeroes
    value_function = np.zeros(nS)
     
    ############################
    # YOUR IMPLEMENTATION HERE #
    delta = 1
    while delta > tol:
        delta = 0
        for s in range(nS):
            v = value_function[s]
            q = 0
            for a in range(nA):
                
                for state_probability, next_state, reward, terminated in P[s][a]:
                    # Calculate the expected value
                    action_probability = diff_policy[s][a]
                    q += action_probability * state_probability * (reward + gamma * value_function[next_state])
                
            value_function[s] = q
            
            delta = max(delta,abs(value_function[s]-v))
            
    print("policy eval done")
            
            
    ############################
    return value_function

def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """

    #diff_policy is a 2D matrix to store action probability of each state-action pair
    diff_policy = np.zeros((nS,nA))

    ############################
    # YOUR IMPLEMENTATION HERE #
    
    for s in range(nS):
        q = np.zeros(nA)
        for a in range(nA):
            for probability, next_state, reward, terminated in P[s][a]:
                q[a] += probability * (reward + gamma * value_from_policy[next_state])
                
        best_a = np.argwhere(q==np.max(q)).flatten()
        
        #everywhere q value is max, the probability is equally divided, everywhere else it is zero
        diff_policy[s] = np.sum([np.eye(env.nA)[i] for i in best_a], axis=0)/len(best_a)
    

    ############################
    return diff_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    diff_policy = np.ones((nS,nA))/nA 

    ############################
    # YOUR IMPLEMENTATION HERE #
    stable_policy = False
    
    while stable_policy != True:
        stable_policy = True
        
        old_diff_policy = diff_policy
        value_function = policy_evaluation(P, nS, nA, diff_policy)
        diff_policy = policy_improvement(P, nS, nA, value_function)
        
        #checking if policy is stable
        if np.max(abs(policy_evaluation(P, nS, nA, old_diff_policy) - policy_evaluation(P, nS, nA, diff_policy))) >= tol:
            stable_policy = False
            
    #converting diff_policy to the policy in given format
    for s in range(nS):
        best_action = 0
        best_act_value = diff_policy[s][0]
        for a in range(1,nA):
            if (diff_policy[s][a] >= best_act_value):
                best_act_value = diff_policy[s][a]
                best_action = a
        
        policy[s] = best_action
    
    ############################
    return value_function, policy 

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    diff_policy = np.ones((nS,nA))/nA 
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    #iterating value function
    delta = 1
    while delta>tol:
        delta = 0
        for s in range(nS):
            v = value_function[s]
            q = np.zeros(nA)
            for a in range(nA):
                
                for state_probability, next_state, reward, terminated in P[s][a]:
                    # Calculate the max value
                    q[a] =  state_probability * (reward + gamma * value_function[next_state])
            value_function[s] = np.max(q)
            
            delta = max(delta,abs(value_function[s]-v))
        
    print("value func eval done")
            
      
    #finding policy
    for s in range(nS):
        q = np.zeros(nA)
        a = value_function[s]
        for a in range(nA):
            for probability, next_state, reward, terminated in P[s][a]:
                q[a] += probability * (reward + gamma * value_function[next_state])
                
        best_a = np.argwhere(q==np.max(q)).flatten()
        #everywhere q value is max, the probability is equally divided, everywhere else it is zero
        diff_policy[s] = np.sum([np.eye(env.nA)[i] for i in best_a], axis=0)/len(best_a)
      
    #converting diff_policy to the policy in given format
    for s in range(nS):
        best_action = 0
        best_act_value = diff_policy[s][0]
        for a in range(1,nA):
            if (diff_policy[s][a] >= best_act_value):
                best_act_value = diff_policy[s][a]
                best_action = a
        
        policy[s] = best_action

    ############################
    return value_function, policy

def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    time.sleep(0.25)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render();
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
      print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":

    # comment/uncomment these lines to switch between deterministic/stochastic environments
    #env = gym.make("Deterministic-4x4-FrozenLake-v0")
    env = gym.make("Stochastic-4x4-FrozenLake-v0")
    #env = gym.make("Deterministic-8x8-FrozenLake-v0")
    
    print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)
    
    print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_vi, 100)
                      
