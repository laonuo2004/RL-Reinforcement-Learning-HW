"""
In this file, you will implement Policy Iterations and 
Q-learning algorithms.
"""

"""
------Below is the implementation of Policy Iteration------
You will implement policy iteration first.

For policy_evaluation, policy_improvement, policy_iteration 
and value_iteration, the parameters `P, nS, nA, gamma` are 
defined as follows:

- P: `Dict[int, Dict[int, List[Tuple[float, int, int, bool]]]]`
    For each pair of states in [1, nS] and actions in [1, nA],
    `P[state][action]` is a tuple of the form `(probability, 
    nextstate, reward, terminal)` where
        - `probability`: `float`
            the probability of transitioning from "state" to 
            "nextstate" with "action"
        - `nextstate`: `int`
            denotes the state we transition to (in range [0, 
            nS - 1])
        - `reward`: `int`
            either 0 or 1, the reward for transitioning from 
            "state" to "nextstate" with "action"
        - `terminal`: `bool`
            True when "nextstate" is a terminal state (hole or 
            goal), False otherwise
- `nS`: `int`
    number of states in the environment
- `nA`: `int`
    number of actions in the environment
- `gamma`: `float`
    Discount factor. Number in range [0, 1)
"""

import numpy as np
from typing import Dict, List, Tuple

__all__ = [
    "PType",
    "policy_iteration", 
    "QLearning"
]

PType = Dict[
    int, 
    Dict[
        int, 
        List[
            Tuple[
                float, 
                int, 
                int, 
                bool
            ]
        ]
    ]
]

def policy_evaluation(
    P: PType, 
    nS: int, 
    nA: int, 
    policy: np.ndarray, 
    gamma: float = 0.9, 
    tol: float = 1e-3
) -> np.ndarray:
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
            The value function of the given policy, where 
            value_function[s] is the value of state s
    """

    value_function = np.zeros(nS)
    while True:
        delta = 0
        for s in range(nS):
            v = value_function[s]
            a = policy[s]
            
            new_v = 0
            for prob, next_state, reward, terminal in P[s][a]:
                new_v += prob * (reward + gamma * value_function[next_state])
            value_function[s] = new_v
            delta = max(delta, abs(v - value_function[s]))
        
        if delta < tol:
            break
    
    return value_function


def policy_improvement(
    P: PType,
    nS: int,
    nA: int,
    value_from_policy: np.ndarray,
    policy: np.ndarray,
    gamma: float = 0.9
) -> np.ndarray:
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
            An array of integers. Each integer is the optimal 
            action to take in that state according to the 
            environment dynamics and the given value function.
    """

    new_policy = np.zeros(nS, dtype="int")
    for s in range(nS):
        q_values = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, terminal in P[s][a]:
                q_values[a] += prob * (reward + gamma * value_from_policy[next_state])
        
        new_policy[s] = np.argmax(q_values)
    
    return new_policy


def policy_iteration(
    P: PType, 
    nS: int, 
    nA: int, 
    gamma: float = 0.9, 
    tol: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs policy iteration.

    You should call the policy_evaluation() and 
    policy_improvement() methods to implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
            defined at beginning of file
    tol: float
            tol parameter used in policy_evaluation()
    

    Returns
    ----------
    value_function: np.ndarray[nS]
            The final value function of the policy after 
            convergence
    policy: np.ndarray[nS]
            The final policy after convergence
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    while True:
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
        
        new_policy = policy_improvement(P, nS, nA, value_function, policy, gamma)
        
        if np.array_equal(policy, new_policy):
            break
        
        policy = new_policy
    
    return value_function, policy


"""
------Below is the implementation of QLearning------

For the QLearning and SARSA functions, the parameters `env, 
num_episodes, gamma, lr, epsilon, epsilon_decay` are defined 
as follows:

- `env`: `gymnasium.Env`
    The environment used to compute the Q function. Must have 
    observation_space and action_space attributes.
- `num_episodes`: `int`
    Number of training episodes.
- `gamma`: `float`
    Discount factor. Value in range [0, 1)
- `lr`: `float`
    Learning rate. Value in range [0, 1)
- `epsilon`: `float`
    Epsilon value used in the ε-greedy method.
- `epsilon_decay`: `float`
    Rate at which epsilon decreases. Value in range [0, 1)

Both functions return a numpy array of shape [nS x nA], 
representing state-action values.
"""
import gymnasium
from datetime import datetime

def QLearning(
    env:gymnasium.Env, 
    num_episodes=2000, 
    gamma=0.9, 
    lr=0.1, 
    epsilon=0.8, 
    epsilon_decay=0.99
) -> np.ndarray:
    """Learn state-action values using the Q-learning 
    algorithm with epsilon-greedy exploration strategy.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gymnasium.Env
        Environment to compute Q function for. Must have nS, 
        nA, and P as
    attributes.
    num_episodes: int
        Number of episodes of training.
    gamma: float
        Discount factor. Number in range [0, 1)
    learning_rate: float
        Learning rate. Number in range [0, 1)
    epsilon: float
        Epsilon value used in the epsilon-greedy method.
    epsilon_decay: float
        Rate at which epsilon falls. Number in range [0, 1)

    Returns
    -------
    np.array
        An array of shape [nS x nA] representing state, action 
        values
    """

    nS:int = env.observation_space.n
    nA:int = env.action_space.n
    Q = np.zeros((nS, nA))
    
    # 用于监控训练进度
    total_rewards = []
    success_count = []
    start_time = datetime.now()
    max_steps_per_episode = 1e6  # 防止无限循环
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < max_steps_per_episode:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Q-learning 更新公式：Q(s,a) ← Q(s,a) + α[r + γ max_a Q(s',a) - Q(s,a)]
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state, best_next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += lr * td_error
            
            state = next_state
            steps += 1
        
        epsilon = max(0.1, epsilon * epsilon_decay)
        total_rewards.append(episode_reward)
        success_count.append(1 if episode_reward > 0 else 0)
        
        # 每 100 个 episode 打印一次进度
        if (episode + 1) % 100 == 0:
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            avg_reward = np.mean(total_rewards[-100:])
            success_rate = np.mean(success_count[-100:]) * 100
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward (last 100): {avg_reward:.3f}, "
                  f"Success Rate (last 100): {success_rate:.3f}%, "
                  f"Time: {elapsed_time:.2f}s, "
                  f"Epsilon: {epsilon:.3f}")
    
    return Q
