"""
This script is the main entry point for the project. It uses Hydra to manage the 
configuration and runs the selected algorithm on the FrozenLake environment.

This file does not need to be modified. The algorithms that need to be modified 
are in the algorithm.py files.
"""

import gymnasium
import hydra
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv, generate_random_map
from algorithm import PType, policy_iteration, QLearning
from config import Config
from typing import Callable

def render_result(
    env: gymnasium.Env, 
    policy: np.ndarray = None, 
    max_steps: int = 100,
    test_iter: int = 100,
):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gymnasium.Env
        Environment to play on. Must have nS, nA, and P as
        attributes.
    Policy: np.array of shape [nS] or [nS x nA]
        The action to take at a given state or the state-action values.
    """
    episode_reward = 0
    average_steps = 0
    for iter in range(test_iter):
        state, info = env.reset()
        for t in range(max_steps):
            if policy is None:
                action = env.action_space.sample()
            else:
                if policy.ndim == 2:
                    action = np.argmax(policy[state])
                else:
                    action = policy[state]
            state, reward, done, _, _ = env.step(action)
            # if reward == 0:
            #     reward = 2
            episode_reward += reward
            if done:
                average_steps += t
                break
        if not done:
            print(f"The agent didn't reach a terminal state in {max_steps} steps.")
    print("Episode reward: %f" % episode_reward)
    print("Average number of steps: %f" % (average_steps / test_iter))

def get_method(algorithm: str) -> Callable:
    if algorithm == "policy_iteration":
        return policy_iteration
    elif algorithm == "QLearning":
        return QLearning
    else:
        return None

@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(cfg: Config) -> None:
    # from omegaconf import DictConfig, OmegaConf
    # print(OmegaConf.to_yaml(cfg))
    # return    
    env = FrozenLakeEnv(
        desc = generate_random_map(
            size = cfg.env.map_size, 
            p = cfg.env.frozen_prob,
            seed = cfg.env.seed
        ),
        is_slippery = cfg.env.is_slippery,
        render_mode = cfg.env.render_mode
    )

    P: PType = env.P
    nS:int = env.observation_space.n
    nA:int = env.action_space.n

    gamma:float = cfg.policy_iteration.gamma
    tol:float = cfg.policy_iteration.tol

    num_episodes = cfg.qlearning.num_episodes
    gamma = cfg.qlearning.gamma
    learning_rate = cfg.qlearning.learning_rate
    epsilon = cfg.qlearning.epsilon
    epsilon_decay = cfg.qlearning.epsilon_decay
    if cfg.env.render_mode == 'human':
        test_iter = 1
    else:
        test_iter = 100

    def _play(method: Callable):
        if method is None:
            print(f"\n{'-' * 27}\nBeginning RANDOM_SAMPLE\n{'-' * 27}")
            render_result(env, max_steps=cfg.render.max_steps)
            return

        print(f"\n{'-' * 27}\nBeginning {method.__name__.upper()}\n{'-' * 27}")
        if method.__name__ == "QLearning":
            env.render_mode = 'ansi'
            # during training, we don't want to render the environment
            Q = method(
                env, 
                num_episodes, 
                gamma, 
                learning_rate, 
                epsilon, 
                epsilon_decay
            )
            env.render_mode = cfg.env.render_mode
            render_result(env, Q, cfg.render.max_steps, test_iter)
        elif method.__name__ == "policy_iteration":
            V, p = method(P, nS, nA, gamma, tol)
            render_result(env, p, cfg.render.max_steps, test_iter)
        else:
            raise ValueError(f"Method {method.__name__} not recognized.")
    
    _play(get_method(cfg.algorithm))
    env.close()

if __name__ == "__main__":
    main()
