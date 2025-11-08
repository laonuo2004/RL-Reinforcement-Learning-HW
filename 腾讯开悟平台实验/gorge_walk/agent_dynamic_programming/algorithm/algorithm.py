#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import copy
import numpy as np


class Algorithm:
    def __init__(self, gamma, theta, episodes, state_size, action_size, logger):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.theta = theta
        self.episodes = episodes

        self.agent_policy = np.ones([self.state_size, self.action_size]) / self.action_size

        # select algorithm (value_iteration or policy_iteration)
        # 选择DP算法类型
        self.algo = "value_iteration"
        self.logger = logger

    def learn(self, F):
        assert self.algo in ["policy_iteration", "value_iteration"], "Invalid algorithm"

        if self.algo == "policy_iteration":
            self.policy_iteration(F)
        elif self.algo == "value_iteration":
            self.value_iteration(F)

    def policy_iteration(self, F):
        """
        Calculate optimal policy using policy iteration

        Args:
            - F (dict): transition function (state-action pair -> next state, reward, done)
            - episodes (int): number of episodes
            - gamma (float): discount factor
            - theta (float): threshold for convergence

        Returns:
            - policy (np.array): optimal policy
            - V (np.array): optimal state-value array
        """
        """
        使用策略迭代计算最优策略

            参数:
                - F (字典): 转移函数 (状态-动作对 -> 下一个状态, 奖励, 完成)
                - episodes (整数): 迭代次数
                - gamma (浮点数): 折扣因子
                - theta (浮点数): 收敛阈值

            返回:
                - policy (np.array): 最优策略
                - V (np.array): 最优状态值数组
        """
        policy = np.ones([self.state_size, self.action_size]) / self.action_size

        i = 0
        while i < self.episodes:
            V = self.policy_evaluation(policy, F)
            Q = self.q_value_iteration(V, F)
            new_policy = self.policy_improvement(Q)

            if np.allclose(policy, new_policy, atol=1e-3):
                break

            policy = copy.copy(new_policy)

            if i % 10 == 0:
                self.logger.info("Iteration {}".format(i))
            i += 1

        self.agent_policy = policy

        return policy, V

    def value_iteration(self, F):
        """
        Calculate optimal policy using value iteration

        Args:
            - F (dict): transition function (state-action pair -> next state, reward, done)
            - episodes (int): number of episodes
            - gamma (float): discount factor
            - theta (float): threshold for convergence

        Returns:
            - policy (np.array): optimal policy
            - V (np.array): optimal state-value array
        """
        """
        使用值迭代计算最优策略

            参数:
                - F (字典): 转移函数 (状态-动作对 -> 下一个状态, 奖励, 完成)
                - episodes (整数): 迭代次数
                - gamma (浮点数): 折扣因子
                - theta (浮点数): 收敛阈值

            返回:
                - policy (np.array): 最优策略
                - V (np.array): 最优状态值数组
        """
        V = np.zeros(self.state_size)

        i = 0
        while i < self.episodes:
            delta = 0

            for state in range(self.state_size):
                v = V[state]

                V[state] = max(self._get_value(state, action, F, V) for action in range(self.action_size))

                delta = max(delta, abs(v - V[state]))

            if delta < self.theta:
                self.episodes_self = i
                break

            policy = self.policy_improvement(self.q_value_iteration(V, F))

            if i % 10 == 0:
                self.logger.info("Iteration {}".format(i))
            i += 1

        self.agent_policy = policy

        return policy, V

    def policy_evaluation(self, policy, F):
        """Calculate state-value array for the given policy

        Args:
            policy (np.array): policy array
            F (dict): transition function (state-action pair -> next state, reward, done)
            gamma (float): discount factor
            theta (float): threshold for convergence

        Returns:
            V (np.array): state-value array for the given policy
        """
        """为给定策略计算状态价值数组

        参数:
            policy (np.array): 策略数组
            F (dict): 状态转移函数（状态-动作对 -> 下一个状态，奖励，完成）
            gamma (float): 折扣因子
            theta (float): 收敛阈值

        返回:
            V (np.array): 给定策略的状态价值数组
        """
        # Initialize state-value array (16,)
        # 初始化状态价值数组 (16,)
        V = np.zeros(self.state_size)
        delta = self.theta + 1

        while delta > self.theta:
            delta = 0
            # Loop over all states
            # 遍历所有状态
            for state in range(self.state_size):
                v = 0
                # Loop over all actions fot the given state
                # 遍历给定状态的所有动作
                for action, action_prob in enumerate(policy[state]):
                    v += action_prob * self._get_value(state, action, F, V)

                # Calculate delta between old and new value for the given state
                # 计算给定状态的旧值和新值之间的差值
                delta = max(delta, abs(v - V[state]))

                # Update state-value array
                # 更新状态值数组
                V[state] = v

        return V

    def q_value_iteration(self, V, F):
        """Calculate the Q value for all state-action pairs

        Args:
            V (np.array): array of state values obtained from policy evaluation
            F (dict): transition function (state-action pair -> next state, reward, done)
            gamma (float): discount factor

        Returns:
            Q (np.array): action-value array for the given state-action pair
        """
        """计算所有状态-动作对的Q值

        参数:
            V (np.array): 来自策略评估的状态值数组
            F (字典): 转移函数 (状态-动作对 -> 下一个状态, 奖励, 完成)
            gamma (浮点数): 折扣因子

        返回:
            Q (np.array): 给定状态-动作对的动作值数组
        """
        Q = np.zeros([self.state_size, self.action_size])

        for state in range(self.state_size):
            for action in range(self.action_size):
                Q[state][action] = self._get_value(state, action, F, V)

        return Q

    def policy_improvement(self, Q):
        """Improve the policy based on action value (Q)

        Args:
            V (np.array): array of state values obtained from policy evaluation
            gamma (float): discount factor
        """
        """基于动作值(Q)改进策略

        参数:
            V (np.array): 来自策略评估的状态值数组
            gamma (浮点数): 折扣因子
        """
        # Blank policy initialized with zeros
        # 初始化policy
        policy = np.zeros([self.state_size, self.action_size])

        for state in range(self.state_size):
            action_values = Q[state]

            # Update policy
            # 更新策略
            policy[state] = np.eye(self.action_size)[np.argmax(action_values)]

        return policy

    def _get_value(self, state, action, F, V):
        """Get value of the state-action pair

        Args:
            state (int): current state
            action (int): action taken
            F (dict): transition function (state-action pair -> next state, reward, done)
            gamma (float): discount factor
            V (np.array): state-value array

        Returns:
            value (float): value of the state-action pair
        """
        """获取状态-动作对的值

        参数:
            state (整数): 当前状态
            action (整数): 执行的动作
            F (字典): 转移函数 (状态-动作对 -> 下一个状态, 奖励, 完成)
            gamma (浮点数): 折扣因子
            V (np.array): 状态值数组

        返回:
            value (浮点数): 状态-动作对的值
        """
        value = 0

        try:
            next_state, reward, _ = F[str(state)][str(action)]
            if reward == 0:
                reward = -1
            value = reward + self.gamma * V[next_state]
        except KeyError:
            pass

        return value
