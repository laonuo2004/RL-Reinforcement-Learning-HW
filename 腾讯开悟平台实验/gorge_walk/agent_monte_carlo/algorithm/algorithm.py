#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np


class Algorithm:
    def __init__(self, gamma, state_size, action_size):
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size

        # Initialize the policy
        # 初始化策略
        self.policy = np.random.choice(self.action_size, self.state_size)
        self.Q = np.zeros([self.state_size, self.action_size])
        self.visit = np.zeros([self.state_size, self.action_size])

    def learn(self, list_sample_data):
        """
        Calculate the optimal policy using Monte Carlo Control - fist visit
            - list_sample is a list a sample: (state, action, reward)
        Return is calculated using the following formula:
            - G = R(t+1) + gamma * R(t+2) + ... + gamma^(T-t-1) * R(T)
        """
        """
        使用蒙特卡洛控制 - 首次访问来计算最优策略
        - list_sample 是一个样本列表：(状态, 动作, 奖励)
        使用以下公式计算返回值：
        - G = R(t+1) + gamma * R(t+2) + ... + gamma^(T-t-1) * R(T)
        """
        G, state_action_return = 0, []

        # Calculate the return for each state-action pair
        # 计算每个状态-动作对的回报
        for sample in reversed(list_sample_data[:-1]):
            state_action_return.append((sample.state, sample.action, G))
            G = self.gamma * G + sample.reward

        state_action_return.reverse()

        # Update the Q-table
        # 更新Q表
        seen_state_action = set()
        for state, action, G in state_action_return:
            if (state, action) not in seen_state_action:
                self.visit[state][action] += 1

                # calculate incremental mean
                # 计算递增均值
                self.Q[state, action] = self.Q[state, action] + (G - self.Q[state, action]) / self.visit[state, action]
                seen_state_action.add((state, action))

        # Update policy
        # 更新策略
        for state in range(self.state_size):
            best_action = np.argmax(self.Q[state])
            self.policy[state] = best_action

        return
