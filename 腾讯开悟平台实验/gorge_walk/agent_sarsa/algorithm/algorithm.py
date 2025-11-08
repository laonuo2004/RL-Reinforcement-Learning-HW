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
    def __init__(self, gamma, learning_rate, state_size, action_size):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.action_size = action_size

        # Reset the Q-table
        # 重置Q表
        self.Q = np.ones([self.state_size, self.action_size])

    def learn(self, list_sample_data):
        """
        Update the Q-table with the given game data:
            - list_sample: each sampple is (state, action, reward, next_state, next_action)
        Using the following formula to update q value:
            - Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * Q(s',a') - Q(s,a)]
        If next_state is the end, then next_action is -1
        """
        """
        使用给定的数据更新Q表格:
        list_sample:每个样本是[state, action, reward, new_state]
        使用以下公式更新Q值:
        Q(s,a) := Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        其中：
        Q(s,a) 表示状态s下采取动作a的Q值
        lr 是学习率(learning rate), 用于控制每次更新的幅度
        R(s,a) 是在状态s下采取动作a所获得的奖励
        gamma 是折扣因子(discount factor), 用于平衡当前奖励和未来奖励的重要性
        max Q(s',a') 表示在新状态s'下采取所有可能动作a'的最大Q值
        """
        sample = list_sample_data[0]
        state, action, reward = sample.state, sample.action, sample.reward
        next_state, next_action = sample.next_state, sample.next_action

        if next_action == -1:
            delta = reward - self.Q[state, action]
        else:
            delta = reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action]

        self.Q[state, action] += self.learning_rate * delta

        return
