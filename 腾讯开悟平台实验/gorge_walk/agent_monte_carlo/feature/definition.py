#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import create_cls, attached


SampleData = create_cls("SampleData", state=None, action=None, reward=None)


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


def reward_shaping(frame_no, score, terminated, truncated, obs):
    reward = 0

    # Using the environment's score as the reward
    # 奖励1. 使用环境的得分作为奖励
    reward += score

    # Penalty for the number of steps
    # 奖励2. 步数惩罚
    if not terminated:
        reward += -1

    # The reward for obtaining a treasure chest
    # 奖励3. 获得宝箱的奖励, 默认不给宝箱奖励
    # if score > 0 and not terminated:
    #     reward += score

    return reward
