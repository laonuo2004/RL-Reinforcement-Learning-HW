#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# Configuration of dimensions
# 关于维度的配置
class Config:

    STATE_SIZE = 64 * 64 * 1024
    ACTION_SIZE = 4
    LEARNING_RATE = 0.8
    GAMMA = 0.9
    EPSILON = 0.1
    EPISODES = 10000

    # dimensionality of the sample
    # 样本维度
    SAMPLE_DIM = 5

    # Dimension of observation
    # 观察维度
    OBSERVATION_SHAPE = 250
