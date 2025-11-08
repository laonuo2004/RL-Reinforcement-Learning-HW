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

    STATE_SIZE = 64 * 64
    ACTION_SIZE = 4
    GAMMA = 0.9
    THETA = 1e-3
    EPISODES = 100

    # dimensionality of the sample
    # 样本维度
    SAMPLE_DIM = 214

    # Dimension of movement action direction
    # 移动动作方向的维度
    OBSERVATION_SHAPE = 250
