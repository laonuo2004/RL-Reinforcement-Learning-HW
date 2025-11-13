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

    # 位置状态数：64*64 = 4096
    # 宝箱状态数：2^10 = 1024 (10个宝箱的所有组合)
    # 总状态数：4096 * 1024 = 4,194,304
    POSITION_SIZE = 64 * 64
    TREASURE_COMBINATIONS = 2 ** 10  # 10个宝箱的所有组合
    STATE_SIZE = POSITION_SIZE * TREASURE_COMBINATIONS  # 4,194,304
    ACTION_SIZE = 4
    GAMMA = 0.9
    THETA = 1e-3
    EPISODES = 250
    
    # 宝箱位置（状态ID）
    TREASURE_POSITIONS = [
        1230,  # 宝箱0: [19, 14], 19 * 64 + 14 = 1230
        604,   # 宝箱1: [9, 28], 9 * 64 + 28 = 604
        620,   # 宝箱2: [9, 44], 9 * 64 + 44 = 620
        2733,  # 宝箱3: [42, 45], 42 * 64 + 45 = 2733
        2071,  # 宝箱4: [32, 23], 32 * 64 + 23 = 2071
        3192,  # 宝箱5: [49, 56], 49 * 64 + 56 = 3192
        2298,  # 宝箱6: [35, 58], 35 * 64 + 58 = 2298
        1527,  # 宝箱7: [23, 55], 23 * 64 + 55 = 1527
        2657,  # 宝箱8: [41, 33], 41 * 64 + 33 = 2657
        3497   # 宝箱9: [54, 41], 54 * 64 + 41 = 3497
    ]

    # dimensionality of the sample
    # 样本维度
    SAMPLE_DIM = 214

    # Dimension of movement action direction
    # 移动动作方向的维度
    OBSERVATION_SHAPE = 250
