#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import copy
from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np
from kaiwu_agent.agent.base_agent import BaseAgent
from kaiwu_agent.agent.base_agent import (
    save_model_wrapper,
    learn_wrapper,
    load_model_wrapper,
)
from agent_dynamic_programming.conf.conf import Config
from agent_dynamic_programming.algorithm.algorithm import Algorithm

ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.logger = logger

        self.algorithm = Algorithm(
            Config.GAMMA, Config.THETA, Config.EPISODES, Config.STATE_SIZE, Config.ACTION_SIZE, self.logger
        )

        super().__init__(agent_type, device, logger, monitor)

    def predict(self, state):
        return np.argmax(self.algorithm.agent_policy[state])

    def exploit(self, state):
        return np.argmax(self.algorithm.agent_policy[state])

    @learn_wrapper
    def learn(self, F):
        self.algorithm.learn(F)

    def observation_process(self, raw_obs, game_info):
        # By default, only positional information is used as features. If additional feature processing is performed,
        # corresponding modifications are needed for the Policy structure, predict, exploit, and learn methods of the algorithm.
        # 默认仅使用位置信息作为特征, 如进行额外特征处理, 则需要对算法的Policy结构, predict, exploit, learn进行相应的改动

        pos = [game_info.pos_x, game_info.pos_z]
        # Feature #1: Current state of the agent (1-dimensional representation)
        # 特征#1: 智能体当前 state (1维表示)
        state = [int(pos[0] * 64 + pos[1])]
        # Feature #2: One-hot encoding of the agent's current position
        # 特征#2: 智能体当前位置信息的 one-hot 编码
        pos_row = [0] * 64
        pos_row[pos[0]] = 1
        pos_col = [0] * 64
        pos_col[pos[1]] = 1

        # Feature #3: Discretized distance of the agent's current position from the endpoint
        # 特征#3: 智能体当前位置相对于终点的距离(离散化)
        # Feature #4: Discretized distance of the agent's current position from the treasure
        # 特征#4: 智能体当前位置相对于宝箱的距离(离散化)
        end_treasure_dists = raw_obs

        raw_obs = np.concatenate(
            [
                state,
                pos_row,
                pos_col,
                end_treasure_dists,
            ]
        )

        return ObsData(feature=int(raw_obs[0]))

    def action_process(self, act_data):
        pass

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        np.save(model_file_path, self.algorithm.agent_policy)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        try:
            self.algorithm.agent_policy = np.load(model_file_path)
            self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            self.logger.info(f"file {model_file_path} not found")
            exit(1)
