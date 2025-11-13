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
        # 扩展状态空间：位置 + 宝箱状态
        # Extended state space: position + treasure status
        
        # 1. 获取位置信息
        pos = [game_info.pos_x, game_info.pos_z]
        pos_id = int(pos[0] * 64 + pos[1])  # 位置ID (0-4095)
        
        # 2. 获取宝箱状态（从 game_info.treasure_status）
        # treasure_status 是长度为10的列表，1表示可收集，0表示已收集或未生成
        treasure_status = game_info.treasure_status
        
        # 3. 将宝箱状态编码为一个整数 (0-1023)
        # 使用二进制编码：第i位为1表示宝箱i可收集
        treasure_encoding = sum([treasure_status[i] * (2 ** i) for i in range(10)])
        
        # 4. 组合成完整状态
        # state = pos_id * 1024 + treasure_encoding
        full_state = pos_id * Config.TREASURE_COMBINATIONS + treasure_encoding
        
        self.logger.debug(f"Position: {pos}, Pos_ID: {pos_id}, Treasure Status: {treasure_status}, "
                         f"Treasure Encoding: {treasure_encoding}, Full State: {full_state}")
        
        return ObsData(feature=int(full_state))

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
