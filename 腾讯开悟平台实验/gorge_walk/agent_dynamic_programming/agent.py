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
        # 方案A：位置 + 阶段状态
        # 状态 = 位置 × 11 + 阶段
        # 阶段表示按TSP顺序已收集了几个宝箱（0-10）
        
        # 1. 获取位置信息
        pos = [game_info.pos_x, game_info.pos_z]
        pos_id = int(pos[0] * 64 + pos[1])  # 位置ID (0-4095)
        
        # 2. 获取宝箱状态
        treasure_status = game_info.treasure_status  # [1/2, 1/2, ..., 1/2] 长度10
        
        # 3. 计算当前阶段（按TSP顺序已收集了几个宝箱）
        # OPTIMAL_TREASURE_ORDER = [0, 1, 2, 4, 5, 9, 8, 3, 6, 7]
        # treasure_status[i]: 1=可收集（未收集），2=未生成，0=已收集
        # 
        # 阶段定义：
        # 阶段0：还没按顺序收集任何宝箱
        # 阶段1：已按顺序收集了1个宝箱（宝箱0）
        # 阶段2：已按顺序收集了2个宝箱（宝箱0,1）
        # ...
        # 阶段10：已按顺序收集了全部10个宝箱
        stage = 0
        for treasure_id in Config.OPTIMAL_TREASURE_ORDER:
            if treasure_status[treasure_id] != 1:
                # 已收集或未生成，认为"已通过"这个阶段
                stage += 1
            else:
                # treasure_status[treasure_id] == 1，可收集（未收集）
                # 遇到第一个未收集的宝箱，阶段就停在这里
                break
        
        # 4. 组合成完整状态
        full_state = pos_id * Config.NUM_STAGES + stage
        
        self.logger.debug(f"Position: {pos}, Pos_ID: {pos_id}, Treasure_Status: {treasure_status}, "
                         f"Stage: {stage}, Full_State: {full_state}")
        
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
