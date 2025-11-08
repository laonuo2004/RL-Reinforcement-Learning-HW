#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import create_cls, attached
from kaiwu_agent.agent.base_agent import BaseAgent
from agent_sarsa.conf.conf import Config
from agent_sarsa.algorithm.algorithm import Algorithm

ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.logger = logger

        # Initialize parameters
        # 参数初始化
        self.state_size = Config.STATE_SIZE
        self.action_size = Config.ACTION_SIZE
        self.learning_rate = Config.LEARNING_RATE
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON
        self.episodes = Config.EPISODES
        self.algorithm = Algorithm(self.gamma, self.learning_rate, self.state_size, self.action_size)

        super().__init__(agent_type, device, logger, monitor)

    @predict_wrapper
    def predict(self, list_obs_data):
        """
        The input is list_obs_data, and the output is list_act_data.
        """
        """
        输入是 list_obs_data, 输出是 list_act_data
        """
        state = list_obs_data[0].feature
        act = self._epsilon_greedy(state=state, epsilon=self.epsilon)

        return [ActData(act=act)]

    @exploit_wrapper
    def exploit(self, list_obs_data):
        state = list_obs_data[0].feature
        act = self._epsilon_greedy(state=state, epsilon=0.1)

        return [ActData(act=act)]

    def _epsilon_greedy(self, state, epsilon=0.1):
        """
        Epsilon-greedy algorithm for action selection
        """
        """
        ε-贪心算法用于动作选择
        """
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, self.action_size)

        # Exploitation
        # 探索
        else:
            """
            Break ties randomly
            If all actions are the same for this state we choose a random one
            (otherwise `np.argmax()` would always take the first one)
            """
            """
            随机打破平局,在某些情况下，当有多个动作或策略具有相同的评估值或优先级时，需要进行决策。
            为了避免总是选择第一个动作或策略，可以使用随机选择的方法来打破平局。以增加多样性和随机性
            """
            if np.all(self.algorithm.Q[state, :]) == self.algorithm.Q[state, 0]:
                action = np.random.randint(0, self.action_size)
            else:
                action = np.argmax(self.algorithm.Q[state, :])

        return action

    @learn_wrapper
    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def observation_process(self, raw_obs, game_info):
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

        # Feature #5: Graph features generation (obstacle information, treasure information, endpoint information)
        # 特征#5: 图特征生成(障碍物信息, 宝箱信息, 终点信息)
        local_view = [game_info.local_view[i : i + 5] for i in range(0, len(game_info.local_view), 5)]
        obstacle_map, treasure_map, end_map = [], [], []
        for sub_list in local_view:
            obstacle_map.append([1 if i == 0 else 0 for i in sub_list])
            treasure_map.append([1 if i == 4 else 0 for i in sub_list])
            end_map.append([1 if i == 3 else 0 for i in sub_list])

        # Feature #6: Conversion of graph features into vector features
        # 特征#6: 图特征转换为向量特征
        obstacle_flat, treasure_flat, end_flat = [], [], []
        for i in obstacle_map:
            obstacle_flat.extend(i)
        for i in treasure_map:
            treasure_flat.extend(i)
        for i in end_map:
            end_flat.extend(i)

        # Feature #7: Information of the map areas visited within the agent's current local view
        # 特征#7: 智能体当前局部视野中的走过的地图信息
        memory_flat = []
        for i in range(game_info.view * 2 + 1):
            idx_start = (pos[0] - game_info.view + i) * 64 + (pos[1] - game_info.view)
            memory_flat.extend(game_info.location_memory[idx_start : (idx_start + game_info.view * 2 + 1)])

        tmp_treasure_status = [x if x != 2 else 0 for x in game_info.treasure_status]

        raw_obs = np.concatenate(
            [
                state,
                pos_row,
                pos_col,
                end_treasure_dists,
                obstacle_flat,
                treasure_flat,
                end_flat,
                memory_flat,
                tmp_treasure_status,
            ]
        )

        pos = int(raw_obs[0])
        treasure_status = [int(item) for item in raw_obs[-10:]]
        state = 1024 * pos + sum([treasure_status[i] * (2**i) for i in range(10)])

        return ObsData(feature=int(state))

    def action_process(self, act_data):
        return act_data.act

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        np.save(model_file_path, self.algorithm.Q)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        try:
            self.algorithm.Q = np.load(model_file_path)
            self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            self.logger.info(f"File {model_file_path} not found")
            exit(1)
