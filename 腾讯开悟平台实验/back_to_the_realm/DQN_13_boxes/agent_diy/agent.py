#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    BaseAgent,
)
from agent_diy.model.model import Model
from kaiwu_agent.utils.common_func import attached
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)

    @predict_wrapper
    def predict(self, list_obs_data):
        pass

    @exploit_wrapper
    def exploit(self, list_obs_data):
        pass

    @learn_wrapper
    def learn(self, list_sample_data):
        pass

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        pass

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        pass

    def observation_process(self, raw_obs, preprocessor, state_env_info=None):
        """
        This function is an important feature processing function, mainly responsible for:
            - Parsing information in the raw data
            - Parsing preprocessed feature data
            - Processing the features and returning the processed feature vector
            - Concatenation of features
            - Annotation of legal actions
        Function inputs:
            - raw_obs: Preprocessed feature data
            - state_env_info: Environment information returned by the game
        Function outputs:
            - observation: Feature vector
            - legal_action: Annotation of legal actions

        该函数是特征处理的重要函数, 主要负责：
            - 解析原始数据里的信息
            - 解析预处理后的特征数据
            - 对特征进行处理, 并返回处理后的特征向量
            - 特征的拼接
            - 合法动作的标注
        函数的输入：
            - raw_obs: 预处理后的特征数据
            - state_env_info: 游戏返回的环境信息
        函数的输出：
            - observation: 特征向量
            - legal_action: 合法动作的标注
        """
        pass

    def action_process(self, act_data):
        pass
