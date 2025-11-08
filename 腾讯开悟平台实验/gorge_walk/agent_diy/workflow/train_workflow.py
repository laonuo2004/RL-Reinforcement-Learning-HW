#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from agent_diy.feature.definition import (
    sample_process,
    reward_shaping,
)
from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
from tools.train_env_conf_validate import check_usr_conf, read_usr_conf
import time
import math
import os


@attached
def workflow(envs, agents, logger=None, monitor=None):
    """
    Users can define their own training workflows here
    用户可以在此处自行定义训练工作流
    """

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    # check_usr_conf is a tool to check whether the game configuration is correct
    # It is recommended to perform a check before calling reset.env
    # check_usr_conf会检查游戏配置是否正确，建议调用reset.env前先检查一下
    valid = check_usr_conf(usr_conf, logger)
    if not valid:
        logger.error("check_usr_conf return False, please check")
        return

    env, agent = envs[0], agents[0]

    # model saving
    # 保存模型
    # agent.save_model()

    return
