#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
from agent_diy.feature.definition import (
    sample_process,
    reward_shaping,
)
from tools.train_env_conf_validate import check_usr_conf, read_usr_conf


@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error(f"usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return
    # check_usr_conf is a tool to check whether the environment configuration is correct
    # It is recommended to perform a check before calling reset.env
    # check_usr_conf会检查环境配置是否正确，建议调用reset.env前先检查一下
    valid = check_usr_conf(usr_conf, logger)
    if not valid:
        logger.error(f"check_usr_conf return False, please check agent_diy/conf/train_env_conf.toml")
        return

    # Please write your DIY training process below
    # 请在下方写你DIY的训练流程

    # At the start of each game, support loading the latest model file
    # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
    agent.load_model(id="latest")

    # model saving
    # 保存模型
    agent.save_model()

    return
