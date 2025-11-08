#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import attached
import time
import os
from tools.map_data_utils import read_map_data
from tools.train_env_conf_validate import check_usr_conf, read_usr_conf
from tools.metrics_utils import get_training_metrics


@attached
def workflow(envs, agents, logger=None, monitor=None):

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_dynamic_programming/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_dynamic_programming/conf/train_env_conf.toml")
        return

    # check_usr_conf is a tool to check whether the game configuration is correct
    # It is recommended to perform a check before calling reset.env
    # check_usr_conf会检查游戏配置是否正确，建议调用reset.env前先检查一下
    valid = check_usr_conf(usr_conf, logger)
    if not valid:
        logger.error("check_usr_conf return False, please check")
        return
    env, agent = envs[0], agents[0]

    # Initializing monitoring data
    # 监控数据初始化
    monitor_data = {
        "reward": 0,
        "diy_1": 0,
        "diy_2": 0,
        "diy_3": 0,
        "diy_4": 0,
        "diy_5": 0,
    }

    logger.info("Start Training...")
    start_t = time.time()

    # Setting the state transition function
    # 设置状态转移函数
    map_data_file = "conf/map_data/F_level_1.json"
    map_data = read_map_data(map_data_file)
    if map_data is None:
        logger.error(f"map_data from file {map_data_file} failed, please check")
        return

    agent.learn(map_data)

    logger.info(f"Training time cost: {time.time() - start_t} s")

    # Reporting training progress
    # 上报训练进度
    monitor_data["reward"] = 0
    if monitor:
        monitor.put_data({os.getpid(): monitor_data})

    # model saving
    # 保存模型
    agent.save_model()
    # Retrieving training metrics
    # 获取训练中的指标
    training_metrics = get_training_metrics()
    if training_metrics:
        logger.info(f"training_metrics is {training_metrics}")

    return
