#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
import os
from kaiwu_agent.utils.common_func import Frame, attached

from tools.train_env_conf_validate import check_usr_conf, read_usr_conf
from agent_target_dqn.feature.definition import (
    reward_shaping,
    sample_process,
)
from agent_target_dqn.feature.preprocessor import Preprocessor
from tools.metrics_utils import get_training_metrics


@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]
    epoch_num = 100000
    episode_num_every_epoch = 1
    g_data_truncat = 256
    last_save_model_time = 0

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_target_dqn/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error(f"usr_conf is None, please check agent_target_dqn/conf/train_env_conf.toml")
        return

    # check_usr_conf is a tool to check whether the game configuration is correct
    # It is recommended to perform a check before calling reset.env
    # check_usr_conf会检查游戏配置是否正确，建议调用reset.env前先检查一下
    valid = check_usr_conf(usr_conf, logger)
    if not valid:
        logger.error(f"check_usr_conf return False, please check")
        return

    for epoch in range(epoch_num):
        epoch_total_rew = 0
        data_length = 0
        for g_data in run_episodes(episode_num_every_epoch, env, agent, g_data_truncat, usr_conf, logger, monitor):
            data_length += len(g_data)
            total_rew = sum([i.rew for i in g_data])
            epoch_total_rew += total_rew
            agent.learn(g_data)
            g_data.clear()

        avg_step_reward = 0
        if data_length:
            avg_step_reward = f"{(epoch_total_rew/data_length):.2f}"

        # save model file
        # 保存model文件
        now = time.time()
        if now - last_save_model_time >= 120:
            agent.save_model()
            last_save_model_time = now

        logger.info(f"Avg Step Reward: {avg_step_reward}, Epoch: {epoch}, Data Length: {data_length}")


def run_episodes(n_episode, env, agent, g_data_truncat, usr_conf, logger, monitor):
    for episode in range(n_episode):
        collector = list()
        preprocessor = Preprocessor()

        # Retrieving training metrics
        # 获取训练中的指标
        training_metrics = get_training_metrics()
        if training_metrics:
            logger.info(f"training_metrics is {training_metrics}")

        # Reset the game and get the initial state
        # 重置游戏, 并获取初始状态
        obs, state_env_info = env.reset(usr_conf=usr_conf)

        # Disaster recovery
        # 容灾
        if obs is None:
            continue

        # At the start of each game, support loading the latest model file
        # The call will load the latest model from a remote training node
        # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
        agent.load_model(id="latest")

        # Feature processing
        # 特征处理
        obs_data, remain_info = agent.observation_process(obs, preprocessor, state_env_info)

        done = False
        step = 0
        bump_cnt = 0
        diy_2 = 0
        diy_3 = 0
        diy_4 = 0
        diy_5 = 0

        while not done:
            # Agent performs inference, gets the predicted action for the next frame
            # Agent 进行推理, 获取下一帧的预测动作
            act_data, model_version = agent.predict(list_obs_data=[obs_data])

            # Unpack ActData into action
            # ActData 解包成动作
            act = agent.action_process(act_data[0])

            # Interact with the environment, execute actions, get the next state
            # 与环境交互, 执行动作, 获取下一步的状态
            frame_no, _obs, score, terminated, truncated, _state_env_info = env.step(act)
            if _obs is None:
                break

            step += 1

            # Feature processing
            # 特征处理
            _obs_data, _remain_info = agent.observation_process(_obs, preprocessor, _state_env_info)
            # Disaster recovery
            # 容灾
            if truncated and frame_no is None:
                break

            treasures_num = 0

            # Calculate reward
            # 计算 reward
            if _obs is None:
                reward = 0
            else:
                (
                    reward,
                    is_bump,
                    reward_end_dist,
                    reward_exploration,
                    reward_treasure_dist,
                    reward_treasure,
                ) = reward_shaping(
                    frame_no,
                    score,
                    terminated,
                    truncated,
                    remain_info,
                    _remain_info,
                    obs,
                    _obs,
                )
                diy_2 += reward_end_dist
                diy_3 += reward_exploration
                diy_4 += reward_treasure_dist
                diy_5 += reward_treasure

                treasure_dists = [organ.status for organ in _obs.frame_state.organs]
                treasures_num = treasure_dists.count(1.0)

                # Wall bump behavior statistics
                # 撞墙行为统计
                bump_cnt += is_bump

            # Determine game over, and update the number of victories
            # 判断游戏结束, 并更新胜利次数
            if truncated:
                logger.info(
                    f"truncated is True, so this episode {episode} timeout, \
                        collected treasures: {treasures_num  - 7}"
                )
            elif terminated:
                logger.info(
                    f"terminated is True, so this episode {episode} reach the end, \
                        collected treasures: {treasures_num  - 7}"
                )
            done = terminated or truncated

            # Construct game frames to prepare for sample construction
            # 构造游戏帧，为构造样本做准备
            frame = Frame(
                obs=obs_data.feature,
                _obs=_obs_data.feature,
                obs_legal=obs_data.legal_act,
                _obs_legal=_obs_data.legal_act,
                act=act,
                rew=reward,
                done=done,
                ret=reward,
            )

            collector.append(frame)

            # If the number of game frames reaches the threshold, the sample is processed and sent to training
            # 如果游戏帧数达到阈值，则进行样本处理，将样本送去训练
            if len(collector) % g_data_truncat == 0:
                collector = sample_process(collector)
                yield collector

            # If the game is over, the sample is processed and sent to training
            # 如果游戏结束，则进行样本处理，将样本送去训练
            if done:
                if len(collector) > 0:
                    collector = sample_process(collector)
                    yield collector

                if monitor:
                    monitor_data = {"diy_2": diy_2, "diy_3": diy_3, "diy_4": diy_4, "diy_5": diy_5}
                    monitor.put_data({os.getpid(): monitor_data})

                break

            # Status update
            # 状态更新
            obs_data = _obs_data
            remain_info = _remain_info
            state_env_info = _state_env_info
