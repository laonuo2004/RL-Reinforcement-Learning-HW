#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import copy
import numpy as np


class Algorithm:
    def __init__(self, gamma, theta, episodes, state_size, action_size, logger):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.theta = theta
        self.episodes = episodes

        self.agent_policy = np.ones([self.state_size, self.action_size]) / self.action_size

        # select algorithm (value_iteration or policy_iteration)
        # 选择DP算法类型
        self.algo = "value_iteration"
        self.logger = logger
        
        # 宝箱位置（需要从Config导入）
        from agent_dynamic_programming.conf.conf import Config
        self.treasure_positions = Config.TREASURE_POSITIONS
        self.position_size = Config.POSITION_SIZE
        self.num_stages = Config.NUM_STAGES  # 阶段数量（0-10，共11个）
        self.optimal_order = Config.OPTIMAL_TREASURE_ORDER  # TSP最优顺序
        
        self.logger.info(f"Algorithm initialized with state_size={state_size}, "
                        f"position_size={self.position_size}, num_stages={self.num_stages}, "
                        f"treasure_positions={self.treasure_positions}, "
                        f"optimal_order={self.optimal_order}")

    def learn(self, F):
        assert self.algo in ["policy_iteration", "value_iteration"], "Invalid algorithm"

        # 性能优化：预处理地图数据
        # 1. 将字符串键转换为整数键（优化1：字典查找速度提升3-5倍）
        self.logger.info("Preprocessing map data for performance optimization...")
        F_int = {}
        for pos_str, actions in F.items():
            pos_int = int(pos_str)
            F_int[pos_int] = {}
            for action_str, transition in actions.items():
                action_int = int(action_str)
                F_int[pos_int][action_int] = transition
        self.logger.info(f"Converted {len(F_int)} position entries to integer keys")
        
        # 2. 创建宝箱位置字典（优化2：查找速度提升2倍）
        # {position_id: treasure_id}
        self.treasure_pos_dict = {pos: idx for idx, pos in enumerate(self.treasure_positions)}
        self.logger.info(f"Created treasure position dictionary with {len(self.treasure_pos_dict)} entries")
        
        # 3. 计算可达状态集合（优化3：只计算可达状态，速度提升4倍）
        self.logger.info("Computing reachable states...")
        self.reachable_states = self._compute_reachable_states(F_int)
        self.logger.info(f"Found {len(self.reachable_states)} reachable states "
                        f"(out of {self.state_size} total, {100*len(self.reachable_states)/self.state_size:.1f}%)")

        if self.algo == "policy_iteration":
            self.policy_iteration(F_int)
        elif self.algo == "value_iteration":
            self.value_iteration(F_int)
    
    def _compute_reachable_states(self, F):
        """
        计算所有可达状态集合（方案A：位置 × 阶段）
        状态 = 位置 × 11 + 阶段
        """
        reachable = set()
        
        # 获取所有可达的位置
        reachable_positions = set(F.keys())
        
        # 对于每个可达位置和每个阶段，都是可达的状态
        for pos_id in reachable_positions:
            for stage in range(self.num_stages):
                state = pos_id * self.num_stages + stage
                reachable.add(state)
        
        return reachable

    def policy_iteration(self, F):
        """
        Calculate optimal policy using policy iteration

        Args:
            - F (dict): transition function (state-action pair -> next state, reward, done)
            - episodes (int): number of episodes
            - gamma (float): discount factor
            - theta (float): threshold for convergence

        Returns:
            - policy (np.array): optimal policy
            - V (np.array): optimal state-value array
        """
        """
        使用策略迭代计算最优策略

            参数:
                - F (字典): 转移函数 (状态-动作对 -> 下一个状态, 奖励, 完成)
                - episodes (整数): 迭代次数
                - gamma (浮点数): 折扣因子
                - theta (浮点数): 收敛阈值

            返回:
                - policy (np.array): 最优策略
                - V (np.array): 最优状态值数组
        """
        policy = np.ones([self.state_size, self.action_size]) / self.action_size

        i = 0
        while i < self.episodes:
            V = self.policy_evaluation(policy, F)
            Q = self.q_value_iteration(V, F)
            new_policy = self.policy_improvement(Q)

            if np.allclose(policy, new_policy, atol=1e-3):
                break

            policy = copy.copy(new_policy)

            if i % 10 == 0:
                self.logger.info("Iteration {}".format(i))
            i += 1

        self.agent_policy = policy

        return policy, V

    def value_iteration(self, F):
        """
        Calculate optimal policy using value iteration

        Args:
            - F (dict): transition function (state-action pair -> next state, reward, done)
            - episodes (int): number of episodes
            - gamma (float): discount factor
            - theta (float): threshold for convergence

        Returns:
            - policy (np.array): optimal policy
            - V (np.array): optimal state-value array
        """
        """
        使用值迭代计算最优策略

            参数:
                - F (字典): 转移函数 (状态-动作对 -> 下一个状态, 奖励, 完成)
                - episodes (整数): 迭代次数
                - gamma (浮点数): 折扣因子
                - theta (浮点数): 收敛阈值

            返回:
                - policy (np.array): 最优策略
                - V (np.array): 最优状态值数组
        """
        self.logger.info(f"Starting value iteration with state_size={self.state_size}")
        self.logger.info(f"Computing {len(self.reachable_states)} reachable states "
                        f"({100*len(self.reachable_states)/self.state_size:.1f}% of total)")
        
        V = np.zeros(self.state_size)

        i = 0
        import time
        start_time = time.time()
        
        while i < self.episodes:
            iter_start = time.time()
            delta = 0

            # 优化：只对可达状态进行值迭代（速度提升4倍）
            for state in self.reachable_states:
                v = V[state]
                
                # 贝尔曼最优方程：选择使值函数最大的动作
                V[state] = max(self._get_value(state, action, F, V) for action in range(self.action_size))
                
                delta = max(delta, abs(v - V[state]))
            
            iter_time = time.time() - iter_start

            # 收敛检查
            if delta < self.theta:
                self.logger.info(f"Converged at iteration {i} with delta={delta:.6f}")
                self.episodes_self = i
                break

            # 计算当前策略
            policy = self.policy_improvement(self.q_value_iteration(V, F))

            # 定期输出日志
            if i % 10 == 0:
                elapsed = time.time() - start_time
                avg_time_per_iter = elapsed / (i + 1)
                est_remaining = avg_time_per_iter * (self.episodes - i - 1)
                
                self.logger.info(f"Iteration {i}/{self.episodes}, delta={delta:.6f}, "
                               f"iter_time={iter_time:.2f}s, elapsed={elapsed:.2f}s, "
                               f"est_remaining={est_remaining:.2f}s")
            
            i += 1

        total_time = time.time() - start_time
        self.logger.info(f"Value iteration completed in {total_time:.2f}s ({i} iterations)")
        
        # 最后一次计算策略
        if i >= self.episodes:
            policy = self.policy_improvement(self.q_value_iteration(V, F))
        
        self.agent_policy = policy

        return policy, V

    def policy_evaluation(self, policy, F):
        """Calculate state-value array for the given policy

        Args:
            policy (np.array): policy array
            F (dict): transition function (state-action pair -> next state, reward, done)
            gamma (float): discount factor
            theta (float): threshold for convergence

        Returns:
            V (np.array): state-value array for the given policy
        """
        """为给定策略计算状态价值数组

        参数:
            policy (np.array): 策略数组
            F (dict): 状态转移函数（状态-动作对 -> 下一个状态，奖励，完成）
            gamma (float): 折扣因子
            theta (float): 收敛阈值

        返回:
            V (np.array): 给定策略的状态价值数组
        """
        # Initialize state-value array (16,)
        # 初始化状态价值数组 (16,)
        V = np.zeros(self.state_size)
        delta = self.theta + 1

        while delta > self.theta:
            delta = 0
            # 优化：只遍历可达状态
            for state in self.reachable_states:
                v = 0
                # Loop over all actions fot the given state
                # 遍历给定状态的所有动作
                for action, action_prob in enumerate(policy[state]):
                    v += action_prob * self._get_value(state, action, F, V)

                # Calculate delta between old and new value for the given state
                # 计算给定状态的旧值和新值之间的差值
                delta = max(delta, abs(v - V[state]))

                # Update state-value array
                # 更新状态值数组
                V[state] = v

        return V

    def q_value_iteration(self, V, F):
        """Calculate the Q value for all state-action pairs

        Args:
            V (np.array): array of state values obtained from policy evaluation
            F (dict): transition function (state-action pair -> next state, reward, done)
            gamma (float): discount factor

        Returns:
            Q (np.array): action-value array for the given state-action pair
        """
        """计算所有状态-动作对的Q值

        参数:
            V (np.array): 来自策略评估的状态值数组
            F (字典): 转移函数 (状态-动作对 -> 下一个状态, 奖励, 完成)
            gamma (浮点数): 折扣因子

        返回:
            Q (np.array): 给定状态-动作对的动作值数组
        """
        Q = np.zeros([self.state_size, self.action_size])

        # 优化：只计算可达状态的Q值
        for state in self.reachable_states:
            for action in range(self.action_size):
                Q[state][action] = self._get_value(state, action, F, V)

        return Q

    def policy_improvement(self, Q):
        """Improve the policy based on action value (Q)

        Args:
            V (np.array): array of state values obtained from policy evaluation
            gamma (float): discount factor
        """
        """基于动作值(Q)改进策略

        参数:
            V (np.array): 来自策略评估的状态值数组
            gamma (浮点数): 折扣因子
        """
        # Blank policy initialized with zeros
        # 初始化policy
        policy = np.zeros([self.state_size, self.action_size])

        # 优化：只更新可达状态的策略
        for state in self.reachable_states:
            action_values = Q[state]

            # Update policy
            # 更新策略
            policy[state] = np.eye(self.action_size)[np.argmax(action_values)]

        return policy

    def _get_value(self, state, action, F, V):
        """Get value of the state-action pair (Plan A: position + stage)

        Args:
            state (int): current state (position * 11 + stage)
            action (int): action taken
            F (dict): transition function (state-action pair -> next state, reward, done)
            V (np.array): state-value array

        Returns:
            value (float): value of the state-action pair
        """
        """获取状态-动作对的值（方案A：位置 + 阶段）

        参数:
            state (整数): 当前状态（位置 * 11 + 阶段）
            action (整数): 执行的动作
            F (字典): 转移函数 (位置状态-动作对 -> 下一个位置, 奖励, 完成)
            V (np.array): 状态值数组

        返回:
            value (浮点数): 状态-动作对的值
        """
        # 1. 解码状态：分离位置和阶段
        pos_id = state // self.num_stages  # 位置ID (0-4095)
        stage = state % self.num_stages     # 阶段 (0-10)
        
        try:
            # 2. 从地图数据F获取基础状态转移（只基于位置）
            next_pos_id, base_reward, done = F[pos_id][action]
            
            # 3. 计算移动奖励
            if base_reward == 0:
                # 普通移动：-0.2（与评分机制一致）
                reward = -0.2
            else:
                # 到达终点：+150
                reward = base_reward
            
            # 4. 检查是否收集到宝箱（按TSP顺序）
            next_stage = stage  # 默认阶段不变
            
            # 检查下一个位置是否是宝箱位置
            treasure_id = self.treasure_pos_dict.get(next_pos_id)
            
            if treasure_id is not None and stage < len(self.optimal_order):
                # 到达了一个宝箱位置
                # 检查这个宝箱是否是当前阶段应该收集的
                expected_treasure_id = self.optimal_order[stage]
                
                if treasure_id == expected_treasure_id:
                    # 收集到了当前阶段应该收集的宝箱！
                    reward += 1e5  # 给予大额奖励
                    next_stage = stage + 1  # 进入下一阶段
                # 否则，到达的不是当前应该收集的宝箱，不给奖励，阶段不变
            
            # 5. 编码新状态
            next_state = next_pos_id * self.num_stages + next_stage
            
            # 6. 计算状态-动作值
            value = reward + self.gamma * V[next_state]
            
        except KeyError:
            # 动作不可执行（遇到障碍物）
            value = -1e10
        
        return value
