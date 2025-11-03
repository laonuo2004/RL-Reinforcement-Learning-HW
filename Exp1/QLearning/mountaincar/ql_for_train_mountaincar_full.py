# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import gym
import time
import matplotlib


# epsilons-greedy strategy
class EGreedyExpStrategy():
    # 变量初始化
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=1000000):
        self.init_epsilon = init_epsilon # 初始epsilon
        self.min_epsilon = min_epsilon   # 最小epsilon
        self.epsilon = init_epsilon     
        self.decay_steps = decay_steps   # epsilon衰减步数
        # 借助 np.logspace创建一个递减的等比数列，作为epsilon的衰减系数
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0                       # 步数
        self.exploratory_action_taken = None # 标注是否采取探索动作

    def _epsilon_update(self):
        # 逐渐衰减的epsilon，探索程度随学习进程逐渐减小，当超过衰减步数时，采用预设的最小epsilon
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon

    # 按照ε-贪婪策略选择动作，平衡探索与利用
    def select_action(self, q_values, state):
         # 以（1-epsilon）的概率选择最佳动作（Q值最大的动作），侧重利用（exploitation）
        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
            self.exploratory_action_taken = False
        # 以epsilon的概率产生一个随机的动作，侧重探索（exploration）
        else:
            action = np.random.randint(len(q_values))
            self.exploratory_action_taken = True
        # 更新epsilon
        self._epsilon_update()
        return action
    
# 从环境获取状态，并离散化
def get_discrete_state(state):
    DISCRETE_OS_SIZE = [20,20] # 连续的位置离散到20个桶之一，连续的速度离散到20个桶之一
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int64)) # 转为整数


# Q-learning算法
def QLearning(env, lr, discount, epsilon, min_eps, episodes):
    # 确定状态空间的大小
    num_states = [20, 20]
    
    # 随机初始化 一个 Q table
    Q = np.random.uniform(low=-2, high=0, size=(num_states + [env.action_space.n]))
    
    # 初始化追踪记录rewards的变量
    reward_list = []
    ave_reward_list = []
    reach_rate = np.zeros(int(episodes/100)) # 每隔100 episodes统计一次
    r = 0
    
    # 指定训练策略
    training_strategy = EGreedyExpStrategy(init_epsilon=0.8, min_epsilon=0.01, decay_steps=500000)
    
    # 运行 Q learning 算法
    for i in range(episodes): # episodes为训练轮数
        # 初始化参数
        done = False
        tot_reward, reward = 0,0
        state = env.reset()
        # 状态离散化
        state_adj = get_discrete_state(state)
        while done != True:               
            # 根据epsilon-贪心策略，选择下一个动作
            action = training_strategy.select_action(Q[state_adj[0], state_adj[1]], state)
            # 获取奖励和下一个状态
            new_state, reward, done, info = env.step(action) 
            # 状态离散化
            new_state_adj = get_discrete_state(new_state)
            # 游戏结束状态，更新 Q table
            if done and new_state[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
            # 更新 Q table
            else:
                delta = lr * (reward + discount * np.max(Q[new_state_adj[0], new_state_adj[1]]) - 
                                 Q[state_adj[0], state_adj[1], action])
                Q[state_adj[0], state_adj[1],action] += delta                                 
            # 更新变量
            tot_reward += reward
            state_adj = new_state_adj
            if (i+1) % 50 == 0:
                env.render() # 更新并渲染画面

        if tot_reward != -200.0:
                reach_rate[r] = reach_rate[r] + 1 # 到达山顶次数加1
        # 记录每个episode下的总奖励
        reward_list.append(tot_reward)
        if (i+1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward) # 每100 episodes的平均奖励
            reward_list = [] 
            print('Episode {} Average Reward: {} reach times: {}'.format(i+1, ave_reward, reach_rate[r]))
            r = r + 1
    env.close()
    return ave_reward_list, reach_rate


if __name__ == "__main__":
    # 导入小车上山环境并初始化
    env = gym.make('MountainCar-v0')
    env.reset()

    # 运行 Q-learning 算法
    rewards, reach = QLearning(env, lr=0.2, discount=0.9, 
                               epsilon=0.8, min_eps=0, episodes=1500)

    # 绘制训练过程中的平均奖励和到达次数
    plt.plot(100*(np.arange(len(rewards)) + 1), rewards, label='Average Reward Curve')
    plt.plot(100*(np.arange(len(reach)) + 1), reach, label = "Reach times Curve")
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward & Reach times')
    plt.title('Average Reward & Reach times vs Episodes')
    plt.legend()  # 添加图例
    plt.show()
    # plt.savefig('rewards_and_reach_times.jpg')     
    plt.close()  