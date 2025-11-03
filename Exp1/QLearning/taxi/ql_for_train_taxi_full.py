# -*- coding: utf-8 -*-

# 导入openai gym环境库
import gym
# 导入python的random库，random库是python中用于生成随机数的函数库
import random
# NumPy取自”Numeric(数值)“和”Python“的简写，它是处理数值计算最为基础的库。
# ‘as np’是给予numpy库一个“np”的别称，方便后面在代码中的使用
import numpy as np
# matplotlib.pyplot是一个函数集合，每一个pyplot函数都是一个和绘制图像相关的功能，
# 例如创建一幅图、在图中创建一个绘图区域、在绘图区域中添加一条线等等。
import matplotlib.pyplot as plt
import matplotlib

# 设置全局字体为支持中文的字体，例如使用SimHei
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# 打开图像显示的交互模式，图像将在显示一定时间后关闭
plt.ion()
# 这里是指创建一个自定义的图像，图像编号为1
plt.figure(1)
# 用来存储训练的轮次信息
count_list=[]
# 用来存储每一轮奖励的值
r_list=[]

# 创建一个出租车问题的环境对象，在gym环境中，出租车问题的环境名称为'Taxi-v3'
env = gym.make('Taxi-v3')
# 设置学习率，学习率越大，保留之前训练的效果就越多
alpha = 0.5
# 设置折扣因子，gamma越接近于1代表它越有远见，会着重考虑后续状态的的价值，
# 越接近0的时候就会变得近视只考虑当前的利益的影响
gamma = 0.9
# 设置贪心策略的ε，此处以0.05的概率随机选择动作
epsilon = 0.05

# 此环境有500种离散的不同状态，有6种不同的行为，因此Q表大小为500*6
# 此处 env.observation_space.n = 500
# 此处 env.action_space.n = 6 
# 现在，我们将Q表初始化为一个字典，该字典存储指定状态s中执行动作a的值的状态动作对。
Q = {}
# 利用python中的range函数创建一个大小为env.observation_space.n=500的整数列表
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        # Q表中的值初始值均为零
        Q[(s, a)] = 0

# 更新Q表           先前的状态     采取的动作       当前获得的奖励   接下来要转移到的状态     学习率    折扣因子
def update_q_table( prev_state,    action,          reward,        nextstate,            alpha,    gamma):
	# 从所有可能的状态-动作对中，选取对应的Q值最大的
    qa = max([Q[(nextstate, a)] for a in range(env.action_space.n)])
    # 通过更新规则更新先前状态的Q值。
    Q[(prev_state, action)] += alpha * (reward + gamma * qa - Q[(prev_state, action)])

# ε-贪婪策略              当前状态  ε的值
def epsilon_greedy_policy(state,  epsilon):
	#uniform() 方法将随机生成一个实数，它在 [x, y] 范围内。
    if random.uniform(0, 1) < epsilon:
    	#调用sample()这一方法将产生一个随机的动作
        return env.action_space.sample()
    else:
        # 这里利用python中的lambda表达式，将range(env.action_space.n)转化为list形式后，
        # 选取list中的最大的那个值，选取的依据是比较list元素里的每一个元素对应的Q值的大小
        return max(list(range(env.action_space.n)), key=lambda x: Q[(state, x)])

# 训练
# 这里我们循环训练1000次，每一次都以环境达到终止状态作为结束（一轮）
for i in range(1000):
	# python中的append() 方法用于在列表末尾添加新的对象。
    count_list.append(i)
    # 总reward初始化为0
    r = 0
    # env.reset()用于重置环境，回到游戏的初始状态
    state = env.reset()
    while True:
        # 渲染图像，用于输出当前的游戏状态
        env.render()
        # 使用贪心策略选取动作
        # Q-learning的特点在这里体现，Q-learning每一步都采用ε-贪婪策略重新选取动作，
        # 即下一步选取的动作跟上一步的动作没有直接关系
        # off-policy（异）：产生数据的策略与评估和要改善的策略不是同一个策略。
        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        # ---------------------             1 填空           -----------------------------#
        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        #1 选取动作
        action = epsilon_greedy_policy(state, epsilon)

        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        # ---------------------             2 填空           -----------------------------#
        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        #2 执行选取的动作得到新的状态，奖励，done的值为True或False，代表本轮游戏是否到了终止状态，
        #2 info为一些其他的信息，在这个问题里我们用不到
        nextstate, reward, done, info = env.step(action)

        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        # ---------------------             3 填空           -----------------------------#
        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        #3 更新Q表
        update_q_table(state, action, reward, nextstate, alpha, gamma)

        # 更新state，将nextstate的值更新给当前的状态，用于下一次的动作的选择
        state = nextstate
        # 更新总的奖励回报
        r += reward
        # 如果本次游戏结束，那么结束当前的循环
        if done:
            break
    # 打印每一轮训练奖励 
    print("[Episode %d] Total reward: %d" % (i + 1, r))
    # 将本轮训练的奖励记录到r_list中
    r_list.append(r)
    # 不断更新显示绘制每一轮的奖励得分情况
    plt.plot(count_list, r_list, c='r', ls='-', marker='o', mec='b', mfc='w')  
    # 设置x轴标题
    plt.xlabel('训练轮数')
    # 设置y轴标题
    plt.ylabel('奖励')
    # 图像每次显示0.01秒
    plt.pause(0.01)
# 退出当前环境
env.close()