# -*- coding: utf-8 -*-

import os
# 使用经过处理的gym游戏环境，位于atari_wrappers.py中
# 其中包括一些对游戏画面的剪切操纵等，便于focus on RL algorithm
from atari_wrappers import wrap_deepmind, make_atari
# Python中的random模块用于生成随机数
import random
# Tqdm 是 Python 进度条库，可以在 Python 长循环中添加一个进度提示信息，
# 用户只需要封装任意的迭代器 tqdm(iterator)即可
from tqdm import tqdm
#  Collections这个模块实现了一些很好的数据结构，能帮助解决各种实际问题
# deque是一个双向队列，后面我们会用到
from collections import deque
# 倒入pytorch
import torch
# torch.nn（neural networks）里面包含构造神经网络的一些基本模块
# nn构建于autograd之上，可以用来定义和运行神经网络
# torch.nn封装了底层计算图的高度抽象的接口，这使得构建网络十分方便。
# torch.nn中的每一个模块都可以接受输入的tesnor，计算输出的tensor，而且还保存了
# 一些内部状态比如需要学习的tensor的参数等。同时nn包中不光有一些激活函数和卷积池化
# 操作，还包含常见的损失函数（loss functions），用来训练神经网络。
import torch.nn as nn
# torch.optim是一个实现了各种优化算法的库。大部分学界常用的模型方法在这里都能找到
# 并且其接口具备足够的通用性，使得未来能够集成更加复杂的网络优化方法。
import torch.optim as optim
# torch.nn.functional在功能上与torch.nn有一定相似性，在具体使用上稍有不同
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

# 设置全局字体为支持中文的字体，例如使用SimHei
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False



BATCH_SIZE = 16            # 在训练时每次从经验池中采样的样本数量
GAMMA = 0.9                # GAMMA值越高，表示我们希望agent更加关注未来回报
# 用于创建新的ϵ-greedy-decay策略，与之前去qlearning/sarsa不同，算法的ϵ的值不是不变的，
# 而是随着训练代数的增加而线性递减的。训练初期，策略更接近随机探索，因此ϵ的值应较大，
# 训练后期，策略倾向于选择具有最优Q值的动作，因此此时ϵ的值应较小。
# 所以，当训练的步数改变后，eps_decay_steps的值也要相应改变，使epsilon随着整个训练过程合理地减小。
EPS_START = 1.             # ϵ的初始值为1
EPS_END = 0.1              # ϵ的最小值为0.1，当衰减到0.1后，ϵ的值不再减小
EPS_DECAY = 100000         # 利用前100000步训练来进行ϵ衰减
TARGET_UPDATE = 1000       # 每隔多少步训练，来更新一次目标网络（将策略网络的参数复制到目标网络）
NUM_STEPS = 2e7            # 最大的训练迭代步数
M_SIZE = 100000            # 经验池的大小
POLICY_UPDATE = 4          # 每隔多少步训练，来优化一次策略网络
EVALUATE_FREQ = 100000     # 每隔多少步训练，保存一次模型


# 定义自已的网络：
    #1 需要继承nn.Module类，并实现forward方法。
    #2 一般把网络中具有可学习参数的层放在网络类的函数__init__()中，
    #3 不具有可学习参数的层(如ReLU)可放在构造函数中，也可不放在__init__()函数
    #3 中(而在forward中使用nn.functional来代替)
# 定义DQN网络
class DQN(nn.Module):
	# 定义类的时候，若是添加__init__方法，那么在创建类的实例的时候，实例会自动
	# 调用这个方法，一般用来对实例的属性进行初使化
    def __init__(self, h, w, outputs, device):
    	# 这是对继承自父类的属性进行初始化。子类DQN继承了父类nn.Module的属性和方法
        super(DQN, self).__init__()
        # PyTorch的nn.Conv2d（）是用于设置网络中的卷积层
        # 输入数据的通道数为4，输出数据的通道数为32，卷积核大小为（8，8）
        # 步长为4，不需要偏置项
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        # PyTorch的nn.Linear（）是用于设置网络中的全连接层
        # 第一个参数为输入的维度大小，第二个参数为输出的维度大小
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, outputs)
        # 确定用cpu或者gpu来进行计算
        self.device = device

    # 使用kaiming法初始化训练的参数
    def init_weights(self, m):
    	# 对于全连接层
        if type(m) == nn.Linear:
        	# 利用正态分布的方法来初始化参数
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            # m.bias.data.fill_将bias直接初始化为0.0（float类型）
            m.bias.data.fill_(0.0)
        # 对于卷积层
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    # 定义网络模型的正向传递的方式
    def forward(self, x):
        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        # ---------------------             2 填空           -----------------------------#
        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
    	
        

# 创建经验池
class ReplayMemory(object):
    def __init__(self, capacity, state_shape, n_actions, device):
    	# c,h,w分别代表输入图像数据的通道数、高、宽
        c,h,w = state_shape
        # capacity代表经验池的总的大小
        self.capacity = capacity
        # 数据计算使用的设备
        self.device = device
        # 返回一个形状为(capacity, c, h, w),类型为dtype，里面的每一个值都是0的tensor
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        # 返回一个形状为(capacity, 1),类型为dtype，里面的每一个值都是0的tensor
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        # 相当于一个指针，指向下一次要存储的位置是哪里
        self.position = 0
        # 当前经验池真实数据已经占用的大小
        self.size = 0

    # 将每一步的训练数据存入经验池中
    def push(self, state, action, reward, done):
    	# 存入每一步得到的states信息
        self.m_states[self.position] = state
        self.m_actions[self.position,0] = action
        self.m_rewards[self.position,0] = reward
        self.m_dones[self.position,0] = done
        # 更新位置指针，方便下一次的存储
        self.position = (self.position + 1) % self.capacity
        # 更新经验池大小信息，如果经验池未存满，则size的大小与position大小相同
        # 如果经验池已经存满，则size的大小保持capacity不变
        self.size = max(self.size, self.position)

    #数据的抽样  bs==BATCH_SIZE
    def sample(self, bs):
    	# torch.randint返回一个张量，这个张量由（low、high）间的随机整数组成
    	# i为一个长度为bs的一维tensor，tensor中的每一个数值的大小在0-self.size之间
        i = torch.randint(0, high=self.size, size=(bs,))
        # 从经验池中采样状态
        bs = self.m_states[i, :4]
        # 从经验池中采样对应的动作
        ba = self.m_actions[i].to(self.device)
        # 从经验池中采样对应的奖励
        br = self.m_rewards[i].to(self.device).float()
        # 从经验池中采样对应的游戏是否结束的状态
        bd = self.m_dones[i].to(self.device).float()
        return bs, ba, br, bd
    # 返回经验池目前的大小
    def __len__(self):
        return self.size

# 贪心策略选择行为
class ActionSelector(object):
    def __init__(self, INITIAL_EPSILON, FINAL_EPSILON, policy_net, EPS_DECAY, n_actions, device):
        self._eps = INITIAL_EPSILON                        # ϵ的初始值
        self._FINAL_EPSILON = FINAL_EPSILON                # ϵ的最终值
        self._INITIAL_EPSILON = INITIAL_EPSILON            # ϵ的初始值
        self._policy_net = policy_net                      # 策略网络
        self._EPS_DECAY = EPS_DECAY                        # ϵ衰减周期
        self._n_actions = n_actions                        # 行为空间的大小
        self._device = device                              # 计算所在的设备

    def select_action(self, state, training=False):
        # random() 方法返回随机生成的一个实数，在[0,1)范围内。
        sample = random.random()
        # 在ϵ衰减周期内进行ϵ衰减操作
        if training:
            self._eps -= (self._INITIAL_EPSILON - self._FINAL_EPSILON) / self._EPS_DECAY
            self._eps = max(self._eps, self._FINAL_EPSILON)

        #-------------------------------------------------------------------------------#
        #-------------------------------------------------------------------------------#
        #---------------------             1 填空           -----------------------------#
        #-------------------------------------------------------------------------------#
        #-------------------------------------------------------------------------------#
        # 实现epsilon-贪心策略选择动作a
        
        
        return a.numpy()[0, 0].item()

def main(test=False):
    # 判断是否可以使用gpu来加速计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置环境名称
    env_name="SpaceInvaders"
    # 太空侵略者游戏环境的预处理
    env_raw = make_atari('{}NoFrameskip-v4'.format(env_name))
    env = wrap_deepmind(env_raw, frame_stack=False, episode_life=True, clip_rewards=True)
    # 获取初始的环境状态
    n_frame = torch.from_numpy(env.reset())
    # 获取经过预处理的图像的高
    h = n_frame.shape[-2]
	# 经过预处理的图像的宽和高相等
    w=h
    # 获取环境的动作空间的大小
    n_actions = env.action_space.n
    # 创建策略网络对象
    policy_net = DQN(h, w, n_actions, device).to(device)
    # 创建目标网络对象
    target_net = DQN(h, w, n_actions, device).to(device)
    # 创建队列
    q = deque(maxlen=5)
    done = True

    if test:
        # 如果是测试，则加载预训练的模型文件
        policy_net.load_state_dict(torch.load(os.getcwd() + "/dqn_atari.pth"))
        NUM_STEPS = 10
    else:
        # 如果是训练，则初始化模型
        NUM_STEPS = 250000 # 25000000
        policy_net.apply(policy_net.init_weights)
    # 初始化目标网络
    target_net.load_state_dict(policy_net.state_dict())
    # 利用adam算法来进行模型的优化，学习率为0.0000625，eps是为了提高数值稳定性而添加到分母的一个项
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0000625, eps=1.5e-4)
    # 初始化经验池
    memory = ReplayMemory(M_SIZE, [5, h, w], n_actions, device)
    # 创建动作选择对象
    my_action = ActionSelector(EPS_START, EPS_END, policy_net, EPS_DECAY, n_actions, device)

    loss_list = []
    count_list = []
    def optimize_model(train, step):
        if not train:
            return
        # 从经验池采样 状态、动作、奖励、结束情况
        state_batch, action_batch, reward_batch, done_batch = memory.sample(BATCH_SIZE)

        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        # ---------------------             3 填空           -----------------------------#
        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        # 计算策略网络的估计价值
        q = 
        # 计算目标网络得到的目标价值
        nq = 
        # 计算期望的状态-价值函数值
        expected_state_action_values = 
        # 使用smooth_L1函数来计算损失
        loss = 

        # 模型中参数的梯度设为0
        optimizer.zero_grad()
        # loss的反向传播，计算当前梯度
        loss.backward()
        # 采用梯度截断Clip策略来避免梯度爆炸，将梯度约束在某一个区间内
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        # 更新模型
        optimizer.step()

        # 打印训练过程的loss
        loss_list.append(loss.cpu().item())
        count_list.append(step)
        # 不断更新显示绘制每一轮的奖励得分情况
        plt.plot(count_list, loss_list, c='r', ls='-', marker='o', mec='b', mfc='w')  
        # 设置x轴标题
        plt.xlabel('训练步数')
        # 设置y轴标题
        plt.ylabel('loss')
        # 图像每次显示0.02秒
        plt.pause(0.02)

    # 模型的测试
    def evaluate(step, policy_net, device, env, n_actions, eps=0.05, num_episode=5, test=False):
        # 创建环境
        env = wrap_deepmind(env)
        # 创建动作选择对象
        my_action = ActionSelector(eps, eps, policy_net, EPS_DECAY, n_actions, device)
        # 存储奖励
        e_rewards = []
        # 创建队列
        q = deque(maxlen=5)
        for i in range(num_episode):
            # 环境重置
            env.reset()
            e_reward = 0
            for _ in range(10):  
                n_frame, _, done, _ = env.step(0)
                n_frame = torch.from_numpy(n_frame).view(1,h,h)
                q.append(n_frame)
            # 测试模型
            while not done:
                # 渲染画面
                #env.render()
                state = torch.cat(list(q))[1:].unsqueeze(0)
                action = my_action.select_action(state, train)
                n_frame, reward, done, info = env.step(action)
                n_frame = torch.from_numpy(n_frame).view(1,h,h)
                q.append(n_frame)
                e_reward += reward
            e_rewards.append(e_reward)
        if test:
            print("SpaceInvaders-Test num: {} , rewards: {}".format(step + 1, (float(sum(e_rewards)) / float(num_episode))))
    # 使用tdqm封装迭代器range(NUM_STEPS)，用进度条的方式显示训练情况
    progressive = tqdm(range(NUM_STEPS), total=NUM_STEPS, ncols=50, leave=False, unit='b')
    for step in progressive:
        # 如果游戏结束，则重置游戏
        if done:
            env.reset()
            img, _, _, _ = env.step(1)
            # 前10针画面跳过，什么都不做，有利于增加出使画面的随机性，不容易陷入过拟合
            for i in range(10):
                n_frame, _, _, _ = env.step(0)
                n_frame = torch.from_numpy(n_frame).view(1,h,h)
                q.append(n_frame)

        #1 当经验池中数据量大于50000后才开始训练
        train = len(memory) > 50000
        # 对队列q的值进行拼接和增加纬度的操作
        state = torch.cat(list(q))[1:].unsqueeze(0)

        #2 选择动作
        action = my_action.select_action(state, train)

        #3 环境执行动作，并得到新的状态、奖励、本次游戏是否结束，其他信息
        n_frame, reward, done, info = env.step(action)
        # 在这里，将n_frame从numpy类转换成tensor类
        n_frame = torch.from_numpy(n_frame).view(1,h,h)

        q.append(n_frame)
        # 存入经验池
        memory.push(torch.cat(list(q)).unsqueeze(0), action, reward,done)

        # 模型测试
        if test:
            evaluate(step, policy_net, device, env_raw, n_actions, eps=0.01, num_episode=1, test=True)

        # 每隔POLICY_UPDATE步更新一次模型
        if step % POLICY_UPDATE == 0 and not test:
            optimize_model(train, step)

        # 在目标网络上执行一次更新
        if step % TARGET_UPDATE == 0 and not test and step!=0:
            target_net.load_state_dict(policy_net.state_dict())
        #模型的保存
        if step !=0 and step % EVALUATE_FREQ == 0 and not test:
            torch.save(policy_net.state_dict(), os.getcwd() + "/dqn_atari.pth")
    env.close()

if __name__ == "__main__":
    main()
