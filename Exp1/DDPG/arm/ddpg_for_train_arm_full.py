# -*- coding: utf-8 -*-
import pyglet
import numpy as np
import torch
from torch import nn
from torch.nn import init
from copy import deepcopy

MAX_EPISODES = 100000  # 要训练多少轮
MAX_EP_STEPS = 200     # 每轮最大多少步
ON_TRAIN = True        # 是否进行训练

LR_A = 0.001  # actor的学习率
LR_C = 0.001  # critic的学习率
GAMMA = 0.9   # 用于奖励的消耗
TAU = 0.01    # 用于参数的软替换
MEMORY_CAPACITY = 30000   # 经验池的大小
BATCH_SIZE = 32           # 由于一次训练的一批数据的量的大小

# 创建Actor网络
class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(ActorNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = torch.tensor(action_bound[1], dtype=torch.float)
        num_hiddens = 300
        # torch.nn.Sequential是一个有序的容器，神经网络模块将按照传入的顺序，
        # 依次被添加到计算图中执行，很方便快捷，Sequential会帮你实现一些操纵
        # 与torch.nn.Module的区别，使用nn.Module，我们可以根据自己的需求改变传播过程
        # 如果需要快速构建（无需定义Net），则直接使用torch.nn.Sequential即可
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, num_hiddens),
            nn.ReLU(),# ReLU激活函数
            nn.Linear(num_hiddens, self.action_dim),
            nn.Tanh() # tanh激活函数
        )
        # 权重初始化
        # named_parameters()输出模型中每一个参数的名称（字符串）与这个参数
        for name, params in self.net.named_parameters():
            if 'bias' in name:
            	# 使用0值填充偏执项
                init.constant_(params, val=0.)
            else:
            	# 从给定均值和标准差的正态分布N(mean, std)中生成值来填充变量
                init.normal_(params, mean=0., std=0.001)
    # 定义网络的前向传播
    def forward(self, states):
        actions = self.net(states)
        scaled_actions = actions * self.action_bound
        return scaled_actions
# 创建评判者网络
class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        num_hiddens = 300
        # 将参数存在字典类型里，可以像常规的Python字典一样对ParameterDict进行索引
        self.params = nn.ParameterDict({
            'w1_s': nn.Parameter(torch.randn(self.state_dim, num_hiddens) * 0.001),
            'w1_a': nn.Parameter(torch.randn(action_dim, num_hiddens) * 0.001),
            'b1': nn.Parameter(torch.zeros(1, num_hiddens))
        })
        self.linear = nn.Linear(num_hiddens, 1)
        for name, params in self.linear.named_parameters():
            if 'bias' in name:
            	# 使用0值填充偏执项
                init.constant_(params, val=0.)
            else:
            	# 从给定均值和标准差的正态分布N(mean, std)中生成值来填充变量
                init.normal_(params, mean=0., std=0.001)
    def forward(self, states, actions):
    	# torch.mm(a, b)是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，
    	# b的维度是(2, 3)，返回的就是(1, 3)的矩阵
        y1 = torch.mm(states, self.params['w1_s'])
        y2 = torch.mm(actions, self.params['w1_a'])
        y = torch.relu(y1 + y2 + self.params['b1'])
        q = self.linear(y)
        return q

# 这里DDPG类继承了object对象，将拥有更多可操作对象，包括python类中的高级特性
# 对于要着手写框架或者写大型项目来说，这些特性有时很有用
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        # 创建经验池，大小为MEMORY_CAPACITY
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        # 保存最后一次存储的位置的下一个位置
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self._build_actor()
        self._build_critic()

    def _build_actor(self):
        # 创建actor网络
        self.actor_eval_net = ActorNet(self.s_dim, self.a_dim, self.a_bound)
        # 定义actor网络的优化器  优化器中定义所要优化的参数以及学习率
        self.actor_optimizer = torch.optim.Adam(self.actor_eval_net.parameters(), lr=LR_A)
        # 创建actor目标网络
        self.actor_target_net = deepcopy(self.actor_eval_net)

    def _build_critic(self):
        # 创建评价网络
        self.critic_eval_net = CriticNet(self.s_dim, self.a_dim)
        # 定义评价网络的优化器   优化器中定义所要优化的参数以及学习率
        self.critic_optimizer = torch.optim.Adam(self.critic_eval_net.parameters(), lr=LR_C)
        # 创建评价网络的目标网络
        self.critic_target_net = deepcopy(self.critic_eval_net)

    # 网络权重的软更新
    def soft_replace(self):
    	# zip() 函数用于将可迭代的对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        for t, e in zip(self.actor_target_net.parameters(), self.actor_eval_net.parameters()):
            t.data.copy_((1 - TAU) * t + TAU * e)
        for t, e in zip(self.critic_target_net.parameters(), self.critic_eval_net.parameters()):
            t.data.copy_((1 - TAU) * t + TAU * e)
    # 选择动作
    def choose_action(self, state):
    	# np.newaxis的功能:插入新维度，为state增加一维
        state = state[np.newaxis, :]                                                
        tensor_state = torch.tensor(state, dtype=torch.float)
        tensor_action = self.actor_eval_net(tensor_state)
        return tensor_action.detach().numpy()[0]  # [[a]] ===> [a]

    def learn(self):
        # 软策略替换
        self.soft_replace()
        # 从memory中得到batch
        # np.random.choice是从MEMORY中以一定概率P随机选择BATCH_SIZE个, 这里p没有指定的时候相当于均匀分布
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        # 从batch中取相应的值
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        # 转化为torch.tensor
        tensor_batch_states = torch.tensor(bs, dtype=torch.float)
        tensor_batch_actions = torch.tensor(ba, dtype=torch.float)
        tensor_batch_rewards = torch.tensor(br, dtype=torch.float)
        tensor_batch_next_states = torch.tensor(bs_, dtype=torch.float)
        # 执行者网络前馈
        # 将sample出来的states输入到actor 网络中得到action
        tensor_predicted_batch_actions = self.actor_eval_net(tensor_batch_states)
        # 再将action与states输入到critic 网络中得到q值
        tensor_Q_critic_eval = self.critic_eval_net(tensor_batch_states, tensor_predicted_batch_actions)
        # 求actor_loss = mean(q)， 反向传播更新actor网络
        # 虽然a_loss是通过actor 网络和critic 网络最终求得， 但是此处只更新actor 网络
        actor_loss = - torch.mean(tensor_Q_critic_eval)
        # 更新actor 网络
        self.actor_optimizer.zero_grad()     # 梯度置零
        actor_loss.backward()                # 梯度计算
        self.actor_optimizer.step()          # 梯度更新
        # 将sample出来的actions和得states一起输入critic eval网络，得到q值
        tensor_Q_eval = self.critic_eval_net(tensor_batch_states, tensor_batch_actions)
        # 将sample出来的states输入到actor target网络得到action_
        tensor_batch_next_actions = self.actor_target_net(tensor_batch_next_states)
        # 将sample出来的states和之前得到的action_一起输入critic target网络， 得到q_值
        tensor_Q_next = self.critic_target_net(tensor_batch_next_states, tensor_batch_next_actions)
        # 计算q_target = sample出来的rewards + gamma * q_ 
        tensor_Q_target = tensor_batch_rewards + GAMMA * tensor_Q_next
        tensor_td_error = tensor_Q_target - tensor_Q_eval
        # 计算评价者网络的损失
        critic_loss = torch.mean(tensor_td_error * tensor_td_error)
        # 评价者网络更新
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    # 中间数据保存到经验池
    def store_transition(self, s, a, r, s_):
    	# np.hstack():在水平方向上堆叠
        transition = np.hstack((s, a, [r], s_))
        # 当前应该插入的位置
        index = self.pointer % MEMORY_CAPACITY  
        self.memory[index, :] = transition
        # self.pointer向后移动一位
        self.pointer += 1
    # 保存模型
    def save(self):
        torch.save(self.actor_eval_net, "ddpg_arm.pth")
    # 加载模型
    def restore(self):
        self.actor_eval_net = torch.load("ddpg_arm.pth",map_location=torch.device('cpu'))

# 自己创建一个机械手臂的环境
class ArmEnv(object):
    viewer = None             # 可视参数，决定是否开启可视化
    dt = .1                   # 单位时间 dt ，用于计算角速度
    action_bound = [-1, 1]    # 转动角动作的范围，弧度制，对应到角度约为（-57°，57°）
    goal = {'x': 100., 'y': 100., 'l': 40}     # 初始蓝色点的位置与大小
    state_dim = 9             # 状态信息的个数
    action_dim = 2            # 动作空间的大小（两个关节可动）

    def __init__(self):
        # arm_info用来存储每个机械臂的信息，如每个手臂的转动角，[(100., 0.), (100., 0.)]
        self.arm_info = np.zeros(2, dtype=[('l', np.float32), ('r', np.float32)])
        self.arm_info['l'] = 100        # 机械臂的长度信息
        self.arm_info['r'] = np.pi / 6  # 机械臂的角度信息
        self.on_goal = 0                # finger是否触到了目标点

    def step(self, action):
        done = False
        # 对输入的转角做一个范围限定，防止机械臂运动幅度过大
        # 计算单位时间 dt 内旋转的角度, 将角度限制在360度以内
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2  

        (a1l, a2l) = self.arm_info['l']  # l机械臂长度  半径
        (a1r, a2r) = self.arm_info['r']  # r弧度       角度
        a1xy = np.array([200., 200.])  # 屏幕的中心点
        # a1r作为a1的转动角。np.cos(弧度制)来计算cos值，a1l乘以a1r的余弦值，得到其在x轴上的投影
        # a1l乘以a1r的正弦值，得到其在y轴上的投影
        # 加上a1xy，即将a1l从（0，0）点沿着（（0，0），a1xy）的方向进行平移，得到此时的a1机械臂的位置信息
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 尾端 和 a2 的开始 (x1, y1)
        # 同样，这里（a1r + a2r）是a2的转动角，因为a1转了多少，a2就转了多少，再加上a2自己转的
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 尾端 (x2, y2)
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]   # 中间关节点到目标的距离（x、y的差）
        dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400] # 手臂末端到目标的距离（x、y的差）
        # reward 只用dist2
        r = -np.sqrt(dist2[0] ** 2 + dist2[1] ** 2)
        # 判断finger是否触到了目标点 和 奖励+1
        # 判断finger的x的值，是否在goal中心点x值的二分之一goal的长度的范围内
        if self.goal['x'] - self.goal['l'] / 2 < finger[0] < self.goal['x'] + self.goal['l'] / 2:
            if self.goal['y'] - self.goal['l'] / 2 < finger[1] < self.goal['y'] + self.goal['l'] / 2:
                r += 1.
                self.on_goal += 1
                if self.on_goal > 50:
                    done = True
        else:
            self.on_goal = 0

        # 拼接起来组成state       2            2              4                       1
        # 这里我们设计了9维的状态空间，分别是中间关节的位置*2，末端*2，两个距离*4，是否触摸到
        s = np.concatenate((a1xy_ / 200, finger / 200, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s, r, done
    # 环境重置
    def reset(self):
        # 随机goal的位置
        # 返回一个服从“0~1”均匀分布的随机样本值
        self.goal['x'] = np.random.rand() * 400.
        self.goal['y'] = np.random.rand() * 400.
        # 随机机械臂的两个角度
        self.arm_info['r'] = 2 * np.pi * np.random.rand(2)
        self.on_goal = 0
        (a1l, a2l) = self.arm_info['l']
        (a1r, a2r) = self.arm_info['r']
        a1xy = np.array([200., 200.])
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]
        s = np.concatenate((a1xy_ / 200, finger / 200, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s
    # 渲染仿真环境画面
    def render(self):
        # 调用viewer环境进行可视化
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal)
        self.viewer.render()
    # 随机生成转角
    def sample_action(self):
        return np.random.rand(2) - 0.5  # 两个 radians

# 将render功能与可视化功能分开编写，方便使用
class Viewer(pyglet.window.Window):
    bar_thc = 5    #机械臂的宽度

    def __init__(self, arm_info, goal):
        #                       窗口  宽          高          窗口尺寸不变      窗口名称      默认刷新频率
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
        # pyglet中设置背景颜色的方法，这里设置为白色
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arm_info = arm_info
        self.goal_info = goal
        self.center_coord = np.array([200, 200])   #屏幕中心点
        # 将所有的构建加入pyglet的batch中，方便显示图像
        self.batch = pyglet.graphics.Batch()  
        # goal为机械臂末端要到达的地方，定义其为多边形，同时定义四个顶点的位置
        # 四个顶点中，每个顶点都是通过当前goal的中点位置，加减二分之一goal的长度来计算
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,  # 位置
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))  # 区域颜色蓝色
        # 初始化arm1和arm2的位置和颜色
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,  # 位置
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))  # 区域颜色红色
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,  # 位置
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))  # 区域颜色红色
    def render(self):
        # 更新机械臂位置信息
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        # 刷新画面
        self.clear()
        # 显示图像信息
        self.batch.draw()
    # 更新机械臂位置信息
    def _update_arm(self):
        # 更新目标   由中心点坐标和物体长宽，求四个顶点的坐标
        # 四个顶点中，每个顶点都是通过当前goal的中点位置，加减二分之一goal的长度来计算
        self.goal.vertices = (
            self.goal_info['x'] - self.goal_info['l'] / 2, self.goal_info['y'] - self.goal_info['l'] / 2,
            self.goal_info['x'] + self.goal_info['l'] / 2, self.goal_info['y'] - self.goal_info['l'] / 2,
            self.goal_info['x'] + self.goal_info['l'] / 2, self.goal_info['y'] + self.goal_info['l'] / 2,
            self.goal_info['x'] - self.goal_info['l'] / 2, self.goal_info['y'] + self.goal_info['l'] / 2)

        # 角度和长度计算更新机械臂位置
        (a1l, a2l) = self.arm_info['l']  # l为机械臂的长度
        (a1r, a2r) = self.arm_info['r']  # r为弧度
        a1xy = self.center_coord  # a1 始终在屏幕中心
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 尾端 和 a2 的开始 (x1, y1)
        a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 尾端 (x2, y2)
        # 计算与a1、a2的方向垂直的角度
        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
        # 根据垂直的角度来计算各个机械手臂的4个点信息
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        # 组合新的手臂的位置信息
        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))
    # 监听鼠标事件，将鼠标位置转换为目标的位置
    def on_mouse_motion(self, x, y, dx, dy):
        self.goal_info['x'] = x
        self.goal_info['y'] = y


# 创建环境
env = ArmEnv()
s_dim = env.state_dim        # 状态空间维度
a_dim = env.action_dim       # 动作空间维度
a_bound = env.action_bound   # 动作的范围

# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
# ---------------------             1 填空           -----------------------------#
# -------------------------------------------------------------------------------#
# -------------------------------------------------------------------------------#
#1 创建DDPG智能体对象
rl = DDPG(a_dim, s_dim, a_bound)
steps = []
# 训练
def train():
    for i in range(MAX_EPISODES):# 循环训练的MAX_EPISODES次
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):# 每次训练环境最多执行MAX_EP_STEPS步
            # -------------------------------------------------------------------------------#
            # -------------------------------------------------------------------------------#
            # ---------------------             2 填空           -----------------------------#
            # -------------------------------------------------------------------------------#
            # -------------------------------------------------------------------------------#
            #2 利用ddpg模型选择行为
            a = rl.choose_action(s)
            # 获取环境的返回信息
            s_, r, done = env.step(a)
            # 经验存储
            rl.store_transition(s, a, r, s_)
            ep_r += r
            # -------------------------------------------------------------------------------#
            # -------------------------------------------------------------------------------#
            # ---------------------             3 填空           -----------------------------#
            # -------------------------------------------------------------------------------#
            # -------------------------------------------------------------------------------#
            #3 当经验池存满后开始训练
            if rl.pointer > MEMORY_CAPACITY:
                rl.learn()
            # 更新state
            s = s_
            if done or j == MAX_EP_STEPS - 1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
        # 保存模型
        if (i+1)%500==0:
            rl.save()

def eval():
    # 加载预训练模型
    rl.restore()
    # 开启可视化
    env.viewer.set_vsync(True)
    s = env.reset()
    while True:
    	# 选择动作
        a = rl.choose_action(s)
        # 执行动作
        s, r, done = env.step(a)

if ON_TRAIN:
    train()
else:
    eval()