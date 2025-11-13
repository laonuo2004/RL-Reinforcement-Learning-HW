# coding :utf-8

import argparse
import datetime
import time
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gym
from torch import nn
import numpy as np
import torch
import os
import pathlib
from pathlib import Path
import random
import matplotlib.pyplot as plt


class PPOMemory:
    """PPO算法的经验回放缓冲区，用于存储训练过程中的交互数据"""
    def __init__(self, batch_size):
        # 初始化存储列表
        self.states = []  # 状态列表
        self.probs = []   # 动作概率列表
        self.vals = []    # 状态价值估计列表
        self.actions = [] # 动作列表
        self.rewards = [] # 奖励列表
        self.dones = []   # 终止标志列表（是否结束该回合）
        self.batch_size = batch_size  # 数据批处理大小

    def sample(self):
        """
        从缓冲区中采样数据用于训练
        返回：状态、动作、旧概率、价值、奖励、终止标志及批次索引
        """
        # 生成批次的起始索引
        batch_step = np.arange(0, len(self.states), self.batch_size)
        # 生成所有数据的索引并打乱
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)
        # 按批次大小划分索引
        batches = [indices[i:i + self.batch_size] for i in batch_step]
        # 返回所有数据的数组形式及批次索引
        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def push(self, state, action, probs, vals, reward, done):
        """将单步交互数据存入缓冲区"""
        # *****************************     实现经验存储       ****************************#

        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        # --------------------------         1 填空       --------------------------------#
        # -------------------------------------------------------------------------------#
        # -------------------------------------------------------------------------------#
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        """清空缓冲区数据（每轮更新后调用）"""
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


"""策略网络（Actor）：输入状态，输出动作的概率分布"""
class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim):
        super(Actor, self).__init__()
        # 定义策略网络结构：3层全连接网络，ReLU激活，最后用Softmax输出概率
        self.actor = nn.Sequential(
            nn.Linear(n_states, hidden_dim),   # 输入层：状态维度 -> 隐藏层维度
            nn.ReLU(), # 激活函数
            nn.Linear(hidden_dim, hidden_dim), # 隐藏层：隐藏层维度 -> 隐藏层维度
            nn.ReLU(), # 激活函数
            nn.Linear(hidden_dim, n_actions),  # 输出层：隐藏层维度 -> 动作维度
            nn.Softmax(dim=-1)                 # 输出动作概率分布
        )

    def forward(self, state):
        """前向传播：输入状态，返回动作的概率分布"""
        dist = self.actor(state)  # 得到动作概率
        # 构建分类分布（适用于离散动作空间）
        dist = torch.distributions.categorical.Categorical(dist)
        return dist


"""价值网络（Critic）：输入状态，输出状态的价值估计"""
class Critic(nn.Module):
    def __init__(self, n_states, hidden_dim):
        super(Critic, self).__init__()
        # 定义价值网络结构：3层全连接网络，ReLU激活，输出单个价值
        self.critic = nn.Sequential(
            nn.Linear(n_states, hidden_dim),   # 输入层：状态维度 -> 隐藏层维度
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_dim, hidden_dim), # 隐藏层：隐藏层维度 -> 隐藏层维度
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_dim, 1)           # 输出层：隐藏层维度 -> 1（状态价值）
        )

    def forward(self, state):
        """前向传播：输入状态，返回状态价值"""
        value = self.critic(state)
        return value


"""PPO（Proximal Policy Optimization）算法核心类"""
class PPO:
    def __init__(self, n_states, n_actions, cfg):
        self.gamma = cfg['gamma']             # 折扣因子（未来奖励的衰减系数）
        self.continuous = cfg['continuous']   # 是否为连续动作空间（此处用于CartPole离散空间）
        self.policy_clip = cfg['policy_clip'] # PPO裁剪系数（通常为0.2）
        self.n_epochs = cfg['n_epochs']       # 每轮更新的迭代次数
        self.gae_lambda = cfg['gae_lambda']   # GAE（广义优势估计）的lambda系数
        self.device = cfg['device']           # 计算设备（CPU/
        # 初始化策略网络(actor)和价值网络(critic)，并移动到指定设备
        self.actor = Actor(n_states, n_actions, cfg['hidden_dim']).to(self.device)
        self.critic = Critic(n_states, cfg['hidden_dim']).to(self.device)
        # 初始化优化器（Adam优化器）
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg['critic_lr'])
        # 初始化经验缓冲区
        self.memory = PPOMemory(cfg['batch_size'])
        self.loss = 0 # 记录当前损失

    def choose_action(self, state):
        """根据当前状态选择动作，并返回动作、动作概率和状态价值"""
        state = np.array([state])  # 先转成数组再转tensor更高效
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()
        if self.continuous:
            action = torch.tanh(action)
        else:
            action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return action, probs, value

    def update(self):
        """PPO的核心更新逻辑：使用缓冲区数据更新策略网络和价值网络"""
        # 多轮迭代更新（n_epochs）
        for _ in range(self.n_epochs):
            # 从缓冲区采样数据
            state_arr, action_arr, old_prob_arr, vals_arr, \
                reward_arr, dones_arr, batches = self.memory.sample()
            values = vals_arr[:] # 状态价值列表
            ### 计算优势函数（Advantage）###
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr) - 1):
                discount = 1  # 折扣系数累计值
                a_t = 0       # 优势值
                # 计算从t时刻到序列末尾的优势
                for k in range(t, len(reward_arr) - 1):
                    # GAE公式：优势 = 即时奖励 + 折扣*下一状态价值*未终止标志 - 当前状态价值
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda    # 累计折扣（gamma*lambda）
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device) # 转为Tensor并移动到设备
            ###  stochastic gradient descent（SGD）更新 ###
            values = torch.tensor(values).to(self.device)       # 状态价值转为Tensor
            # 按批次更新
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device) # 状态
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device) # 旧策略的动作概率
                actions = torch.tensor(action_arr[batch]).to(self.device)     # 动作
                # 新策略的动作分布和价值估计
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value) # 压缩维度

                # -------------------------------------------------------------------------------#
                # -------------------------------------------------------------------------------#
                # ---------------------             3 填空           -----------------------------#
                # -------------------------------------------------------------------------------#
                # -------------------------------------------------------------------------------#
                # 计算新策略下各动作的对数概率
                new_probs = dist.log_prob(actions)
                # 计算新旧策略概率比
                ratio = torch.exp(new_probs - old_probs)
                # 对比例进行截断限制
                clipped_ratio = torch.clamp(ratio, 1 - self.policy_clip,
                                            1 + self.policy_clip)
                # 当前批次对应的优势
                batch_advantage = advantage[batch]
                # policy_loss（选择较低值进行优化，如果能将较小的值优化到令人满意的程度，那么对于其他的情况，模型的表现会更好）
                actor_loss = -torch.min(ratio * batch_advantage,
                                        clipped_ratio * batch_advantage).mean()
                # value_loss
                returns = batch_advantage + values[batch]
                value_loss = (returns - critic_value) ** 2
                value_loss = value_loss.mean()
                total_loss = actor_loss + 0.5 * value_loss
                self.loss = total_loss.item()
                # actor：反向传播与参数更新
                self.actor_optimizer.zero_grad()
                # critic：反向传播与参数更新
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
        # 更新完成后清空缓冲区
        self.memory.clear()

    def save_model(self, path):
        """保存模型参数到指定路径"""
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)   # 确保路径存在
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')   # 策略网络保存路径
        critic_checkpoint = os.path.join(path, 'ppo_critic.pt') # 价值网络保存路径
        torch.save(self.actor.state_dict(), actor_checkpoint)
        torch.save(self.critic.state_dict(), critic_checkpoint)

    def load_model(self, path):
        """从指定路径加载模型参数"""
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint = os.path.join(path, 'ppo_critic.pt')
        self.actor.load_state_dict(torch.load(actor_checkpoint))
        self.critic.load_state_dict(torch.load(critic_checkpoint))


# 训练函数
def train(arg_dict, env, agent):
    """
    训练智能体
    参数：
        arg_dict: 参数字典
        env: 环境对象
        agent: 智能体对象
    返回：
        训练过程的奖励记录
    """
    startTime = time.time() # 记录开始时间
    print(f"环境名: {arg_dict['env_name']}, 算法名: {arg_dict['algo_name']}, Device: {arg_dict['device']}")
    print("开始训练智能体......")
    rewards = []     # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    steps = 0        # 记录总步数
    # 循环训练回合
    for i_ep in range(arg_dict['train_eps']):
        state = env.reset()  # 重置环境，获取初始状态
        done = False         # 回合终止标志
        ep_reward = 0        # 记录当前回合的总奖励
        while not done:
            # 训练时是否渲染环境
            if arg_dict['train_render']:
                env.render()
            # -------------------------------------------------------------------------------#
            # -------------------------------------------------------------------------------#
            # ---------------------             2 填空           -----------------------------#
            # -------------------------------------------------------------------------------#
            # -------------------------------------------------------------------------------#
            # 智能体选择动作
            action, prob, val = agent.choose_action(state)
            # 与环境交互，获取下一个状态、奖励、终止标志
            state_, reward, done, _ = env.step(action)
            steps += 1
            ep_reward += reward
            # 将交互数据存入经验缓冲区
            agent.memory.push(state, action, prob, val, reward, done)
            # 每达到指定步数，更新一次策略网络
            if steps % arg_dict['update_fre'] == 0:
                agent.update()
            # 更新状态
            state = state_
        # 记录当前回合奖励
        rewards.append(ep_reward)
        # 计算滑动平均奖励（平滑系数0.9）
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        # 每10回合打印一次训练信息
        if (i_ep + 1) % 10 == 0:
            print(f"回合：{i_ep + 1}/{arg_dict['train_eps']}，奖励：{ep_reward:.2f}")
    print('训练结束 , 用时: ' + str(time.time() - startTime) + " s")
    # 关闭环境
    env.close()
    return {'episodes': range(len(rewards)), 'rewards': rewards}


# 测试函数
def test(arg_dict, env, agent):
    """
    测试已训练的智能体
    参数：
        arg_dict: 参数字典
        env: 环境对象
        agent: 智能体对象（已加载训练好的参数）
    返回：
        测试过程的奖励记录
    """
    startTime = time.time()
    print("开始测试智能体......")
    print(f"环境名: {arg_dict['env_name']}, 算法名: {arg_dict['algo_name']}, Device: {arg_dict['device']}")
    rewards = []     # 记录所有测试回合的奖励
    ma_rewards = []  # 记录所有测试回合的滑动平均奖励
    for i_ep in range(arg_dict['test_eps']):
        state = env.reset()  # 重置环境
        done = False         # 终止标志
        ep_reward = 0        # 当前回合奖励
        while not done:
            # 测试时是否渲染环境
            if arg_dict['test_render']:
                env.render()
            # 智能体选择动作（测试时不更新策略网络）
            action, prob, val = agent.choose_action(state)
            # 与环境交互
            state_, reward, done, _ = env.step(action)
            ep_reward += reward   # 累计奖励
            state = state_        # 更新状态
        # 记录测试奖励
        rewards.append(ep_reward)
        # 计算滑动平均奖励
        if ma_rewards:
            ma_rewards.append(
                0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print('回合：{}/{}, 奖励：{}'.format(i_ep + 1, arg_dict['test_eps'], ep_reward))
    print("测试结束 , 用时: " + str(time.time() - startTime) + " s")
    env.close()
    return {'episodes': range(len(rewards)), 'rewards': rewards}

# 为所有随机因素设置一个统一的种子
def all_seed(env, seed=520):
    # 环境种子设置
    env.seed(seed)
    # numpy随机数种子设置
    np.random.seed(seed)
    # python自带随机数种子设置
    random.seed(seed)
    # CPU种子设置
    torch.manual_seed(seed)
    # GPU种子设置
    torch.cuda.manual_seed(seed)
    # python scripts种子设置
    os.environ['PYTHONHASHSEED'] = str(seed)
    # cudnn的配置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def plot_rewards(rewards, cfg, tag='train'):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {cfg['device']} of {cfg['algo_name']} for {cfg['env_name']}")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.legend()
    plt.show()


# 创建环境和智能体
def create_env_agent(arg_dict):
    # 创建环境
    env = gym.make(arg_dict['env_name'])
    # 设置随机种子
    all_seed(env, seed=arg_dict["seed"])
    # 获取状态数
    try:
        n_states = env.observation_space.n
    except AttributeError:
        n_states = env.observation_space.shape[0]
    # 获取动作数
    n_actions = env.action_space.n
    print(f"状态数: {n_states}, 动作数: {n_actions}")
    # 将状态数和动作数加入算法参数字典
    arg_dict.update({"n_states": n_states, "n_actions": n_actions})
    # 实例化智能体对象
    agent = PPO(n_states, n_actions, arg_dict)
    # 返回环境，智能体
    return env, agent


if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 获取当前路径
    curr_path = os.path.dirname(os.path.abspath(__file__))
    # 获取当前时间
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    # 相关参数设置
    parser = argparse.ArgumentParser(description="hyper parameters")
    parser.add_argument('--algo_name', default='PPO', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='CartPole-v0', type=str, help="name of environment")
    parser.add_argument('--continuous', default=False, type=bool,
                        help="if PPO is continuous")  # PPO既可适用于连续动作空间，也可以适用于离散动作空间
    parser.add_argument('--train_eps', default=200, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--batch_size', default=5, type=int)  # mini-batch SGD中的批量大小
    parser.add_argument('--n_epochs', default=4, type=int)
    parser.add_argument('--actor_lr', default=0.0003, type=float, help="learning rate of actor net")
    parser.add_argument('--critic_lr', default=0.0003, type=float, help="learning rate of critic net")
    parser.add_argument('--gae_lambda', default=0.95, type=float)
    parser.add_argument('--policy_clip', default=0.2, type=float)  # PPO-clip中的clip参数，一般是0.1~0.2左右
    parser.add_argument('--update_fre', default=20, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--device', default='cpu', type=str, help="cpu or cuda")
    parser.add_argument('--seed', default=520, type=int, help="seed")
    parser.add_argument('--show_fig', default=False, type=bool, help="if show figure or not")
    parser.add_argument('--train_render', default=True, type=bool,
                        help="Whether to render the environment during training")
    parser.add_argument('--test_render', default=True, type=bool,
                        help="Whether to render the environment during testing")
    args = parser.parse_args()
    default_args = {'model_path': f"{curr_path}/"}
    # 将参数转化为字典 type(dict)
    arg_dict = {**vars(args), **default_args}
    print("算法参数字典:", arg_dict)

    # 创建环境和智能体
    env, agent = create_env_agent(arg_dict)
    # 传入算法参数、环境、智能体，然后开始训练
    res_dic = train(arg_dict, env, agent)
    print("算法返回结果字典:", res_dic)
    # 保存相关信息
    agent.save_model(path=arg_dict['model_path'])
    plot_rewards(res_dic['rewards'], arg_dict, tag="train")

    # 创建新环境和智能体用来测试
    print("=" * 300)
    env, agent = create_env_agent(arg_dict)
    # 加载已保存的智能体
    agent.load_model(path=arg_dict['model_path'])
    res_dic = test(arg_dict, env, agent)
    plot_rewards(res_dic['rewards'], arg_dict, tag="test")


