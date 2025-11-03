# -*- coding: utf-8 -*-
import os
from atari_wrappers import wrap_deepmind, make_atari
import random
from tqdm import tqdm
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

BATCH_SIZE = 16
GAMMA = 0.99
EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 100000
TARGET_UPDATE = 1000
NUM_STEPS = 6e6
M_SIZE = 100000
POLICY_UPDATE = 4
EVALUATE_FREQ = 5e6
steps_done = 0

class DQN(nn.Module):
    def __init__(self, h, w, outputs, device):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, outputs)
        self.device = device

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)

        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = x.to(self.device).float() / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)

class ReplayMemory(object):
    def __init__(self, capacity, state_shape, n_actions, device):
        c,h,w = state_shape
        self.capacity = capacity
        self.device = device
        self.m_states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, done):
        self.m_states[self.position] = state
        self.m_actions[self.position,0] = action
        self.m_rewards[self.position,0] = reward
        self.m_dones[self.position,0] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def sample(self, bs):
        i = torch.randint(0, high=self.size, size=(bs,))
        bs = self.m_states[i, :4]
        bns = self.m_states[i, 1:]
        ba = self.m_actions[i].to(self.device)
        br = self.m_rewards[i].to(self.device).float()
        bd = self.m_dones[i].to(self.device).float()
        return bs, ba, br, bns, bd

    def __len__(self):
        return self.size

class ActionSelector(object):
    def __init__(self, INITIAL_EPSILON, FINAL_EPSILON, policy_net, EPS_DECAY, n_actions, device):
        self._eps = INITIAL_EPSILON
        self._FINAL_EPSILON = FINAL_EPSILON
        self._INITIAL_EPSILON = INITIAL_EPSILON
        self._policy_net = policy_net
        self._EPS_DECAY = EPS_DECAY
        self._n_actions = n_actions
        self._device = device

    def select_action(self, state, training=False):
        sample = random.random()
        if training:
            self._eps -= (self._INITIAL_EPSILON - self._FINAL_EPSILON) / self._EPS_DECAY
            self._eps = max(self._eps, self._FINAL_EPSILON)
        if sample > self._eps:
            with torch.no_grad():
                a = self._policy_net(state).max(1)[1].cpu().view(1, 1)
        else:
            a = torch.tensor([[random.randrange(self._n_actions)]], device='cpu', dtype=torch.long)

        return a.numpy()[0, 0].item()

def main(test=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name="SpaceInvaders"
    env_raw = make_atari('{}NoFrameskip-v4'.format(env_name))
    env = wrap_deepmind(env_raw, frame_stack=False, episode_life=True, clip_rewards=True)
    n_frame = torch.from_numpy(env.reset())
    h = n_frame.shape[-2];w=h
    n_actions = env.action_space.n

    policy_net = DQN(h, w, n_actions, device).to(device)
    target_net = DQN(h, w, n_actions, device).to(device)
    q = deque(maxlen=5)
    done = True
    episode_len = 0

    if test:
        policy_net.load_state_dict(torch.load(os.getcwd() + "/dqn_atari.pth",map_location=torch.device('cpu')))
        NUM_STEPS = 10
    else:
        NUM_STEPS = 25000000
        policy_net.apply(policy_net.init_weights)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0000625, eps=1.5e-4)
    memory = ReplayMemory(M_SIZE, [5, h, w], n_actions, device)
    my_action = ActionSelector(EPS_START, EPS_END, policy_net, EPS_DECAY, n_actions, device)

    def optimize_model(train):
        if not train:
            return
        state_batch, action_batch, reward_batch, n_state_batch, done_batch = memory.sample(BATCH_SIZE)

        q = policy_net(state_batch).gather(1, action_batch)
        nq = target_net(n_state_batch).max(1)[0].detach()

        expected_state_action_values = (nq * GAMMA) * (1. - done_batch[:, 0]) + reward_batch[:, 0]

        loss = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    def evaluate(step, policy_net, device, env, n_actions, eps=0.05, num_episode=5, test=False):
        env = wrap_deepmind(env)
        my_action = ActionSelector(eps, eps, policy_net, EPS_DECAY, n_actions, device)
        e_rewards = []
        q = deque(maxlen=5)
        for i in range(num_episode):
            env.reset()
            e_reward = 0
            for _ in range(10):
                n_frame, _, done, _ = env.step(0)
                n_frame = torch.from_numpy(n_frame).view(1,h,h)
                q.append(n_frame)
            while not done:
                env.render()
                time.sleep(0.01)
                state = torch.cat(list(q))[1:].unsqueeze(0)
                action = my_action.select_action(state, train)
                n_frame, reward, done, info = env.step(action)
                n_frame = torch.from_numpy(n_frame).view(1,h,h)
                q.append(n_frame)
                e_reward += reward
            e_rewards.append(e_reward)
        if test:
            print("SpaceInvaders-Test num: {} , rewards: {}".format(step + 1, (float(sum(e_rewards)) / float(num_episode))))


    progressive = tqdm(range(NUM_STEPS), total=NUM_STEPS, ncols=50, leave=False, unit='b')
    for step in progressive:
        if done:
            env.reset()
            episode_len = 0
            img, _, _, _ = env.step(1)
            for i in range(10):
                n_frame, _, _, _ = env.step(0)
                n_frame = torch.from_numpy(n_frame).view(1,h,h)
                q.append(n_frame)

        train = len(memory) > 50000
        state = torch.cat(list(q))[1:].unsqueeze(0)
        action = my_action.select_action(state, train)
        n_frame, reward, done, info = env.step(action)
        n_frame = torch.from_numpy(n_frame).view(1,h,h)

        q.append(n_frame)
        memory.push(torch.cat(list(q)).unsqueeze(0), action, reward,
                    done)
        episode_len += 1

        if test:
            evaluate(step, policy_net, device, env_raw, n_actions, eps=0.01, num_episode=1, test=True)

        if step % POLICY_UPDATE == 0 and not test:
            optimize_model(train)

        if step % TARGET_UPDATE == 0 and not test:
            target_net.load_state_dict(policy_net.state_dict())

        if step % EVALUATE_FREQ == 0 and not test and step!=0:
            torch.save(policy_net.state_dict(), os.getcwd() + "/dqn_atari.pth")
    env.close()

if __name__ == "__main__":
    #仅用于测试模型
    main(test=True)
