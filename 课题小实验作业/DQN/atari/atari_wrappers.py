from collections import deque

import cv2
import gym
import numpy as np
from gym import spaces

cv2.ocl.setUseOpenCL(False)

#Gym Atari环境预处理Wrapper

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        #该wrapper的作用是在reset环境的时候，使用随机数量的no-op动作（假设其为环境的动作0）来采样初始化状态，如果在中途环境已经返回done了，则重新reset环境。这有利于增加初始画面的随机性，减小陷入过拟合的几率。
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        # 在[1，noop_max]中执行一定数量的无操作动作。
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        #在一些Atari游戏中，有开火键，比如Space Invaders，该wrapper的作用是返回一个选择开火动作后不done的环境状态。
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        #在许多游戏中，玩家操纵的角色有不止一条命，为了加速Agent的训练，使其尽量避免死亡，将每条命死亡后的done设为True，同时使用一个属性self.was_real_done来标记所有生命都用完之后的真正done。
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:

            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        #仅在生命耗尽时重置，这样一来，即使生命是周期性的，所有状态仍然可以到达
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    #该Wrapper提供了跳帧操作，即每skip帧返回一次环境状态元组，在跳过的帧里执行相同的动作，将其奖励叠加，并且取最后两帧像素值中的最大值。在Atari游戏中，有些画面是仅在奇数帧出现的，因此要对最后两帧取最大值。
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    #对于不同游戏来说，其得分衡量也是不同的，为了便于统一度量和学习，将所有奖励统一定义为1（reward > 0），0（reward = 0）或-1（reward < 0）。
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    #该Wrap对观察到的帧的图片数据进行了处理。首先将3维RGB图像转为灰度图像，之后将其resize为84 × 84的灰度图像。为使Agent更关注于游戏本身的画面，避免被得分等图像区域误导，故对画面进行裁切，对于不同的游戏，裁切的方法可能不同。
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        #本wrapper的作用是将k帧灰度图像并为一帧，以此来为CNN提供一些序列信息（Human Level control through deep reinforcement learning）。Wrapper会维持一个大小为k的deque，之后依次使用最新的ob来替代最久远的ob，达到不同时间的状态叠加的效果。最后返回一个LazyFrame。如果想要使用LazyFrame，只需利用np.array(lazy_frames_instance)即可将LazyFrame对象转为ndarray对象。
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    #该Wrapper的目的是将0 ~ 255的图像归一化到 [0, 1]。
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
    #该Wrapper确保观测之间的帧画面仅存储一次。纯粹是为了优化内存使用而存在。
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

#创建环境并对环境进行预处理
def make_atari(env_id, max_episode_steps=400000):
    env = gym.make(env_id)
    env._max_episode_steps = max_episode_steps
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=1)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=False, scale=False):
    #为DeepMind风格的Atari游戏配置环境
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

