from Common.logger import Logger
from Common.play import Play
from Brain.agent import Agent
from Common.utils import *
from Common.config import get_params
import time

if __name__ == '__main__':
    params = get_params()
    test_env = make_atari(params["env_name"])
    params.update({"n_actions": test_env.action_space.n})
    #创建测试环境
    env = make_atari(params["env_name"])
    env.seed(int(time.time()))
    #初始化智能体
    agent = Agent(**params)
    logger = Logger(agent, **params)
    #加载已经训练好的模型
    chekpoint = logger.load_weights()
    player = Play(agent, env, chekpoint["online_model_state_dict"], **params)
    #开始测试模型
    player.evaluate()

