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
import sys
import platform
from multiprocessing import Process
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.http_utils import http_utils_request
from kaiwudrl.common.utils.common_func import (
    python_exec_shell,
    find_pids_by_cmdline,
    scan_for_errors,
)
import kaiwudrl.server.learner.learner as learner
import kaiwudrl.server.aisrv.aisrv as aisrv
from kaiwudrl.common.config.config_control import CONFIG
from typing import List

# To run the train_test, you must modify the algorithm name here.
# It must be one of dynamic_programming, monte_carlo, q_learning, sarsa or diy.
# Simply modify the value of the algorithm_name variable.
# 运行train_test前必须修改这里的算法名字, 必须是dynamic_programming、monte_carlo、q_learning、sarsa、diy里的一个, 修改algorithm_name的值即可
algorithm_name_list = [
    "dynamic_programming",
    "monte_carlo",
    "q_learning",
    "sarsa",
    "diy",
]
algorithm_name = "dynamic_programming"


# train
# 训练
def train():

    start_time = time.time()

    # To modify the value in the environment variable and initiate training for the learner as soon as possible
    # 修改环境变量里的值, 尽快让learner进行训练
    os.environ.update(
        {
            "use_ckpt_sync": "False",
            "replay_buffer_capacity": "10",
            "preload_ratio": "10",
            "train_batch_size": "2",
            "use_prometheus": "True",
            "dump_model_freq": "1",
        }
    )

    # To modify the configuration and execute the script directly
    # 修改配置, 调用脚本直接执行
    if algorithm_name not in algorithm_name_list:
        print("\033[92m" + f"algorithm_name: {algorithm_name} not in list {algorithm_name_list}" + "\033[0m")

        python_exec_shell(f"sh tools/stop.sh all")

    python_exec_shell(f"sh /root/tools/change_algorithm_all.sh {algorithm_name}")
    print(f"current algorithm_name is {algorithm_name}")

    # Setting the sample transmission to either Reverb or ZMQ based on the operating system
    # 根据不同的操作系统设置发送样本的是reverb还是zmq
    architecture = platform.machine()
    platform_maps = {
        "aarch64": "zmq",
        "arm64": "zmq",
        "x86_64": "reverb",
        "AMD64": "reverb",
    }
    sample_tool_type = platform_maps.get(architecture)
    if sample_tool_type is None:
        print(f"Architecture '{architecture}' may not exist or not be supported.")
    else:
        result_code, result_str = python_exec_shell(f"sh tools/change_sample_server.sh {sample_tool_type}")
        if result_code != 0:
            raise ValueError(f"Execution error! Please check the error detail: {result_str}")

    CONFIG.set_configure_file("conf/kaiwudrl/learner.toml")
    CONFIG.parse_learner_configure()

    # To delete previous model files and logs
    # 删除以前的model文件和日志
    python_exec_shell(f"rm -rf {CONFIG.user_ckpt_dir}/{CONFIG.app}_{algorithm_name}/*")
    python_exec_shell(f"rm -rf {CONFIG.log_dir}/*")
    done_file = f"/data/ckpt/{CONFIG.app}_{CONFIG.algo}/process_stop.done"
    if os.path.exists(done_file):
        os.remove(done_file)

    model_base = f"/data/ckpt/{CONFIG.app}_{CONFIG.algo}/model.ckpt-1"
    possible_extensions = [".npy", ".pkl"]
    for ext in possible_extensions:
        model_file = model_base + ext
        if os.path.exists(model_file):
            os.remove(model_file)

    # To start aisrv, learner, actor, and battlesrv
    # 启动aisrv, learner, actor, battlesrv
    procs: List[Process] = []
    procs.append(Process(target=learner.main, name="learner"))
    procs.append(Process(target=aisrv.main, name="aisrv"))
    python_exec_shell("sh tools/modelpool_start.sh learner")

    for proc in procs:
        proc.start()
        time.sleep(10)
        check(proc)

    while True:

        error_msg = scan_for_errors(CONFIG.log_dir, error_indicator="ERROR")
        train_success = check_train_success_by_model_file()
        process_done = check_process_stop_done()

        # If an error log is generated, exit early
        # 如果有错误日志产生, 提前退出
        if error_msg:
            time.sleep(5)
            print("\033[1;31m" + "find error log, please check" + "\033[0m")
            python_exec_shell(f"sh tools/stop.sh all")
        else:
            if train_success or process_done:

                time.sleep(5)
                print(
                    "\033[1;32m"
                    + f"Train test succeed, will exit, cost {time.time() - start_time:.2f} seconds "
                    + "\033[0m"
                )

                python_exec_shell(f"sh tools/stop.sh all")

        time.sleep(1)
        for proc in procs:
            check(proc)


# If the file 'process_stop.done' is created, exit prematurely
# 如果有process_stop.done的文件生成, 提前退出
def check_process_stop_done():
    done_file = f"/data/ckpt/{CONFIG.app}_{CONFIG.algo}/process_stop.done"
    if os.path.exists(done_file):
        return True

    return False


# To check if a process is alive, any error log
# 检测进程是否存活, 是否有错误日志
def check(proc: Process):
    if not proc.is_alive():
        time.sleep(5)
        print("\033[1;31m" + f"{proc.name} is not alive, please check error log" + "\033[0m")
        python_exec_shell(f"sh tools/stop.sh all")


# "Based on the landed model file, if the serialization equals 1,
# it indicates that the learner has trained for one step and the model file has been correctly landed."
# 按照落地的model文件来做判断, 如果序列化等于1了, 则代表learner训练1步并且正确落地model文件
def check_train_success_by_model_file():

    model_base = f"/data/ckpt/{CONFIG.app}_{CONFIG.algo}/model.ckpt-1"
    possible_extensions = [".npy", ".pkl"]
    for ext in possible_extensions:
        model_file = model_base + ext
        if os.path.exists(model_file):
            os.remove(model_file)
            return True


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\033[1;31m" + "KeyboardInterrupt, please check" + "\033[0m")
        python_exec_shell(f"sh tools/stop.sh all")
