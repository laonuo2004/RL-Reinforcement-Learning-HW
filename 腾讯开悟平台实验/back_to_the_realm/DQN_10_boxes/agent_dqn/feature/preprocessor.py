#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################

"""
Author: Tencent AI Arena Authors

"""


from collections import defaultdict, deque
import itertools
import math

import numpy as np
from arena_proto.back_to_the_realm.custom_pb2 import (
    FloatPosition,
    Position,
    RealmFeature,
    RelativeDirection,
    RelativeDistance,
    RelativePosition,
)
from agent_target_dqn.conf.conf import Config


def norm(pos):
    """
    Coordinate normalization
    坐标归一化

    Args:
        a (int): x
        b (int): z

    Returns:
        _type_: FloatPosition
        类型: FloatPosition
    """
    float_pos = FloatPosition(
        x=pos.x / 64000,
        z=pos.z / 64000,
    )
    return float_pos


def polar_norm(pos):
    """
    Convert polar coordinates and normalize
    转换极坐标并归一化

    Args:
        a (int): x
        b (int): z

    Returns:
        _type_: FloatPosition
        类型: FloatPosition
    """
    r = math.hypot(pos.x, pos.z) / (64000 * math.sqrt(2))
    theta = math.atan2(pos.z, pos.x)
    if theta < 0:
        theta = 2 * math.pi + theta
    theta = theta / (math.pi * 2)

    float_pos = FloatPosition(x=r, z=theta)
    return float_pos


def ln_distance(a1, b1, a2, b2, n):
    """
    Calculate l_n distance
    计算 l_n距离

    Args:
        a1 (float): x1
        b1 (float): z1
        a2 (float): x2
        b2 (float): z2
        n (int): n

    Returns:
        _type_: ln distance
        类型: ln距离
    """
    a = abs(a1 - a2)
    b = abs(b1 - b2)

    return (a**n + b**n) ** (1 / n)


def get_direction(pos_1, pos_2):
    """
    Calculate the bearing between two points
    输入两个点计算方位

    Args:
        x1 (int): x1
        z1 (int): z1
        x2 (int): x2
        z2 (int): z2

    Returns:
        int: Direction encoding, refer to the protocol
        int: 方向编码，参考协议
    """
    x1, z1 = pos_1.x, pos_1.z
    x2, z2 = pos_2.x, pos_2.z

    x = x2 - x1
    z = z2 - z1

    # Calculate the angle
    # 计算角度
    theta = math.atan2(z, x)
    if theta < 0:
        # range of atan2 is -pi to pi
        # atan2 的范围是 -pi 到 pi
        theta = 2 * math.pi + theta

    # Encode the direction based on the angle; refer to the protocol for the encoding method
    # 根据角度对方向进行编码，编码方式见协议
    r_direction = round(theta / (math.pi / 4)) + 1
    if r_direction == 9:
        r_direction = 1
    return r_direction


def bfs_from_center_to_goal(map, goal):
    """
    Perform BFS from the center of the map to the goal.
    从地图中心出发到达目标点的BFS。

    Args:
        map (np.array): 地图, 0表示不可通过, 1表示可通过。The map, 0 for non-passable, 1 for passable.
        goal (tuple): 目标位置。The goal position.

    Returns:
        int: 从中心点到目标位置的距离, -1表示不可到达。The distance from the center to the goal position.
    """
    if not goal:
        return -1

    center_x = map.shape[0] // 2
    center_y = map.shape[1] // 2
    start = (center_x, center_y)

    out = np.full((map.shape[0], map.shape[1]), -1)
    visited = {start}
    queue = deque([(start, 0)])

    while queue:
        loc, dist = queue.popleft()
        out[loc] = dist

        if loc == goal:
            return dist

        x, y = loc
        for dx, dy in [(-1, 0), (-1, -1), (1, 0), (1, -1), (0, -1), (-1, 1), (0, 1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < map.shape[0] and 0 <= ny < map.shape[1] and map[nx][ny] == 1 and (nx, ny) not in visited:
                queue.append(((nx, ny), dist + 1))
                visited.add((nx, ny))

    return -1


def get_grid_relative_pos_info(hero_grid_pos, other_grid_pos, grid):
    """
    Get the direction and distance information of the organ relative to the current position
    获得 organ相对当前位置的方向和距离信息

    Args:
        hero_grid_pos: 英雄网格位置 Hero Grid Position
        other_grid_pos: 目标网格位置 other Grid Position

    Returns:
        _type_: RelativePosition
        类型: RelativePosition
    """

    rel_pos = RelativePosition()
    rel_pos.direction = get_direction(hero_grid_pos, other_grid_pos)
    rel_pos.l2_distance = ln_distance(hero_grid_pos.x, hero_grid_pos.z, other_grid_pos.x, other_grid_pos.z, 2)
    rel_pos.path_distance = 0
    rel_pos.grid_distance = bfs_from_center_to_goal(
        grid, get_relative_grid_pos(hero_grid_pos, other_grid_pos, Config.VIEW_SIZE)
    )

    return rel_pos


def get_relative_grid_pos(hero_grid_pos, target_grid_pos, n=50):
    # Calculate the relative position of target_grid_pos on a local grid centered at hero_grid_pos.
    # 计算 target_grid_pos 在以 hero_grid_pos 为中心的局部网格上的相对坐标。
    hero_x, hero_y = hero_grid_pos.x, hero_grid_pos.z
    target_x, target_y = target_grid_pos.x, target_grid_pos.z

    # Calculate the boundaries of the local grid
    # 计算局部网格的边界
    min_x = hero_x - n
    max_x = hero_x + n
    min_y = hero_y - n
    max_y = hero_y + n

    # Check if the target point is within the local grid
    # 检查目标点是否在局部网格内
    if min_x <= target_x <= max_x and min_y <= target_y <= max_y:
        # Calculate the relative coordinates
        # 计算相对坐标
        relative_x = target_x - min_x
        relative_y = target_y - min_y
        return (relative_x, relative_y)
    else:
        # The target point is not within the local grid
        # 目标点不在局部网格内
        return None


def get_null_relative_pos():
    """
    Returns an empty RelativePosition, which is called when the treasure chest does not exist or is acquired.
    返还空的 RelativePosition, 当宝箱不存在或被获取后调用

    Returns:
        _type_: RelativePosition
        类型: RelativePosition
    """
    rel_pos = RelativePosition()
    rel_pos.direction = RelativeDirection.RELATIVE_DIRECTION_NONE
    rel_pos.l2_distance = RelativeDistance.VeryLarge
    rel_pos.path_distance = RelativeDistance.VeryLarge
    rel_pos.grid_distance = -1
    return rel_pos


def convert_pos_to_grid_pos(x, z):
    """
    Convert pos to grid coordinates
    将pos转换为珊格化后坐标

    Args:
        x (float): x
        z (float): z

    Returns:
        _type_: tuple
        类型: 元组
    """

    x = (x + 2250) // 500
    z = (z + 5250) // 500

    # This step is necessary in order to be aligned with the order of json files
    # 这一步是必要的，用于与 json 文件的顺序保持一致
    x, z = z, x

    return (x, z)


def get_grid_pos(x, z):
    """
    Convert pos to grid coordinates
    将pos转换为珊格化后坐标

    Args:
        x (float): x
        z (float): z

    Returns:
        _type_: Position
        类型: Position
    """

    x = (x + 2250) // 500
    z = (z + 5250) // 500

    # This step is necessary in order to be aligned with the order of json files
    # 这一步是必要的，用于与 json 文件的顺序保持一致
    x, z = z, x

    return Position(x=x, z=z)


def get_legal_act(talent_status):
    return [1, 1] if bool(talent_status) else [1, 0]


def init_memory_map():
    return np.zeros((128, 128))


class Preprocessor:
    def __init__(self):

        # The map is rasterized into 256 * 256 grids, creating a memory map of the corresponding size
        # 地图被栅格化为128 * 128个网格, 创建对应大小的记忆地图
        self.memory_map = init_memory_map()
        # self.memory_map = np.zeros((256, 256))

        self.recent_position_map = defaultdict(int)
        self.arrival_position_map = defaultdict(int)
        self.recent_position_max = 100
        self.recent_positions = deque(maxlen=self.recent_position_max)
        self.last_pos = None

    def update_position(self, hero_grid_pos):
        # If the queue is full, reduce the count of the oldest position
        # 如果队列满了，减少最旧位置的计数
        if len(self.recent_positions) == self.recent_position_max:
            oldest_pos = self.recent_positions.popleft()
            self.recent_position_map[oldest_pos] -= 1
            if self.recent_position_map[oldest_pos] == 0:
                del self.recent_position_map[oldest_pos]

        # Add the new position to the queue and dictionary
        # 将新位置添加到队列和字典中
        pos = (hero_grid_pos.x, hero_grid_pos.z)
        self.recent_positions.append(pos)
        self.recent_position_map[pos] += 1
        self.arrival_position_map[pos] += 1

    def process(self, state_env_info):

        # grid = np.array(map_data["Flags"]).reshape(map_data["Height"], map_data["Width"])

        # Get the map in the local view
        # 获取局部视野下的地图
        grid = []
        for map_info in state_env_info.map_info:
            grid.append(map_info.values)
        grid = np.array(grid)

        hero_pos = state_env_info.frame_state.heroes[0].pos
        hero_norm_pos = norm(hero_pos)
        hero_grid_pos = get_grid_pos(hero_pos.x, hero_pos.z)
        self.update_position(get_grid_pos(hero_pos.x, hero_pos.z))

        buff_pos = get_null_relative_pos()

        start_grid_pos = get_grid_pos(state_env_info.game_info.start_pos.x, state_env_info.game_info.start_pos.z)
        end_grid_pos = get_grid_pos(state_env_info.game_info.end_pos.x, state_env_info.game_info.end_pos.z)
        start_pos = get_grid_relative_pos_info(hero_grid_pos, start_grid_pos, grid)
        end_pos = get_grid_relative_pos_info(hero_grid_pos, end_grid_pos, grid)
        treasure_collected_count = state_env_info.game_info.treasure_collected_count
        treasure_count = state_env_info.game_info.treasure_count

        treasure_pos = [get_null_relative_pos()] * 15
        treasure_grids = set()

        organs = state_env_info.frame_state.organs
        for organ in organs:
            if organ.sub_type == 2:
                # Buff
                # 增益效果
                if organ.status == 1:
                    # Desirable
                    # 可取
                    buff_pos = get_grid_relative_pos_info(hero_grid_pos, get_grid_pos(organ.pos.x, organ.pos.z), grid)
                else:
                    # Undesirable
                    # 不可取
                    buff_pos = get_null_relative_pos()

            elif organ.sub_type == 1:
                # treasure
                # 宝箱
                if organ.status == 1:
                    # Desirable
                    # 可取
                    organ_grid_pos = get_grid_pos(organ.pos.x, organ.pos.z)
                    treasure_pos[organ.config_id - 1] = get_grid_relative_pos_info(hero_grid_pos, organ_grid_pos, grid)
                    treasure_grids.add((organ_grid_pos.x, organ_grid_pos.z))
                    # print(organ.config_id)

        grid_pos_x, grid_pos_z = hero_grid_pos.x, hero_grid_pos.z

        view = Config.VIEW_SIZE

        obstacle_map = np.zeros((view * 2 + 1, view * 2 + 1))
        local_memory_map = np.zeros((view * 2 + 1, view * 2 + 1))
        treasure_map = np.zeros((view * 2 + 1, view * 2 + 1))
        end_map = np.zeros((view * 2 + 1, view * 2 + 1))

        # Dynamically update the local_memory_map, end_map, treasure_map, obstacle_map features, and pad the map borders with 0
        # 动态更新 local_memory_map, end_map, treasure_map, obstacle_map 四个特征，地图边界处用0进行padding
        for i, j in itertools.product(range(-view, view + 1), range(-view, view + 1)):
            local_i = view + i
            local_j = view + j

            if 0 <= local_i < len(grid) and 0 <= local_j < len(grid[0]):
                obstacle_map[local_i][local_j] = grid[local_i][local_j]
                if 0 <= grid_pos_x + i < self.memory_map.shape[0] and 0 <= grid_pos_z + j < self.memory_map.shape[1]:
                    local_memory_map[local_i][local_j] = self.memory_map[grid_pos_x + i, grid_pos_z + j]
                if (grid_pos_x + i, grid_pos_z + j) in treasure_grids:
                    treasure_map[local_i][local_j] = 1
                if (grid_pos_x + i, grid_pos_z + j) == (end_grid_pos.x, end_grid_pos.z):
                    end_map[local_i][local_j] = 1

        # Memory map needs to be updated after local_memory_map is initialized
        # Memory map 需要在local_memory_map初始化后更新
        self.memory_map[grid_pos_x, grid_pos_z] = min(1, 0.2 + self.memory_map[grid_pos_x, grid_pos_z])

        return (
            hero_norm_pos,
            hero_grid_pos,
            start_pos,
            end_pos,
            buff_pos,
            treasure_pos,
            obstacle_map.flatten().astype(int).tolist(),
            local_memory_map.flatten().tolist(),
            treasure_map.flatten().astype(int).tolist(),
            end_map.flatten().astype(int).tolist(),
            self.recent_position_map.copy(),
            treasure_collected_count,
            treasure_count,
        )
