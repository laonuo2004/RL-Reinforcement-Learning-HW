#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
计算峡谷漫步的理论最高分

将问题建模为图论问题：
- 节点：起点、终点、10个宝箱（共12个节点）
- 边：节点间的最短路径距离
- 目标：找到从起点出发，经过所有宝箱，到达终点的最短路径
"""

import json
import sys
from collections import deque
from itertools import permutations
import time

# 配置信息
START_POS = [29, 9]
END_POS = [11, 55]
MAX_STEP = 1999

# 宝箱位置（从配置文件读取）
TREASURE_POSITIONS = [
    1230,  # 宝箱0: [19, 14]
    604,   # 宝箱1: [9, 28]
    620,   # 宝箱2: [9, 44]
    2733,  # 宝箱3: [42, 45]
    2071,  # 宝箱4: [32, 23]
    3192,  # 宝箱5: [49, 56]
    2298,  # 宝箱6: [35, 58]
    1527,  # 宝箱7: [23, 55]
    2657,  # 宝箱8: [41, 33]
    3497   # 宝箱9: [54, 41]
]

# 动作定义
ACTIONS = {
    0: "UP",    # 上
    1: "DOWN",  # 下
    2: "LEFT",  # 左
    3: "RIGHT"  # 右
}


def pos_to_id(x, z):
    """将坐标转换为状态ID"""
    return x * 64 + z


def id_to_pos(pos_id):
    """将状态ID转换为坐标"""
    x = pos_id // 64
    z = pos_id % 64
    return [x, z]


def load_map_data(map_file):
    """加载地图数据"""
    with open(map_file, 'r') as f:
        return json.load(f)


def build_transition_graph(F):
    """
    从地图数据构建状态转移图
    
    Returns:
        graph: dict, {pos_id: {action: next_pos_id}}
    """
    graph = {}
    for pos_str, actions in F.items():
        pos_id = int(pos_str)
        graph[pos_id] = {}
        for action_str, transition in actions.items():
            action = int(action_str)
            next_pos_id, reward, done = transition
            graph[pos_id][action] = next_pos_id
    return graph


def bfs_shortest_path(graph, start_pos_id, end_pos_id):
    """
    使用BFS计算两个位置之间的最短路径长度
    
    Args:
        graph: 状态转移图
        start_pos_id: 起点位置ID
        end_pos_id: 终点位置ID
    
    Returns:
        distance: 最短路径长度，如果不可达返回None
        path: 最短路径（位置ID列表），如果不可达返回None
    """
    if start_pos_id == end_pos_id:
        return 0, [start_pos_id]
    
    if start_pos_id not in graph:
        return None, None
    
    # BFS
    queue = deque([(start_pos_id, 0, [start_pos_id])])
    visited = {start_pos_id}
    
    while queue:
        current_pos, distance, path = queue.popleft()
        
        if current_pos not in graph:
            continue
        
        for action, next_pos in graph[current_pos].items():
            if next_pos == end_pos_id:
                return distance + 1, path + [next_pos]
            
            if next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, distance + 1, path + [next_pos]))
    
    return None, None


def calculate_all_pair_distances(graph, nodes):
    """
    计算所有节点对之间的最短路径距离
    
    Args:
        graph: 状态转移图
        nodes: 节点列表（位置ID列表）
    
    Returns:
        distances: dict, {(node1, node2): distance}
    """
    distances = {}
    n = len(nodes)
    
    print(f"计算 {n} 个节点之间的最短路径...")
    start_time = time.time()
    
    for i in range(n):
        for j in range(i + 1, n):
            node1, node2 = nodes[i], nodes[j]
            distance, path = bfs_shortest_path(graph, node1, node2)
            
            if distance is not None:
                distances[(node1, node2)] = distance
                distances[(node2, node1)] = distance  # 无向图
            else:
                print(f"警告：节点 {node1} 和 {node2} 之间不可达！")
                distances[(node1, node2)] = float('inf')
                distances[(node2, node1)] = float('inf')
        
        if (i + 1) % 3 == 0:
            elapsed = time.time() - start_time
            print(f"  进度: {i+1}/{n} ({100*(i+1)/n:.1f}%), 耗时: {elapsed:.2f}s")
    
    elapsed = time.time() - start_time
    print(f"完成！总耗时: {elapsed:.2f}s")
    
    return distances


def solve_tsp_with_fixed_ends(distances, start_node, end_node, treasure_nodes):
    """
    解决TSP问题（固定起点和终点）
    
    问题：从起点出发，经过所有宝箱，到达终点，使得总距离最短
    
    Args:
        distances: 节点间距离字典
        start_node: 起点节点
        end_node: 终点节点
        treasure_nodes: 宝箱节点列表
    
    Returns:
        best_path: 最优路径（节点列表）
        best_distance: 最优路径长度
    """
    print(f"\n求解TSP问题（固定起点和终点）...")
    print(f"起点: {start_node}, 终点: {end_node}")
    print(f"宝箱数量: {len(treasure_nodes)}")
    print(f"需要尝试的排列数: {len(treasure_nodes)}! = {len(list(permutations(treasure_nodes)))}")
    
    best_distance = float('inf')
    best_path = None
    
    # 遍历所有宝箱的排列
    start_time = time.time()
    total_permutations = 1
    for i in range(1, len(treasure_nodes) + 1):
        total_permutations *= i
    
    count = 0
    for treasure_order in permutations(treasure_nodes):
        count += 1
        
        # 计算路径：起点 -> 宝箱1 -> 宝箱2 -> ... -> 宝箱10 -> 终点
        path = [start_node] + list(treasure_order) + [end_node]
        
        # 计算总距离
        total_distance = 0
        for i in range(len(path) - 1):
            node1, node2 = path[i], path[i + 1]
            dist = distances.get((node1, node2), float('inf'))
            if dist == float('inf'):
                total_distance = float('inf')
                break
            total_distance += dist
        
        if total_distance < best_distance:
            best_distance = total_distance
            best_path = path.copy()  # 确保是副本，避免引用问题
        
        # 进度显示
        if count % 100000 == 0 or count == total_permutations:
            elapsed = time.time() - start_time
            progress = 100 * count / total_permutations
            print(f"  进度: {count}/{total_permutations} ({progress:.1f}%), "
                  f"当前最优: {best_distance}步, 耗时: {elapsed:.2f}s")
    
    elapsed = time.time() - start_time
    print(f"完成！总耗时: {elapsed:.2f}s")
    
    # 验证最优路径包含所有宝箱
    if best_path is not None:
        path_treasures = [node for node in best_path[1:-1] if node in treasure_nodes]
        if len(path_treasures) != len(treasure_nodes):
            print(f"警告：最优路径只包含 {len(path_treasures)} 个宝箱，应该包含 {len(treasure_nodes)} 个！")
            print(f"路径中的宝箱: {path_treasures}")
            print(f"应该包含的宝箱: {treasure_nodes}")
        else:
            print(f"✓ 最优路径包含所有 {len(treasure_nodes)} 个宝箱")
    
    return best_path, best_distance


def calculate_score(total_steps, max_step=1999):
    """
    计算总积分
    
    总积分 = 终点积分 + 步数积分 + 宝箱积分
    
    - 终点积分：150
    - 步数积分：(最大步数 - 完成步数) × 0.2
    - 宝箱积分：10 × 100 = 1000
    """
    end_score = 150
    step_score = (max_step - total_steps) * 0.2
    treasure_score = 10 * 100
    
    total_score = end_score + step_score + treasure_score
    
    return total_score, end_score, step_score, treasure_score


def print_path_details(path, distances):
    """打印路径详情"""
    print("\n" + "=" * 60)
    print("最优路径详情")
    print("=" * 60)
    
    total_distance = 0
    
    # 打印起点
    start_node = path[0]
    pos = id_to_pos(start_node)
    print(f"起点: [{pos[0]}, {pos[1]}] (ID: {start_node})")
    
    # 打印中间所有节点（宝箱）
    for i in range(len(path) - 1):
        node1, node2 = path[i], path[i + 1]
        dist = distances.get((node1, node2), float('inf'))
        total_distance += dist
        
        pos2 = id_to_pos(node2)
        
        # 判断node2是宝箱还是终点
        if node2 in TREASURE_POSITIONS:
            treasure_id = TREASURE_POSITIONS.index(node2)
            print(f"  -> 宝箱{treasure_id}: [{pos2[0]}, {pos2[1]}] (ID: {node2}), 距离: {dist}步")
        else:
            # 是终点
            print(f"  -> 终点: [{pos2[0]}, {pos2[1]}] (ID: {node2}), 距离: {dist}步")
    
    print(f"\n总距离: {total_distance}步")
    
    # 验证路径完整性
    print(f"\n路径验证:")
    print(f"  路径长度: {len(path)} (应该是12: 起点+10个宝箱+终点)")
    print(f"  路径节点ID: {path}")
    
    # 检查是否包含所有宝箱
    path_treasures = [node for node in path[1:-1] if node in TREASURE_POSITIONS]
    missing_treasures = [tid for tid, pos in enumerate(TREASURE_POSITIONS) if pos not in path_treasures]
    if missing_treasures:
        print(f"  警告：路径中缺少宝箱: {missing_treasures}")
    else:
        print(f"  ✓ 路径包含所有10个宝箱")
    
    return total_distance


def main():
    """主函数"""
    print("=" * 60)
    print("峡谷漫步理论最高分计算")
    print("=" * 60)
    
    # 1. 加载地图数据
    print("\n步骤1: 加载地图数据...")
    map_file = "conf/map_data/F_level_1.json"
    try:
        F = load_map_data(map_file)
        print(f"成功加载地图数据，包含 {len(F)} 个位置")
    except Exception as e:
        print(f"错误：无法加载地图数据: {e}")
        return
    
    # 2. 构建状态转移图
    print("\n步骤2: 构建状态转移图...")
    graph = build_transition_graph(F)
    print(f"状态转移图构建完成，包含 {len(graph)} 个位置")
    
    # 3. 定义节点
    start_pos_id = pos_to_id(START_POS[0], START_POS[1])
    end_pos_id = pos_to_id(END_POS[0], END_POS[1])
    nodes = [start_pos_id, end_pos_id] + TREASURE_POSITIONS
    
    print(f"\n节点定义:")
    print(f"  起点: [{START_POS[0]}, {START_POS[1]}] (ID: {start_pos_id})")
    print(f"  终点: [{END_POS[0]}, {END_POS[1]}] (ID: {end_pos_id})")
    for i, treasure_pos in enumerate(TREASURE_POSITIONS):
        pos = id_to_pos(treasure_pos)
        print(f"  宝箱{i}: [{pos[0]}, {pos[1]}] (ID: {treasure_pos})")
    
    # 4. 计算所有节点对之间的最短路径
    print("\n步骤3: 计算所有节点对之间的最短路径...")
    distances = calculate_all_pair_distances(graph, nodes)
    
    # 5. 验证所有节点是否可达
    print("\n步骤4: 验证节点可达性...")
    all_reachable = True
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1, node2 = nodes[i], nodes[j]
            if distances.get((node1, node2), float('inf')) == float('inf'):
                print(f"警告：节点 {node1} 和 {node2} 之间不可达！")
                all_reachable = False
    
    if not all_reachable:
        print("错误：存在不可达的节点对，无法计算理论最高分")
        return
    
    print("所有节点对之间都可达！")
    
    # 6. 求解TSP问题
    print("\n步骤5: 求解TSP问题（寻找最优路径）...")
    best_path, best_distance = solve_tsp_with_fixed_ends(
        distances, start_pos_id, end_pos_id, TREASURE_POSITIONS
    )
    
    if best_path is None:
        print("错误：无法找到可行路径")
        return
    
    # 7. 打印路径详情
    print_path_details(best_path, distances)
    
    # 8. 计算理论最高分
    print("\n" + "=" * 60)
    print("理论最高分计算")
    print("=" * 60)
    
    total_score, end_score, step_score, treasure_score = calculate_score(best_distance, MAX_STEP)
    
    print(f"\n路径信息:")
    print(f"  总步数: {best_distance}步")
    print(f"  最大步数: {MAX_STEP}步")
    print(f"  剩余步数: {MAX_STEP - best_distance}步")
    
    print(f"\n积分计算:")
    print(f"  终点积分: {end_score}分")
    print(f"  步数积分: {step_score:.1f}分 = ({MAX_STEP} - {best_distance}) × 0.2")
    print(f"  宝箱积分: {treasure_score}分 = 10 × 100")
    print(f"  {'-' * 40}")
    print(f"  理论最高分: {total_score:.1f}分")
    
    # 9. 与当前成绩对比
    print("\n" + "=" * 60)
    print("成绩对比")
    print("=" * 60)
    current_score = 1477
    print(f"  当前成绩: {current_score}分")
    print(f"  理论最高分: {total_score:.1f}分")
    print(f"  差距: {total_score - current_score:.1f}分")
    print(f"  完成度: {100 * current_score / total_score:.2f}%")
    
    if abs(current_score - total_score) < 1:
        print("\n🎉 恭喜！你的成绩已经达到理论最高分！")
    elif current_score >= total_score * 0.99:
        print("\n✨ 非常接近理论最高分！")
    else:
        print(f"\n💡 还有 {total_score - current_score:.1f} 分的提升空间")
    
    # 10. 输出最优路径序列
    print("\n" + "=" * 60)
    print("最优路径序列（宝箱收集顺序）")
    print("=" * 60)
    treasure_order = []
    for node in best_path[1:-1]:  # 排除起点和终点
        if node in TREASURE_POSITIONS:
            treasure_id = TREASURE_POSITIONS.index(node)
            pos = id_to_pos(node)
            treasure_order.append(treasure_id)
            print(f"  宝箱{treasure_id}: [{pos[0]}, {pos[1]}]")
    
    print(f"\n宝箱收集顺序: {treasure_order}")
    
    print("\n" + "=" * 60)
    print("计算完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

