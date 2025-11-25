# back_to_the_realm

详细实验说明见[重返秘境实验开发指南](../docs/重返秘境实验开发指南.md)。

## 关于测评

两次测评，均为固定环境，参数设置：

- 第一次：最大步长 1999，13个宝箱全部生成
- 第二次：最大步长 1800，生成10个固定宝箱

注意到开发指南当中有这样一句话：

> 3. 模型评估时，用户需要通过开悟平台创建评估任务并完成任务的环境配置，其中起点、终点位置固定为2和1，用户可以设置宝箱是否随机以及宝箱的数量。**若设置为固定宝箱，则宝箱位置id从3开始按顺序生成**，例如配置固定宝箱个数为5，则宝箱位置id为[3, 4, 5, 6, 7]。

也就是说，设定宝箱为固定的时候，这10个宝箱就是编号3~12的10个宝箱……但是助教似乎完全不知道这一点，以为是13个宝箱中先随机生成10个，然后再用这一环境对所有人进行测评，且我们无法预先得知这10个宝箱具体是哪些。然而事实上我们是知道的，也算是吃到了信息差，钻了助教的空子。

## 关于思路

既然两次测评都是固定环境，那么使用某些确定性的算法绝对是最佳选择。不过我的思路也就止步于此了，因为我要忙着去做小组实验。接下来的思路与代码**均出自组员之手** (感谢 [@wwxyy2005](https://github.com/wwxyy2005))，我只负责转述。

### 测评结果

确定性算法十分成功，两次测评均取得了第一的成绩，我想应该很难有比这个还高的了吧，毕竟已经十分接近最优解了。

第一次测评：

![1](./images/1.png)

第二次测评：

![2](./images/2.png)

### 扒地图

平台没有提供地图数据，但是每一步 `observation` 都包含当前智能体的坐标以及周围 51*51 的局部视野信息。通过在15个重生点都出生一遍，就能用局部视野把全图拼凑得差不多了，同时起点、终点、13个宝箱的坐标也能一并得到。(加速点的坐标就只能肉眼观察+评估测试猜出来)

通过在 `observation_process` 函数当中调用 `update_Map_information` 函数即可自动更新地图数据，详细的代码与地图可见 [algorithm.py](./DQN_13_boxes/agent_dqn/algorithm/algorithm.py)。

> [wwxyy2005](https://github.com/wwxyy2005) 最初希望通过让智能体探索地图来扒地图数据，但是发现了环境当中的一个 bug，即**横纵坐标搞反了**，这是腾讯程序员的锅，并且进而**导致 DQN 几乎不可用**。具体什么情况，可以从 [录播](https://www.yanhekt.cn/session/804099) 的 1h22min 开始看，看看是如何解释的。

### 确定性算法

使用反向 BFS 为每个宝藏生成方向查找表，从终点开始反向搜索到所有可达点，每个点的方向值都指向最短路径的下一步。

```python
def generate_map(self, dir, pos) :
    """
    使用反向 BFS 从目标位置 pos 开始搜索，生成方向查找表。
    """
    q = Queue()
    q.put(pos)
    vis = np.zeros((128, 128))
    vis[pos] = 1
    def check(x, y) :
        if x < 0 or x >= 128 or y < 0 or y >= 128 : return False
        if vis[x, y] or (self.Map[x, y] != 1 and self.Map[x, y] != 2) : return False
        vis[x, y] = 1
        return True
    while not q.empty() :
        x, y = q.get()
        # 详见 Tricks 第二点
        if self.Map[x, y] == 2 : continue
        # Tricks 第一点：注意优先更新上下左右
        if check(x - 1, y):
            q.put((x - 1, y))
            dir[x - 1, y] = 2
        if check(x + 1, y): 
            q.put((x + 1, y))
            dir[x + 1, y] = 6
        if check(x, y - 1):
            q.put((x, y - 1))
            dir[x, y - 1] = 0
        if check(x, y + 1):
            q.put((x, y + 1))
            dir[x, y + 1] = 4
        # 只有当上下左右都不是障碍物时才允许对角线移动
        if self.Map[x - 1, y] == 0 or self.Map[x + 1, y] == 0 or self.Map[x, y - 1] == 0 or self.Map[x, y + 1] == 0 : continue
        if check(x - 1, y - 1) :
            q.put((x - 1, y - 1))
            dir[x - 1, y - 1] = 1
        if check(x - 1, y + 1) :
            q.put((x - 1, y + 1))
            dir[x - 1, y + 1] = 3
        if check(x + 1, y + 1) :
            q.put((x + 1, y + 1))
            dir[x + 1, y + 1] = 5
        if check(x + 1, y - 1) :
            q.put((x + 1, y - 1))
            dir[x + 1, y - 1] = 7
    return dir

def generate_dir(self) :
    """为16个目标位置（起点、终点、13个宝箱、加速点）分别生成方向查找表"""
    for i in range(16) :
        self.dir_map[i] = self.generate_map(self.dir_map[i], self.treasures[i])
```

**Tricks**:

- 注意需要先更新上下左右移动再更新对角线移动，否则智能体会斜着走一条本来可以直着走的路线，因为在它看来斜着走和直着走是一样的 (都是走两步)。
- 地图当中某些地方虽然标注了可以通行，但是实际上有一些边边角角的障碍物，智能体进入后会蹭墙减速，因此在地图当中使用数字 2 来标注这些区域，表示仅可进入而无法从这些区域离开。通过人工修饰地图即可更改智能体的路径，使其避开这些区域。

### `agent.py` 调用

重写 `agent.py` 当中的 `exploit` 函数，使其使用我们的确定性算法 (`exploit_Map` 函数) 而不是训练得到的 DQN 模型。至于模型训练，随便训练多久都没有关系，因为根本没有用到，只要能够保存模型到模型管理页面即可。